from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import time
import requests
import openai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Environment Variables
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_STORAGE_BASE_URL = os.getenv("AZURE_STORAGE_BASE_URL")
AZURE_BLOB_SAS_TOKEN = os.getenv("AZURE_BLOB_SAS_TOKEN")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# OpenAI Azure Config
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-12-01-preview"
openai.api_key = AZURE_OPENAI_KEY

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Save file locally
        filename = f"{uuid.uuid4()}_{file.filename}"
        local_path = f"/tmp/{filename}"
        with open(local_path, "wb") as f:
            f.write(await file.read())

        # Upload to Azure Blob
        blob_url = f"{AZURE_STORAGE_BASE_URL}/{filename}?{AZURE_BLOB_SAS_TOKEN}"
        with open(local_path, "rb") as data:
            upload_headers = {"x-ms-blob-type": "BlockBlob"}
            blob_response = requests.put(blob_url, headers=upload_headers, data=data)
            if blob_response.status_code not in [201, 200]:
                return {"error": f"Blob upload failed: {blob_response.text}"}

        # Submit Transcription Job
        transcription_payload = {
            "contentUrls": [blob_url.split("?")[0]],
            "locale": "en-US",
            "displayName": "Chiro Transcription Job"
        }
        transcription_url = f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "application/json"
        }
        create_response = requests.post(transcription_url, json=transcription_payload, headers=headers)
        if create_response.status_code != 202:
            return {"error": f"Transcription job creation failed: {create_response.text}"}

        # Poll for Transcription Completion
        transcription_location = create_response.headers["Location"]
        while True:
            status_check = requests.get(transcription_location, headers=headers).json()
            status = status_check.get("status")
            if status in ["Succeeded", "Failed"]:
                break
            time.sleep(5)

        if status == "Failed":
            return {"error": "Azure transcription job failed"}

        # Get Transcript File Link
        files_url = status_check["links"]["files"]
        files_list = requests.get(files_url, headers=headers).json()
        transcript_file_url = next(
            (file["links"]["contentUrl"] for file in files_list["values"] if file["kind"] == "Transcription"),
            None
        )

        if not transcript_file_url:
            return {"error": "Transcript file not found"}

        transcript_text = requests.get(transcript_file_url).text

        # Send transcript to Azure OpenAI (GPT)
        def ask_gpt(prompt):
            chat_response = openai.ChatCompletion.create(
                engine=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a medical assistant that helps chiropractors summarize patient visits."},
                    {"role": "user", "content": prompt}
                ]
            )
            return chat_response.choices[0].message.content.strip()

        soap_note = ask_gpt(f"Generate a SOAP note for the following transcript:\n{transcript_text}")
        referral_letter = ask_gpt(f"Write a referral letter based on this transcript:\n{transcript_text}")
        codes = ask_gpt(f"Suggest ICD-10 and CPT codes for this transcript:\n{transcript_text}")

        return {
            "transcript": transcript_text,
            "soap_note": soap_note,
            "referral_letter": referral_letter,
            "codes": codes
        }

    except Exception as e:
        return {"error": str(e)}
