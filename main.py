from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import uuid
import os
import time
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS setup (allow frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your Vercel frontend URL
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

# Setup OpenAI Azure API
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-12-01-preview"
openai.api_key = AZURE_OPENAI_KEY

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Step 1: Save uploaded file locally
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = f"/tmp/{filename}"
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        # Step 2: Upload file to Azure Blob Storage
        blob_url = f"{AZURE_STORAGE_BASE_URL}/{filename}?{AZURE_BLOB_SAS_TOKEN}"
        with open(filepath, "rb") as data:
            upload_headers = {"x-ms-blob-type": "BlockBlob"}
            upload_response = requests.put(blob_url, data=data, headers=upload_headers)

        if upload_response.status_code not in [201, 200]:
            return {"error": f"Blob upload failed with status: {upload_response.status_code}"}

        # Step 3: Submit audio for transcription
        transcription_url = f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions"
        transcription_id = str(uuid.uuid4())
        transcription_payload = {
            "contentUrls": [blob_url.split('?')[0]],
            "locale": "en-US",
            "displayName": "Chiro Upload"
        }
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "application/json"
        }
        response = requests.post(transcription_url, headers=headers, json=transcription_payload)

        if response.status_code != 202:
            return {"error": f"Transcription submission failed. Status: {response.status_code}"}

        transcription_location = response.headers["Location"]

        # Step 4: Poll for transcription completion
        while True:
            poll_response = requests.get(transcription_location, headers=headers)
            status = poll_response.json().get("status")
            if status in ["Succeeded", "Failed"]:
                break
            time.sleep(5)

        if status == "Failed":
            return {"error": "Azure transcription failed."}

        # Step 5: Get transcription result
        files_url = poll_response.json()["links"]["files"]
        files_list = requests.get(files_url, headers=headers).json()
        transcript_file_url = next(
            (file["links"]["contentUrl"] for file in files_list["values"] if file["kind"] == "Transcription"),
            None
        )

        if not transcript_file_url:
            return {"error": "Transcript file not found."}

        transcript_text = requests.get(transcript_file_url).text

        # Step 6: Generate AI Outputs (SOAP Note, Referral Letter, Codes) using Azure OpenAI
        def generate_gpt_output(prompt):
            completion = openai.ChatCompletion.create(
                engine=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a medical assistant that helps chiropractors summarize patient visits."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content.strip()

        soap_note = generate_gpt_output(f"Generate a SOAP note from this transcript:\n{transcript_text}")
        referral_letter = generate_gpt_output(f"Write a professional referral letter from this patient visit transcript:\n{transcript_text}")
        codes = generate_gpt_output(f"Suggest appropriate CPT and ICD-10 codes based on this conversation:\n{transcript_text}")

        return {
            "transcript": transcript_text,
            "soap_note": soap_note,
            "referral_letter": referral_letter,
            "codes": codes
        }

    except Exception as e:
        return {"error": str(e)}
