from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import uuid
import os
import openai
import time
from dotenv import load_dotenv

# Load env variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure env vars
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_STORAGE_BASE_URL = os.getenv("AZURE_STORAGE_BASE_URL")
AZURE_BLOB_SAS_TOKEN = os.getenv("AZURE_BLOB_SAS_TOKEN")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo")

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-12-01-preview"
openai.api_key = AZURE_OPENAI_KEY

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        print("Step 1: File received")
        # Save file locally
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = f"/tmp/{filename}"
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        print(f"Step 2: Saved file as {filepath}")

        # Upload to Blob Storage
        blob_url = f"{AZURE_STORAGE_BASE_URL}/{filename}?{AZURE_BLOB_SAS_TOKEN}"
        with open(filepath, "rb") as data:
            upload_headers = {"x-ms-blob-type": "BlockBlob"}
            upload_response = requests.put(blob_url, data=data, headers=upload_headers)

        print(f"Step 3: Blob upload status {upload_response.status_code}")

        if upload_response.status_code not in [200, 201]:
            return {"error": f"Blob upload failed with status {upload_response.status_code}"}

        # Submit transcription job
        transcription_url = f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions"
        transcription_payload = {
            "contentUrls": [blob_url.split('?')[0]],
            "locale": "en-US",
            "displayName": "Chiro Upload"
        }
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "application/json"
        }
        transcribe_response = requests.post(transcription_url, headers=headers, json=transcription_payload)

        print(f"Step 4: Transcription job status {transcribe_response.status_code}")

        if transcribe_response.status_code != 202:
            return {"error": f"Transcription submission failed: {transcribe_response.text}"}

        transcription_location = transcribe_response.headers["Location"]

        # Poll transcription status
        while True:
            poll_response = requests.get(transcription_location, headers=headers)
            status = poll_response.json().get("status")
            print(f"Polling status: {status}")
            if status in ["Succeeded", "Failed"]:
                break
            time.sleep(5)

        if status == "Failed":
            return {"error": "Transcription failed at Azure."}

        # Get transcript
        files_url = poll_response.json()["links"]["files"]
        files_list = requests.get(files_url, headers=headers).json()
        transcript_file_url = next(
            (file["links"]["contentUrl"] for file in files_list["values"] if file["kind"] == "Transcription"),
            None
        )

        if not transcript_file_url:
            return {"error": "Transcript file not found."}

        transcript_text = requests.get(transcript_file_url).text
        print("Step 5: Transcription retrieved")

        # GPT AI Processing
        def gpt_task(prompt):
            print(f"Sending GPT Prompt: {prompt[:60]}...")
            completion = openai.ChatCompletion.create(
                engine=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a medical assistant helping chiropractors write SOAP notes, referral letters, and medical billing codes."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content.strip()

        soap_note = gpt_task(f"Generate a SOAP note from this transcript:\n{transcript_text}")
        referral_letter = gpt_task(f"Write a professional referral letter from this patient visit transcript:\n{transcript_text}")
        codes = gpt_task(f"Suggest appropriate CPT and ICD-10 codes from this conversation:\n{transcript_text}")

        print("Step 6: GPT tasks completed")

        return {
            "transcript": transcript_text,
            "soap_note": soap_note,
            "referral_letter": referral_letter,
            "codes": codes
        }

    except Exception as e:
        print(f"Server error: {str(e)}")
        return {"error": str(e)}
