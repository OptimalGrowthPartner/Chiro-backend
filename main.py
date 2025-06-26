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
        print("Step 1: File received.")
        filename = f"{uuid.uuid4()}_{file.filename}"
        local_path = f"/tmp/{filename}"
        with open(local_path, "wb") as f:
            f.write(await file.read())
        print(f"Step 2: File saved locally at {local_path}")

        # Upload to Azure Blob Storage
        blob_url = f"{AZURE_STORAGE_BASE_URL}/{filename}?{AZURE_BLOB_SAS_TOKEN}"
        with open(local_path, "rb") as data:
            blob_response = requests.put(blob_url, headers={"x-ms-blob-type": "BlockBlob"}, data=data)
        print(f"Step 3: Blob upload status: {blob_response.status_code}")

        if blob_response.status_code not in [200, 201]:
            return {"error": f"Blob upload failed: {blob_response.text}"}

        # Start Azure Speech Transcription Job
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
        print(f"Step 4: Transcription job submission status: {create_response.status_code}")

        if create_response.status_code != 202:
            return {"error": f"Transcription job creation failed: {create_response.text}"}

        transcription_location = create_response.headers["Location"]
        print(f"Step 5: Transcription Location URL: {transcription_location}")

        # Poll transcription job status with a timeout
        max_wait_time = 120  # 2 minutes
        start_time = time.time()

        while True:
            poll_response = requests.get(transcription_location, headers=headers)
            status = poll_response.json().get("status")
            print(f"Polling status: {status}")

            if status == "Succeeded":
                break
            if status == "Failed":
                return {"error": "Azure transcription job failed"}

            if time.time() - start_time > max_wait_time:
                return {"error": "Azure transcription timed out after 2 minutes"}

            time.sleep(5)

        # Fetch transcript file URL
        files_url = poll_response.json()["links"]["files"]
        files_list = requests.get(files_url, headers=headers).json()
        transcript_file_url = next(
            (f["links"]["contentUrl"] for f in files_list["values"] if f["kind"] == "Transcription"),
            None
        )

        if not transcript_file_url:
            return {"error": "No transcript file found"}

        transcript_text = requests.get(transcript_file_url).text
        print("Step 6: Transcript text downloaded.")

        # Generate AI Outputs with GPT
        def ask_gpt(prompt):
            print(f"Sending GPT prompt: {prompt[:50]}...")
            chat_response = openai.ChatCompletion.create(
                engine=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a medical assistant helping chiropractors generate SOAP notes, referral letters, and billing codes."},
                    {"role": "user", "content": prompt}
                ]
            )
            return chat_response.choices[0].message.content.strip()

        soap_note = ask_gpt(f"Generate a SOAP note for this transcript:\n{transcript_text}")
        referral_letter = ask_gpt(f"Write a professional referral letter for this patient visit:\n{transcript_text}")
        codes = ask_gpt(f"Suggest CPT and ICD-10 codes for this transcript:\n{transcript_text}")

        print("Step 7: AI outputs generated successfully.")

        return {
            "transcript": transcript_text,
            "soap_note": soap_note,
            "referral_letter": referral_letter,
            "codes": codes
        }

    except Exception as e:
        print(f"Server error: {str(e)}")
        return {"error": str(e)}

# Basic health check route to prevent shutdown
@app.get("/")
def health_check():
    return {"status": "Backend running"}
