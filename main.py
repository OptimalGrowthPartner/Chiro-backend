from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel
import uuid
import requests
import os
import time

app = FastAPI()

# Allow front-end to access backend (adjust for production later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure settings (set your own in environment variables)
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = "audiofiles"

blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
container_client = blob_service.get_container_client(AZURE_BLOB_CONTAINER)

class TranscriptionResponse(BaseModel):
    transcript: str

@app.post("/upload", response_model=TranscriptionResponse)
async def upload_audio(file: UploadFile = File(...)):
    # Step 1: Save to Azure Blob
    file_id = str(uuid.uuid4())
    blob_name = f"{file_id}_{file.filename}"
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = await file.read()
    blob_client.upload_blob(blob_data, overwrite=True)

    blob_url = f"https://{blob_service.account_name}.blob.core.windows.net/{AZURE_BLOB_CONTAINER}/{blob_name}"

    # Step 2: Send to Azure Speech-to-Text
    transcription_url = f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "contentUrls": [blob_url],
        "locale": "en-US",
        "displayName": f"Chiro Voice - {file.filename}"
    }
    response = requests.post(transcription_url, json=payload, headers=headers)
    if response.status_code != 201:
        raise HTTPException(status_code=500, detail="Failed to start transcription.")

    transcription_status_url = response.json()["self"]

    # Step 3: Poll until transcription is done
    for _ in range(30):
        time.sleep(10)
        status_response = requests.get(transcription_status_url, headers=headers)
        status_data = status_response.json()
        if status_data.get("status") == "Succeeded":
            files_url = status_data["links"]["files"]
            files_response = requests.get(files_url, headers=headers)
            files = files_response.json()["values"]
            transcription_file = next((f for f in files if f["kind"] == "Transcription"), None)
            if transcription_file:
                content_url = transcription_file["links"]["contentUrl"]
                content_response = requests.get(content_url, headers=headers)
                phrases = content_response.json().get("combinedRecognizedPhrases", [])
                transcript = phrases[0].get("display", "") if phrases else ""
                return {"transcript": transcript}
        elif status_data.get("status") == "Failed":
            raise HTTPException(status_code=500, detail="Transcription failed.")

    raise HTTPException(status_code=504, detail="Transcription timed out.")
