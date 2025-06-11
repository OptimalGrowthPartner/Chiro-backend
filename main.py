from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import uuid
import os
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

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_STORAGE_BASE_URL = os.getenv("AZURE_STORAGE_BASE_URL")
AZURE_BLOB_SAS_TOKEN = os.getenv("AZURE_BLOB_SAS_TOKEN")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = "gpt-35-turbo"

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-12-01-preview"
openai.api_key = AZURE_OPENAI_KEY

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    # Save file locally
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = f"/tmp/{filename}"
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Upload to Azure Blob Storage
    blob_url = f"{AZURE_STORAGE_BASE_URL}/{filename}{AZURE_BLOB_SAS_TOKEN}"
    with open(filepath, "rb") as data:
        requests.put(blob_url, data=data, headers={"x-ms-blob-type": "BlockBlob"})

    # Submit transcription job
    transcription_id = str(uuid.uuid4())
    transcription_payload = {
        "contentUrls": [blob_url.split('?')[0]],
        "locale": "en-US",
        "displayName": "Chiro Upload"
    }
    transcription_url = f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/json"
    }
    response = requests.post(transcription_url, headers=headers, json=transcription_payload)
    transcription_location = response.headers["Location"]

    # Poll for completion
    import time
    while True:
        r = requests.get(transcription_location, headers=headers)
        status = r.json()["status"]
        if status in ["Succeeded", "Failed"]:
            break
        time.sleep(5)

    if status == "Failed":
        return {"error": "Transcription failed."}

    # Get transcript file
    files_url = r.json()["links"]["files"]
    files_list = requests.get(files_url, headers=headers).json()
    transcript_file_url = next(f["links"]["contentUrl"] for f in files_list["values"] if f["kind"] == "Transcription")
    transcript_text = requests.get(transcript_file_url).text

    # Generate AI outputs using Azure OpenAI
    def prompt_gpt(prompt):
        completion = openai.ChatCompletion.create(
            engine=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a medical assistant that helps chiropractors summarize patient visits."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()

    soap_prompt = f"Generate a SOAP note from this transcript:\n{transcript_text}"
    referral_prompt = f"Write a professional referral letter from this patient visit transcript:\n{transcript_text}"
    codes_prompt = f"Suggest appropriate CPT and ICD-10 codes based on this conversation:\n{transcript_text}"

    soap_note = prompt_gpt(soap_prompt)
    referral_letter = prompt_gpt(referral_prompt)
    codes = prompt_gpt(codes_prompt)

    return {
        "transcript": transcript_text,
        "soap_note": soap_note,
        "referral_letter": referral_letter,
        "codes": codes
    }
