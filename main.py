import os
import uuid
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional
import aiohttp
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from openai import AsyncAzureOpenAI
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="HIPAA Compliant Chiropractic AI Assistant")

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "audio-files")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")

# Validate required environment variables
required_env_vars = [
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_SPEECH_KEY", 
    "AZURE_SPEECH_REGION",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY"
]

for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

# Configure Azure OpenAI client
openai_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

class TranscriptionResponse(BaseModel):
    transcript: str
    soap_note: str
    referral_letter: str
    codes: dict

@app.get("/")
async def health_check():
    """Health check endpoint for Render deployment"""
    return {"status": "healthy", "service": "Chiropractic AI Assistant"}

@app.post("/upload", response_model=TranscriptionResponse)
async def upload_and_process(file: UploadFile = File(...)):
    """
    Upload audio file, transcribe, and generate clinical documents
    """
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload audio files only.")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        blob_name = f"{file_id}_{file.filename}"
        
        logger.info(f"Uploading to Azure Blob: {blob_name}")
        
        # Upload to Azure Blob Storage
        blob_url = await upload_to_azure_blob(file, blob_name)
        
        logger.info("Starting transcription")
        
        # Start transcription
        transcript = await transcribe_audio(blob_url)
        
        logger.info("Generating clinical documents")
        
        # Generate clinical documents using Azure OpenAI
        soap_note = await generate_soap_note(transcript)
        referral_letter = await generate_referral_letter(transcript)
        codes = await generate_billing_codes(transcript)
        
        # Clean up blob (optional - you may want to keep for audit trail)
        # await delete_azure_blob(blob_name)
        
        logger.info("Processing completed successfully")
        
        return TranscriptionResponse(
            transcript=transcript,
            soap_note=soap_note,
            referral_letter=referral_letter,
            codes=codes
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def upload_to_azure_blob(file: UploadFile, blob_name: str) -> str:
    """Upload file to Azure Blob Storage and return blob URL"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_CONTAINER_NAME, 
            blob=blob_name
        )
        
        # Read file content
        file_content = await file.read()
        
        # Upload file
        await blob_client.upload_blob(file_content, overwrite=True)
        
        # Generate SAS token for secure access
        sas_token = generate_blob_sas(
            account_name=blob_client.account_name,
            container_name=AZURE_STORAGE_CONTAINER_NAME,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        
        blob_url = f"{blob_client.url}?{sas_token}"
        logger.info(f"Blob uploaded successfully: {blob_url}")
        return blob_url
        
    except Exception as e:
        logger.error(f"Blob upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Blob upload failed: {str(e)}")

async def transcribe_audio(blob_url: str) -> str:
    """Transcribe audio using Azure Speech-to-Text"""
    try:
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_SPEECH_KEY,
            'Content-Type': 'application/json'
        }
        
        # Start transcription
        transcription_request = {
            'contentUrls': [blob_url],
            'properties': {
                'diarizationEnabled': True,
                'wordLevelTimestampsEnabled': True,
                'punctuationMode': 'DictatedAndAutomatic',
                'profanityFilterMode': 'Masked'
            },
            'locale': 'en-US',
            'displayName': 'Chiropractic Consultation Transcription'
        }
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Submit transcription job
            async with session.post(
                f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions",
                headers=headers,
                json=transcription_request
            ) as response:
                if response.status != 201:
                    error_text = await response.text()
                    logger.error(f"Failed to start transcription: {response.status} - {error_text}")
                    raise HTTPException(status_code=500, detail=f"Failed to start transcription: {error_text}")
                
                transcription_response = await response.json()
                transcription_id = transcription_response['self'].split('/')[-1]
                logger.info(f"Transcription started with ID: {transcription_id}")
            
            # Poll for completion
            for attempt in range(60):  # 5-minute timeout
                await asyncio.sleep(5)
                
                async with session.get(
                    f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions/{transcription_id}",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        continue
                        
                    status_response = await response.json()
                    logger.info(f"Transcription status: {status_response['status']}")
                    
                    if status_response['status'] == 'Succeeded':
                        # Get transcription files
                        async with session.get(
                            f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions/{transcription_id}/files",
                            headers=headers
                        ) as files_response:
                            if files_response.status != 200:
                                continue
                                
                            files_data = await files_response.json()
                            
                            # Find the transcription result file
                            for file_info in files_data['values']:
                                if file_info['kind'] == 'Transcription':
                                    async with session.get(file_info['links']['contentUrl']) as content_response:
                                        if content_response.status != 200:
                                            continue
                                            
                                        transcription_result = await content_response.json()
                                        
                                        # Extract combined text
                                        combined_text = ""
                                        for phrase in transcription_result.get('combinedRecognizedPhrases', []):
                                            combined_text += phrase.get('display', '') + " "
                                        
                                        result = combined_text.strip()
                                        logger.info(f"Transcription completed: {len(result)} characters")
                                        return result or "No transcription available"
                    
                    elif status_response['status'] == 'Failed':
                        error_msg = status_response.get('error', {}).get('message', 'Unknown error')
                        logger.error(f"Transcription failed: {error_msg}")
                        raise HTTPException(status_code=500, detail=f"Transcription failed: {error_msg}")
            
            logger.error("Transcription timeout")
            raise HTTPException(status_code=408, detail="Transcription timeout")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

async def generate_soap_note(transcript: str) -> str:
    """Generate SOAP note using Azure OpenAI"""
    prompt = f"""
    Based on the following chiropractic consultation transcript, create a professional SOAP note:

    Transcript: {transcript}

    Please format as:

    SUBJECTIVE:
    [Patient's chief complaint, history of present illness, pain scale, activities affected]

    OBJECTIVE:
    [Physical examination findings, range of motion, orthopedic tests, posture analysis]

    ASSESSMENT:
    [Clinical impression, diagnostic considerations]

    PLAN:
    [Treatment plan, patient education, follow-up recommendations]

    Keep it concise and professional. Only include information that can be reasonably inferred from the transcript.
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a professional chiropractic documentation assistant. Create accurate, concise SOAP notes based on consultation transcripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"SOAP note generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SOAP note generation failed: {str(e)}")

async def generate_referral_letter(transcript: str) -> str:
    """Generate referral letter if clinically appropriate"""
    prompt = f"""
    Based on this chiropractic consultation transcript, determine if a referral is clinically appropriate and generate a professional referral letter if needed:

    Transcript: {transcript}

    If a referral is NOT needed, respond with: "No referral indicated based on current presentation."

    If a referral IS needed, format as a professional referral letter with:
    - Date
    - Referring provider information
    - Patient information
    - Reason for referral
    - Relevant clinical findings
    - Specific questions or services requested

    Base your decision on standard chiropractic practice guidelines for when referrals are appropriate (red flags, complex conditions, lack of improvement, etc.).
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a chiropractic clinical decision support assistant. Generate appropriate referral letters only when clinically indicated."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Referral letter generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Referral letter generation failed: {str(e)}")

async def generate_billing_codes(transcript: str) -> dict:
    """Generate suggested CPT and ICD-10 codes"""
    prompt = f"""
    Based on this chiropractic consultation transcript, suggest appropriate billing codes:

    Transcript: {transcript}

    Provide your response in this exact JSON format:
    {{
        "cpt_codes": [
            {{"code": "99213", "description": "Office visit, established patient, low complexity"}},
            {{"code": "98940", "description": "Chiropractic manipulative treatment (CMT), spinal, 1-2 regions"}}
        ],
        "icd10_codes": [
            {{"code": "M54.5", "description": "Low back pain"}},
            {{"code": "M25.511", "description": "Pain in right shoulder"}}
        ]
    }}

    Only suggest codes that are clearly supported by the documented findings. Use standard chiropractic CPT codes (99xxx for E&M, 98xxx for CMT) and relevant ICD-10 codes for the documented conditions.
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a medical coding assistant specialized in chiropractic billing. Provide accurate CPT and ICD-10 code suggestions in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        
        # Parse JSON response
        codes_text = response.choices[0].message.content.strip()
        # Remove any markdown formatting
        if codes_text.startswith('```json'):
            codes_text = codes_text.replace('```json', '').replace('```', '').strip()
        
        return json.loads(codes_text)
        
    except Exception as e:
        logger.error(f"Code generation failed: {str(e)}", exc_info=True)
        # Return fallback structure if parsing fails
        return {
            "cpt_codes": [{"code": "Error", "description": f"Code generation failed: {str(e)}"}],
            "icd10_codes": [{"code": "Error", "description": "Please review transcript manually"}]
        }

async def delete_azure_blob(blob_name: str):
    """Delete blob from Azure Storage (optional cleanup)"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_CONTAINER_NAME, 
            blob=blob_name
        )
        await blob_client.delete_blob()
    except Exception as e:
        logger.warning(f"Failed to delete blob {blob_name}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
