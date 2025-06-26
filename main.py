@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        print("Step 1: File received for upload")
        # Step 1: Save uploaded file locally
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = f"/tmp/{filename}"
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        print(f"Step 2: File saved locally as {filepath}")

        # Step 2: Upload file to Azure Blob Storage
        blob_url = f"{AZURE_STORAGE_BASE_URL}/{filename}?{AZURE_BLOB_SAS_TOKEN}"
        with open(filepath, "rb") as data:
            upload_headers = {"x-ms-blob-type": "BlockBlob"}
            upload_response = requests.put(blob_url, data=data, headers=upload_headers)

        print(f"Step 3: Blob upload response status: {upload_response.status_code}")

        if upload_response.status_code not in [201, 200]:
            return {"error": f"Blob upload failed with status: {upload_response.status_code}"}

        # Step 3: Submit transcription job
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
        response = requests.post(transcription_url, headers=headers, json=transcription_payload)

        print(f"Step 4: Transcription job submission status: {response.status_code}")

        if response.status_code != 202:
            return {"error": f"Transcription submission failed. Status: {response.status_code}"}

        transcription_location = response.headers["Location"]
        print(f"Step 5: Transcription job location: {transcription_location}")

        # Poll for completion
        while True:
            poll_response = requests.get(transcription_location, headers=headers)
            status = poll_response.json().get("status")
            print(f"Polling transcription status: {status}")
            if status in ["Succeeded", "Failed"]:
                break
            time.sleep(5)

        if status == "Failed":
            return {"error": "Azure transcription failed."}

        # Get transcription result
        files_url = poll_response.json()["links"]["files"]
        files_list = requests.get(files_url, headers=headers).json()
        transcript_file_url = next(
            (file["links"]["contentUrl"] for file in files_list["values"] if file["kind"] == "Transcription"),
            None
        )

        print(f"Step 6: Transcript file URL: {transcript_file_url}")

        if not transcript_file_url:
            return {"error": "Transcript file not found."}

        transcript_text = requests.get(transcript_file_url).text
        print(f"Step 7: Transcript text retrieved")

        # Generate AI Outputs (SOAP, referral, codes)
        def generate_gpt_output(prompt):
            print(f"Sending prompt to OpenAI: {prompt[:60]}...")
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

        print("Step 8: AI outputs generated successfully.")

        return {
            "transcript": transcript_text,
            "soap_note": soap_note,
            "referral_letter": referral_letter,
            "codes": codes
        }

    except Exception as e:
        print(f"Server error: {str(e)}")
        return {"error": str(e)}
