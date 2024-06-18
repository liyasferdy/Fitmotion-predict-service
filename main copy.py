import os
import requests
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from google.auth import exceptions as google_exceptions
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Set up Google Cloud Storage client
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/fitmotion/Fitmotion-storage-service/service-account-key.json"
client = storage.Client()
bucket_name = "fitmotion-imu-log"
bucket = client.bucket(bucket_name)

# Define the API endpoint URL
endpoint_url = "https://fitmotion-core-api-qoladrxgiq-et.a.run.app/api/v1/predict/"

class FileRequest(BaseModel):
    filename: str

@app.post("/process-file/")
async def process_file(request: FileRequest):
    filename = request.filename
    blob = bucket.blob(filename)
    
    try:
        file_contents = blob.download_as_bytes()
        files = {'file': (filename, file_contents, 'text/csv')}
        response = requests.post(endpoint_url, files=files)
        
        if response.status_code == 200:
            return {"message": "File processed successfully", "response": response.json()}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    
    except google_exceptions.GoogleAuthError as e:
        raise HTTPException(status_code=500, detail="Authentication error with Google Cloud.") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
