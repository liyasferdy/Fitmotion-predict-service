from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import storage
from google.auth import exceptions as google_exceptions
from fastapi.responses import StreamingResponse
from io import BytesIO
import pandas as pd
from numpy import array, argmax, unique
import onnxruntime as ort
from time import time
import os

# Initialize the FastAPI app
app = FastAPI()

# Set up Google Cloud Storage client
client = storage.Client()
# bucket_name = "fitmotiion-imu-log"
bucket_name = "fitmotion-imu-sensor"
bucket = client.bucket(bucket_name)

# Constants for the prediction model
WINDOW_LENGTH = 150
STRIDE_LENGTH = 10
LABELS = ['dws', 'jog', 'sit', 'std', 'ups', 'wlk']

# Function to generate sequences from the data
def sequence_generator(x, length, stride):
    seq_x = []
    data_length = len(x)

    for i in range(0, data_length - length + 1, stride):
        input_sequence = x.iloc[i : i + length]
        seq_x.append(input_sequence)
    return array(seq_x).astype('float32')

# Function to classify activities using the ONNX model
async def classify(b_file: bytes, runtime=ort.InferenceSession('fitmotion_model.onnx')):
    start = time()
    csv_file = BytesIO(b_file)
    df = pd.read_csv(csv_file)
    df = df.drop(labels=['Unnamed: 0'], axis=1)
    feat = sequence_generator(df, WINDOW_LENGTH, STRIDE_LENGTH)
    
    # predictor
    input_name = runtime.get_inputs()[0].name
    output_name = runtime.get_outputs()[0].name
    y_pred = runtime.run([output_name], {input_name: feat})[0]
    y_pred = argmax(y_pred, axis=1)
    
    val, count = unique(y_pred, return_counts=True)
    index = val[argmax(count)]
    end = (time() - start) * 1000
    return LABELS[index], round(end, 2)

#get data from storage
# @app.get("/download-csv/{filename}")
# async def download_csv(filename: str):
#     if not filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Only .csv files are allowed.")
    
#     try:
#         blob = bucket.blob(filename)
#         if not blob.exists():
#             raise HTTPException(status_code=404, detail="File not found.")
        
#         stream = BytesIO()
#         blob.download_to_file(stream)
#         stream.seek(0)
#         return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
#     except google_exceptions.GoogleAuthError as e:
#         raise HTTPException(status_code=500, detail="Authentication error with Google Cloud.") from e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


#predict data
@app.get("/predict-csv/{filename}")
async def predict_csv(filename: str):
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed.")

    try:
        blob = bucket.blob(filename)
        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found.")

        stream = BytesIO()
        blob.download_to_file(stream)
        stream.seek(0)
        b_file = stream.read()

        # Perform the prediction
        label, time_taken = await classify(b_file)
        return {"label": label, "time_taken_ms": time_taken}
    except google_exceptions.GoogleAuthError as e:
        raise HTTPException(status_code=500, detail="Authentication error with Google Cloud.") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
