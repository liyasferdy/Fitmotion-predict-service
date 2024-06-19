from fastapi import FastAPI, HTTPException
from google.cloud import storage
from google.auth import exceptions as google_exceptions
from io import BytesIO
import pandas as pd
import numpy as np
import onnxruntime as rt
from time import time
import os
from scipy.signal import butter, lfilter

# Initialize the FastAPI app
app = FastAPI()

# Set up Google Cloud Storage client
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/fitmotion/Fitmotion-predict-service/service-account-key.json"  # Update the path
client = storage.Client()
bucket_name = "fitmotion-imu-sensor"
bucket = client.bucket(bucket_name)

# Constants for the prediction model
WINDOW_LENGTH = 150
STRIDE_LENGTH = 10
LABELS = ['sit', 'dws', 'std', 'ups', 'wlk']

MAPPED_FEATURES = ['gravityx', 'gravityy', 'gravityz', 'useraccelerationx', 'useraccelerationy', 'useraccelerationz']
FEATURES = ['Gravity X', 'Gravity Y', 'Gravity Z', 'User Acceleration X', 'User Acceleration Y', 'User Acceleration Z']

FS = 200
ORDER = 2
CUTOFF = 30
LOOKBACK = 20

# Filter data with lowpass-filter butterworth
def butter_filter(data, cutoff, fs, order=5, filter_type='low'):
    nyq = 0.5 * fs
    if isinstance(cutoff, list):
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    y = lfilter(b, a, data)
    return y

# Preprocess data
def preproc(data):
    transformed = []
    df = data[FEATURES]
    
    buffer = [df.iloc[0].values]
    for i in range(1, len(df)):
        if not np.array_equal(df.iloc[i].values, df.iloc[i-1].values):
            buffer.append(df.iloc[i].values)
    buffer = np.array(buffer)

    df_buff = pd.DataFrame(buffer, columns=FEATURES)

    for column in FEATURES:
        if 'User Acceleration' in column:
            df_buff[column] = butter_filter(df_buff[column], CUTOFF, FS, ORDER)

        for i in range(LOOKBACK):
            df_buff[f'{column} (t-{i+1})'] = df_buff[column].shift(i+1)

    df_buff_clean = df_buff.dropna().reset_index(drop=True)
    transformed.append(df_buff_clean)

    df_cleaned = pd.concat(transformed, ignore_index=True)
    return df_cleaned

# Load the model
model_path = "fitmotion_model_rfc.onnx"  # Update the path
sess = rt.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# Perform classification
async def classify(file_content):
    df = pd.read_csv(BytesIO(file_content))
    cleaned = preproc(df)
    
    if cleaned.empty:
        raise ValueError("No valid data after preprocessing")

    start_time = time()
    predictions = sess.run([label_name], {input_name: cleaned.to_numpy()})
    time_taken = (time() - start_time) * 1000

    y_pred = predictions[0]

    ranged = {
        'sit': 0,
        'dws': 0,
        'std': 0,
        'ups': 0,
        'wlk': 0
    }

    ranged_calc = {
        'sit': 0,
        'dws': 0,
        'std': 0,
        'ups': 0,
        'wlk': 0
    }

    for data in y_pred:
        label = LABELS[data]
        ranged[label] += 1

    for item in ranged:
        ranged_calc[item] = ranged[item] / (15 * 60)

    return ranged_calc, time_taken

# Predict data
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
        ranged_calc, time_taken = await classify(b_file)
        return {"ranged_calc": ranged_calc, "time_taken_ms": time_taken}
    except google_exceptions.GoogleAuthError as e:
        raise HTTPException(status_code=500, detail="Authentication error with Google Cloud.") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
