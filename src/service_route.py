from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException
from google.cloud import storage
from google.auth import exceptions as google_exceptions
from io import BytesIO
import pandas as pd
import onnxruntime as rt
from time import time
from os import getenv
from .data_preprocessing import preproc
from dotenv import load_dotenv
from .model.activites import Records
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

service_router = APIRouter()

load_dotenv('.env')
client = storage.Client()
bucket_name = getenv("BUCKET_NAME")
bucket = client.bucket(bucket_name)

# Constants for the prediction model
WINDOW_LENGTH = 150
STRIDE_LENGTH = 10
LABELS = ['sit', 'dws', 'std', 'ups', 'wlk', 'jog']

model_path = "fitmotion_model.onnx"
sess = rt.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

engine = create_engine(getenv("PG_URL"), pool_size=100, pool_recycle=15)

db = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    yield db()

def classify(file_content):
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
        'wlk': 0,
        'jog': 0
    }
    ranged_calc = {
        'sit': 0,
        'dws': 0,
        'std': 0,
        'ups': 0,
        'wlk': 0,
        'jog': 0
    }
    for data in y_pred:
        label = LABELS[data]
        ranged[label] += 1
    for item in ranged:
        ranged_calc[item] = ranged[item] / (15 * 60)
    return ranged_calc, time_taken

@service_router.get("/predict-csv/{filename}")
def predict_csv(filename: str, db: Session = Depends(get_db)):
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

        ranged_calc, time_taken = classify(b_file)
        user_id = filename.split(".")[0]
        prev_record = db.query(Records).where(Records.fk_user_id==str(user_id), Records.created_at==date.today()).first()
        # print(prev_record.created_at, date.today())
        if prev_record == None:
            record = Records(
                fk_user_id=user_id,
                walk_time_min=ranged_calc['wlk'],
                jogging_time_min=ranged_calc['jog'],
                stand_time_min=ranged_calc['std'],
                sit_time_min=ranged_calc['sit'],
                upstair_time_min=ranged_calc['ups'],
                downstair_time_min=ranged_calc['dws'],
                created_at=date.today()
            )
            db.add(record)
            db.commit()
        else:
            prev_record.walk_time_min += ranged_calc['wlk']
            prev_record.jogging_time_min += ranged_calc['jog']
            prev_record.stand_time_min += ranged_calc['std']
            prev_record.sit_time_min += ranged_calc['sit']
            prev_record.upstair_time_min += ranged_calc['ups']
            prev_record.downstair_time_min += ranged_calc['dws']
            prev_record.updated_at = datetime.now()
            db.commit()
        blob.delete()
        return {"status": "saved", "prediction_time_ms": time_taken}
    except google_exceptions.GoogleAuthError as e:
        raise HTTPException(status_code=500, detail="Authentication error with Google Cloud.") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))