# Fitmotion Prediction Service

## Quick setup
1. Create new `.env` file and add Cloud Storage JSON service account key location as `GOOGLE_APPLICATION_CREDENTIALS` variable

2. Install the dependencies
```bash
pip install -r requirements.txt
pip install onnxruntime

# You can also use pipenv instead
pipenv shell
pipenv install
```

3. Start the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```
