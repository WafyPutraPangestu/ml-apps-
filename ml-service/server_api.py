from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from datetime import datetime
from loguru import logger
import threading   
import subprocess  
import os          
logger.add("logs/api_{time}.log", rotation="1 day")
app = FastAPI(title="Ticket Priority ML API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = "ml_model/model.pkl"
VECTORIZER_PATH = "ml_model/vectorizer.pkl"
model = None
vectorizer = None
def load_model_files():
    global model, vectorizer
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info(" Model reloaded successfully into memory")
            print(" Model reloaded successfully into memory")
            return True
        else:
            logger.warning(" Model files not found. Waiting for training.")
            return False
    except Exception as e:
        logger.error(f" Failed to load model: {str(e)}")
        return False
load_model_files()
class TicketInput(BaseModel):
    judul: str
    deskripsi: str
    kategori_gangguan: str
    kategori_pelanggan: str
@app.get("/")
async def root():
    return {
        "service": "Ticket Priority ML API",
        "version": "2.0.0",
        "status": "running" if model else "model_not_loaded",
        "features_used": ["judul", "deskripsi", "kategori_gangguan", "kategori_pelanggan"],
        "endpoints": ["/", "/health", "/predict", "/retrain", "/retrain/status"]
    }
@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
@app.post("/predict")
async def predict(data: TicketInput, x_api_key: str = Header(None)):
    if x_api_key != "rahasia-super-aman-123":
        raise HTTPException(401, "Invalid API Key")
    if not model or not vectorizer:
        if not load_model_files():
            raise HTTPException(503, "Model not loaded. Please train the model first.")
    try:
        text = f"{data.judul} {data.deskripsi} {data.kategori_gangguan} {data.kategori_pelanggan}"
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        confidence = float(model.predict_proba(X).max())
        result = {
            "prioritas": prediction,
            "confidence": round(confidence, 4),
            "input": {
                "judul": data.judul,
                "kategori_gangguan": data.kategori_gangguan
            },
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Prediction: {prediction} ({confidence:.2f})")
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, str(e))
@app.post("/retrain")
async def trigger_retrain(x_api_key: str = Header(None)):
    if x_api_key != "rahasia-super-aman-123":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    def run_training_and_reload():
        try:
            logger.info(" Starting background training...")
            result = subprocess.run(
                ["python", "training.py"], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                logger.info(" Training finished. Reloading model...")
                load_model_files() 
            else:
                logger.error(f" Training failed: {result.stderr}")
        except Exception as e:
            logger.error(f" Background process error: {str(e)}")
    thread = threading.Thread(target=run_training_and_reload)
    thread.start()
    return {
        "message": "Retraining started. Model will be reloaded automatically upon success.",
        "timestamp": datetime.now().isoformat()
    }
@app.get("/retrain/status")
async def retrain_status(x_api_key: str = Header(None)):
    if x_api_key != "rahasia-super-aman-123":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    model_exists = os.path.exists(MODEL_PATH)
    metadata_path = "ml_model/metadata.json"
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except:
            pass
    return {
        "model_exists": model_exists,
        "in_memory": model is not None,
        "last_training": metadata.get("training_date", "Never"),
        "accuracy": metadata.get("accuracy", "N/A"),
        "dataset_size": metadata.get("dataset_size", "N/A")
    }
if __name__ == "__main__":
    import uvicorn
    print("Starting API on http://0.0.0.0:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001)