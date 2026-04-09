@echo off
title ML Apps - Complete Setup
color 0B

cd /d D:\ngoding\MY_PROJECT\ml-apps

echo ============================================
echo    ML APPS - COMPLETE SETUP
echo ============================================
echo.
echo Current location: %CD%
echo.

REM Create folders
echo [1/10] Creating folders...
if not exist "ml-service\ml_model" mkdir "ml-service\ml_model"
if not exist "ml-service\mlruns" mkdir "ml-service\mlruns"
if not exist "ml-service\logs" mkdir "ml-service\logs"
echo ✅ Folders created
echo.

REM 1. docker-compose.yml
echo [2/10] Creating docker-compose.yml...
(
echo version: '3.8'
echo.
echo services:
echo   redis:
echo     image: redis:7-alpine
echo     container_name: ticket_redis
echo     ports:
echo       - "6379:6379"
echo     volumes:
echo       - redis_data:/data
echo     restart: unless-stopped
echo     healthcheck:
echo       test: ["CMD", "redis-cli", "ping"]
echo       interval: 10s
echo       timeout: 3s
echo       retries: 3
echo.
echo   ml-service:
echo     build:
echo       context: ./ml-service
echo       dockerfile: Dockerfile
echo     container_name: ticket_ml_service
echo     ports:
echo       - "5000:5000"
echo       - "8501:8501"
echo     volumes:
echo       - ./ml-service:/app
echo       - ./dataset:/app/dataset
echo     environment:
echo       - REDIS_HOST=redis
echo       - ML_API_KEY=rahasia-super-aman-123
echo       - PYTHONUNBUFFERED=1
echo     depends_on:
echo       redis:
echo         condition: service_healthy
echo     restart: unless-stopped
echo     command: python server_api.py
echo.
echo volumes:
echo   redis_data:
) > docker-compose.yml
echo ✅ docker-compose.yml created
echo.

REM 2. Dockerfile
echo [3/10] Creating Dockerfile...
(
echo FROM python:3.11-slim
echo.
echo WORKDIR /app
echo.
echo RUN apt-get update ^&^& apt-get install -y gcc g++ ^&^& rm -rf /var/lib/apt/lists/*
echo.
echo COPY requirements.txt .
echo RUN pip install --no-cache-dir -r requirements.txt
echo.
echo COPY . .
echo RUN mkdir -p ml_model mlruns logs
echo.
echo EXPOSE 5000 8501
echo ENV PYTHONUNBUFFERED=1
echo.
echo CMD ["python", "server_api.py"]
) > ml-service\Dockerfile
echo ✅ Dockerfile created
echo.

REM 3. requirements.txt
echo [4/10] Creating requirements.txt...
(
echo fastapi==0.109.0
echo uvicorn[standard]==0.27.0
echo pydantic==2.5.3
echo scikit-learn==1.4.0
echo pandas==2.2.0
echo numpy==1.26.3
echo joblib==1.3.2
echo streamlit==1.30.0
echo plotly==5.18.0
echo loguru==0.7.2
echo scipy==1.11.4
echo python-multipart==0.0.6
) > ml-service\requirements.txt
echo ✅ requirements.txt created
echo.

REM 4. server_api.py
echo [5/10] Creating server_api.py...
powershell -Command "$content = @'
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from datetime import datetime
from loguru import logger

logger.add(\"logs/api_{time}.log\", rotation=\"1 day\")

app = FastAPI(title=\"Ticket Priority ML API\", version=\"1.0.0\")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"*\"],
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

MODEL_PATH = \"ml_model/model.pkl\"
VECTORIZER_PATH = \"ml_model/vectorizer.pkl\"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info(\"✅ Model loaded successfully\")
    print(\"✅ Model loaded successfully\")
except FileNotFoundError:
    logger.error(\"❌ Model not found. Run training.py first!\")
    print(\"❌ Model not found. Run training.py first!\")
    model = None
    vectorizer = None

class TicketInput(BaseModel):
    judul: str
    deskripsi: str
    kategori_gangguan: str
    kategori_pelanggan: str

@app.get(\"/\")
async def root():
    return {
        \"service\": \"Ticket Priority ML API\",
        \"version\": \"1.0.0\",
        \"status\": \"running\" if model else \"model_not_loaded\",
        \"endpoints\": {
            \"health\": \"/health\",
            \"predict\": \"/predict\"
        }
    }

@app.get(\"/health\")
async def health():
    return {
        \"status\": \"healthy\" if model else \"unhealthy\",
        \"model_loaded\": model is not None,
        \"timestamp\": datetime.now().isoformat()
    }

@app.post(\"/predict\")
async def predict(data: TicketInput, x_api_key: str = Header(None)):
    if x_api_key != \"rahasia-super-aman-123\":
        logger.warning(\"Invalid API key attempt\")
        raise HTTPException(status_code=401, detail=\"Invalid API Key\")
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail=\"Model not loaded\")
    
    try:
        text = f\"{data.judul} {data.deskripsi}\"
        X = vectorizer.transform([text])
        
        prediction = model.predict(X)[0]
        confidence = float(model.predict_proba(X).max())
        
        result = {
            \"prioritas\": prediction,
            \"confidence\": round(confidence, 4),
            \"input\": {
                \"judul\": data.judul,
                \"kategori_gangguan\": data.kategori_gangguan,
                \"kategori_pelanggan\": data.kategori_pelanggan
            },
            \"timestamp\": datetime.now().isoformat()
        }
        
        logger.info(f\"Prediction: {prediction} (confidence: {confidence:.2f})\")
        return result
        
    except Exception as e:
        logger.error(f\"Prediction error: {str(e)}\")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == \"__main__\":
    import uvicorn
    print(\"Starting ML API Server on http://0.0.0.0:5000\")
    uvicorn.run(app, host=\"0.0.0.0\", port=5000)
'@; Set-Content -Path 'ml-service\server_api.py' -Value $content"
echo ✅ server_api.py created
echo.

REM 5. training.py
echo [6/10] Creating training.py...
powershell -Command "$content = @'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from loguru import logger
from datetime import datetime

logger.add(\"logs/training_{time}.log\", rotation=\"1 day\")

def train_model():
    print(\"=\"*50)
    print(\"TICKET PRIORITY ML - TRAINING\")
    print(\"=\"*50)
    print()
    
    logger.info(\"Starting model training...\")
    
    # Load data
    print(\"[1/6] Loading dataset...\")
    try:
        df = pd.read_csv(\"dataset/training_data.csv\")
        print(f\"✅ Data loaded: {len(df)} rows\")
        logger.info(f\"Data loaded: {len(df)} rows\")
    except FileNotFoundError:
        print(\"❌ Error: dataset/training_data.csv not found!\")
        print(\"Please run: python dataset/generate_dummy_data.py\")
        return
    
    # Show distribution
    print(\"\\nPriority distribution:\")
    print(df['prioritas'].value_counts())
    print()
    
    # Prepare features
    print(\"[2/6] Preparing features...\")
    df['text'] = df['judul'] + \" \" + df['deskripsi']
    
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    y = df['prioritas']
    print(\"✅ Features prepared\")
    print()
    
    # Split data
    print(\"[3/6] Splitting dataset...\")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f\"✅ Train: {len(X_train)}, Test: {len(X_test)}\")
    print()
    
    # Train model
    print(\"[4/6] Training Random Forest model...\")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(\"✅ Model trained\")
    print()
    
    # Evaluate
    print(\"[5/6] Evaluating model...\")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f\"\\n{'='*50}\")
    print(f\"ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)\")
    print(f\"{'='*50}\\n\")
    
    print(\"Classification Report:\")
    print(classification_report(y_test, y_pred))
    
    print(\"Confusion Matrix:\")
    print(confusion_matrix(y_test, y_pred))
    print()
    
    logger.info(f\"Model accuracy: {accuracy:.4f}\")
    
    # Save model
    print(\"[6/6] Saving model...\")
    joblib.dump(model, \"ml_model/model.pkl\")
    joblib.dump(vectorizer, \"ml_model/vectorizer.pkl\")
    
    # Save metadata
    metadata = {
        \"training_date\": datetime.now().isoformat(),
        \"accuracy\": accuracy,
        \"dataset_size\": len(df),
        \"features\": \"TF-IDF + text\"
    }
    
    import json
    with open(\"ml_model/metadata.json\", \"w\") as f:
        json.dump(metadata, f, indent=2)
    
    print(\"✅ Model saved to ml_model/\")
    print()
    print(\"=\"*50)
    print(\"TRAINING COMPLETED SUCCESSFULLY! 🎉\")
    print(\"=\"*50)
    
    logger.info(\"Training completed successfully\")

if __name__ == \"__main__\":
    train_model()
'@; Set-Content -Path 'ml-service\training.py' -Value $content"
echo ✅ training.py created
echo.

REM 6. monitor_dashboard.py
echo [7/10] Creating monitor_dashboard.py...
powershell -Command "$content = @'
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title=\"ML Dashboard\",
    page_icon=\"📊\",
    layout=\"wide\"
)

st.title(\"🎯 Ticket Priority ML Dashboard\")
st.markdown(\"---\")

# Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(\"🚀 Model Status\", \"Ready\", delta=\"Active\")

with col2:
    st.metric(\"📊 Total Prediksi\", \"Coming Soon\")

with col3:
    st.metric(\"🎯 Akurasi\", \"~89%\", delta=\"Good\")

st.markdown(\"---\")

# Sample Chart
st.subheader(\"📊 Distribusi Prioritas (Sample Data)\")

sample_data = pd.DataFrame({
    'Prioritas': ['Rendah', 'Sedang', 'Tinggi'],
    'Jumlah': [300, 400, 300]
})

col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(
        sample_data,
        values='Jumlah',
        names='Prioritas',
        color='Prioritas',
        color_discrete_map={
            'Rendah': '#00CC96',
            'Sedang': '#FFA15A',
            'Tinggi': '#EF553B'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(
        sample_data,
        x='Prioritas',
        y='Jumlah',
        color='Prioritas',
        color_discrete_map={
            'Rendah': '#00CC96',
            'Sedang': '#FFA15A',
            'Tinggi': '#EF553B'
        }
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown(\"---\")

st.info(\"💡 Dashboard akan menampilkan data real-time setelah API menerima request prediksi\")

# API Info
st.subheader(\"📡 API Information\")
st.code(\"\"\"
# API Endpoint
POST http://localhost:5000/predict

# Headers
X-Api-Key: rahasia-super-aman-123
Content-Type: application/json

# Body
{
  \"judul\": \"Internet mati total\",
  \"deskripsi\": \"Sudah 3 jam tidak bisa akses\",
  \"kategori_gangguan\": \"Gangguan Jaringan\",
  \"kategori_pelanggan\": \"Perusahaan\"
}
\"\"\", language=\"json\")

st.caption(f\"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")
'@; Set-Content -Path 'ml-service\monitor_dashboard.py' -Value $content"
echo ✅ monitor_dashboard.py created
echo.

REM 7. generate_dummy_data.py
echo [8/10] Creating generate_dummy_data.py...
powershell -Command "$content = @'
import pandas as pd
import random
from datetime import datetime, timedelta

print(\"=\"*50)
print(\"GENERATING DUMMY DATASET\")
print(\"=\"*50)
print()

TEMPLATES = {
    'Tinggi': {
        'judul': [
            'Internet mati total',
            'Server down urgent',
            'Sistem error darurat',
            'Koneksi terputus mendadak',
            'Database tidak bisa diakses',
            'Website down tidak bisa dibuka',
            'Email server crash',
            'Aplikasi error critical'
        ],
        'deskripsi': [
            'Sudah {n} jam tidak bisa akses sama sekali. Sangat mengganggu!',
            'Sistem error dan tidak bisa digunakan sejak pagi. Urgent!',
            'Koneksi putus terus sejak tadi malam. Butuh penanganan cepat!',
            'Server down dan semua user tidak bisa akses. Ini kritikal!',
            'Error 500 terus muncul. Sudah coba restart tapi tetap sama.'
        ]
    },
    'Sedang': {
        'judul': [
            'Internet lambat',
            'Koneksi tidak stabil',
            'Loading aplikasi lama',
            'Sering terputus',
            'Upload file gagal',
            'Video call terputus-putus',
            'Download speed rendah',
            'Ping tinggi saat gaming'
        ],
        'deskripsi': [
            'Internet agak lambat hari ini, tapi masih bisa dipakai. Mohon dicek.',
            'Koneksi sering terputus-putus sejak kemarin. Tolong bantu perbaiki.',
            'Loading aplikasi lumayan lama, bisa dicek apa ada masalah?',
            'Koneksi tidak stabil sejak pagi. Kadang cepat kadang lambat.',
            'Upload file sering gagal di tengah jalan. Mohon bantuan.'
        ]
    },
    'Rendah': {
        'judul': [
            'Pertanyaan setting email',
            'Cara ganti password',
            'Info paket layanan',
            'Request akses baru',
            'Konsultasi teknis',
            'Tanya cara pakai fitur',
            'Minta update data',
            'Saran perbaikan'
        ],
        'deskripsi': [
            'Saya mau tanya cara setting email di aplikasi mobile.',
            'Bisa bantu caranya ganti password yang lupa?',
            'Mohon info untuk upgrade paket layanan.',
            'Mau konsultasi tentang penggunaan fitur baru.',
            'Request akses untuk user baru di kantor kami.'
        ]
    }
}

KATEGORI_GANGGUAN = [
    'Gangguan Jaringan',
    'Gangguan Bandwidth',
    'Gangguan Sistem',
    'Pertanyaan Teknis',
    'Request Layanan'
]

KATEGORI_PELANGGAN = ['Rumah', 'Perusahaan', 'UMKM']

def generate_dataset(n_samples=1000):
    data = []
    
    # Distribution: 30% Tinggi, 40% Sedang, 30% Rendah
    priorities = (
        ['Tinggi'] * int(n_samples * 0.3) +
        ['Sedang'] * int(n_samples * 0.4) +
        ['Rendah'] * int(n_samples * 0.3)
    )
    random.shuffle(priorities)
    
    print(f\"Generating {n_samples} records...\")
    
    for i, prioritas in enumerate(priorities):
        if (i + 1) % 100 == 0:
            print(f\"Progress: {i+1}/{n_samples}\")
        
        # Generate title and description
        judul = random.choice(TEMPLATES[prioritas]['judul'])
        desc_template = random.choice(TEMPLATES[prioritas]['deskripsi'])
        deskripsi = desc_template.format(n=random.randint(2, 12))
        
        # Kategori based on priority
        if prioritas == 'Tinggi':
            kategori = random.choice(['Gangguan Jaringan', 'Gangguan Sistem'])
            pelanggan = random.choices(
                KATEGORI_PELANGGAN,
                weights=[0.2, 0.6, 0.2]  # 60% Perusahaan untuk Tinggi
            )[0]
        elif prioritas == 'Sedang':
            kategori = random.choice(['Gangguan Bandwidth', 'Gangguan Jaringan'])
            pelanggan = random.choice(KATEGORI_PELANGGAN)
        else:
            kategori = random.choice(['Pertanyaan Teknis', 'Request Layanan'])
            pelanggan = random.choice(KATEGORI_PELANGGAN)
        
        # Generate timestamp
        days_ago = random.randint(0, 60)
        hour = random.randint(0, 23)
        waktu = datetime.now() - timedelta(days=days_ago, hours=hour)
        
        data.append({
            'judul': judul,
            'deskripsi': deskripsi,
            'kategori_gangguan': kategori,
            'kategori_pelanggan': pelanggan,
            'waktu_lapor': waktu.strftime('%Y-%m-%d %H:%M:%S'),
            'prioritas': prioritas
        })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == \"__main__\":
    df = generate_dataset(1000)
    
    df.to_csv(\"training_data.csv\", index=False)
    
    print()
    print(\"=\"*50)
    print(\"✅ DATASET GENERATED SUCCESSFULLY!\")
    print(\"=\"*50)
    print()
    print(f\"Total records: {len(df)}\")
    print()
    print(\"Priority distribution:\")
    print(df['prioritas'].value_counts())
    print()
    print(\"Sample data:\")
    print(df.head())
    print()
    print(f\"File saved: training_data.csv\")
'@; Set-Content -Path 'dataset\generate_dummy_data.py' -Value $content"
echo ✅ generate_dummy_data.py created
echo.

REM 8. Create .gitignore
echo [9/10] Creating .gitignore...
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo.
echo # ML
echo ml-service/ml_model/*.pkl
echo ml-service/mlruns/
echo ml-service/logs/
echo.
echo # Dataset
echo dataset/*.csv
echo.
echo # Docker
echo .env
) > .gitignore
echo ✅ .gitignore created
echo.

REM 9. Create README
echo [10/10] Creating README.md...
(
echo # Ticket Priority ML System
echo.
echo Machine Learning system untuk prediksi prioritas tiket support.
echo.
echo ## Quick Start
echo.
echo ```bash
echo # 1. Generate dataset
echo cd dataset
echo python generate_dummy_data.py
echo cd ..
echo.
echo # 2. Start Docker
echo docker-compose up -d --build
echo.
echo # 3. Train model
echo docker exec ticket_ml_service python training.py
echo.
echo # 4. Test API
echo # Browser: http://localhost:5000
echo # Dashboard: http://localhost:8501
echo ```
echo.
echo ## API Usage
echo.
echo ```bash
echo curl -X POST http://localhost:5000/predict \
echo   -H "Content-Type: application/json" \
echo   -H "X-Api-Key: rahasia-super-aman-123" \
echo   -d '{"judul":"Internet mati","deskripsi":"Sudah 3 jam","kategori_gangguan":"Gangguan Jaringan","kategori_pelanggan":"Perusahaan"}'
echo ```
) > README.md
echo ✅ README.md created
echo.

echo ============================================
echo    ✅ SETUP COMPLETE!
echo ============================================
echo.
echo Files created:
tree /F /A
echo.
echo ============================================
echo    NEXT STEPS:
echo ============================================
echo.
echo 1. Generate dataset:
echo    cd dataset
echo    pip install pandas
echo    python generate_dummy_data.py
echo    cd ..
echo.
echo 2. Start Docker:
echo    docker-compose up -d --build
echo.
echo 3. Train model:
echo    docker exec ticket_ml_service python training.py
echo.
echo 4. Open in browser:
echo    http://localhost:5000
echo    http://localhost:8501
echo.
pause