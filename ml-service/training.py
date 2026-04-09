import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime
import json
import os

# Definisikan path
MODEL_PATH = "ml_model/model.pkl"
VECTORIZER_PATH = "ml_model/vectorizer.pkl"
METADATA_PATH = "ml_model/metadata.json"
DATASET_PATH = "dataset/training_data.csv"

def train_model():
    print("="*50)
    print("TRAINING MODEL PROCESS STARTED")
    print("="*50)
    
    # 1. Load Data
    print("[1/5] Loading dataset...")
    try:
        if not os.path.exists(DATASET_PATH):
            print(f"[ERROR] {DATASET_PATH} not found!")
            exit(1)
            
        df = pd.read_csv(DATASET_PATH)
        print(f"[OK] Data loaded: {len(df)} rows")
        
        # GUARD CLAUSE: Cek jika data terlalu sedikit
        if len(df) < 5:
            print("[WARNING] Data sangat sedikit (< 5). Training mungkin tidak akurat.")
    except Exception as e:
        print(f"[ERROR] Error loading CSV: {e}")
        exit(1)
    
    # 2. Prepare Features
    print("[2/5] Preparing features...")
    try:
        df['text'] = (df['judul'] + " " + 
                      df['deskripsi'] + " " + 
                      df['kategori_gangguan'] + " " + 
                      df['kategori_pelanggan'])
        
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['text'])
        y = df['prioritas']
        print("[OK] Features prepared")
    except Exception as e:
        print(f"[ERROR] Error preparing features: {e}")
        exit(1)
    
    # 3. Split Data (DENGAN LOGIKA PINTAR)
    print("[3/5] Splitting dataset...")
    
    # Jika data kurang dari 10, matikan fitur 'stratify' agar tidak error
    if len(df) < 10:
        print("[INFO] Small dataset detected. Disabling stratification.")
        stratify_mode = None
    else:
        stratify_mode = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_mode
        )
    except ValueError:
        # Fallback jika masih error (misal data cuma 1 atau 2)
        print("[WARN] Split failed. Training on full dataset instead.")
        X_train, X_test, y_train, y_test = X, X, y, y

    # 4. Train Model
    print("[4/5] Training model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("[OK] Model trained")
    
    # 5. Evaluate & Save
    print("[5/5] Saving model...")
    
    os.makedirs("ml_model", exist_ok=True)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    metadata = {
        "training_date": datetime.now().isoformat(),
        "accuracy": accuracy,
        "dataset_size": len(df)
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] SUCCESS! Accuracy: {accuracy:.4f}")
    print("="*50)

if __name__ == "__main__":
    train_model()