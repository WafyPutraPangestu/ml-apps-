import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib, json, os
from datetime import datetime

MODEL_PATH = "ml_model/model.pkl"
VECTORIZER_PATH = "ml_model/vectorizer.pkl"
METADATA_PATH = "ml_model/metadata.json"
DATASET_PATH = "dataset/training_data.csv"

def train_model():
    print("=" * 50)
    print("TRAINING MODEL PROCESS STARTED")
    print("=" * 50)

    # 1. Load Data
    print("\n[1/5] Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] File tidak ditemukan: {DATASET_PATH}")
        print("        Jalankan dulu: python dataset/generate_dummy_data.py")
        exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Loaded {len(df)} rows")
    print(f"     Distribusi kelas:\n{df['prioritas'].value_counts().to_string()}")

    if len(df) < 50:
        print("[WARNING] Dataset terlalu kecil, hasil tidak akan akurat.")

    # 2. Prepare Features
    print("\n[2/5] Preparing features...")
    df['text'] = (
        df['judul'] + " " +
        df['deskripsi'] + " " +
        df['kategori_gangguan'] + " " +
        df['kategori_pelanggan']
    )

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2          # Abaikan kata yang muncul < 2x (kurangi noise)
    )
    X = vectorizer.fit_transform(df['text'])
    y = df['prioritas']
    print(f"[OK] Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # 3. Split Data
    print("\n[3/5] Splitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y        # Pastikan proporsi kelas sama di train & test
    )
    print(f"[OK] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 4. Train Model
    print("\n[4/5] Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,         # Lebih kecil dari sebelumnya (20) untuk cegah overfit
        min_samples_leaf=4,   # Minimal 4 sampel per daun
        max_features='sqrt',  # Hanya pakai subset fitur tiap split
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[OK] Model trained")

    # 5. Evaluate
    print("\n[5/5] Evaluating & saving...")

    y_pred       = model.predict(X_test)
    train_acc    = accuracy_score(y_train, model.predict(X_train))
    test_acc     = accuracy_score(y_test, y_pred)
    gap          = train_acc - test_acc

    # Cross-validation (5-fold) untuk ukuran generalisasi yang jujur
    cv_scores    = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    report_dict  = classification_report(y_test, y_pred, output_dict=True)
    cm           = confusion_matrix(y_test, y_pred, labels=['Tinggi', 'Sedang', 'Rendah'])

    # Deteksi overfit otomatis
    if gap < 0.05:
        overfit_status = "GOOD — tidak overfit"
    elif gap < 0.15:
        overfit_status = "WARNING — sedikit overfit"
    else:
        overfit_status = "OVERFIT — perlu tuning"

    # Simpan model & vectorizer
    os.makedirs("ml_model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("[OK] Model & vectorizer saved")

    # Simpan metadata detail
    metadata = {
        "training_date": datetime.now().isoformat(),
        "model_version": "3.0.0",
        "algorithm": "RandomForestClassifier",
        "dataset": {
            "path": DATASET_PATH,
            "total_rows": len(df),
            "class_distribution": df['prioritas'].value_counts().to_dict()
        },
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "tfidf_max_features": 1000,
            "tfidf_ngram_range": "1,2",
            "tfidf_min_df": 2
        },
        "performance": {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "overfit_gap": round(gap, 4),
            "overfit_status": overfit_status,
            "cross_validation": {
                "folds": 5,
                "mean": round(cv_scores.mean(), 4),
                "std": round(cv_scores.std(), 4),
                "scores": [round(s, 4) for s in cv_scores.tolist()]
            },
            "per_class_metrics": {
                cls: {
                    "precision": round(report_dict[cls]["precision"], 4),
                    "recall":    round(report_dict[cls]["recall"], 4),
                    "f1_score":  round(report_dict[cls]["f1-score"], 4),
                    "support":   int(report_dict[cls]["support"])
                }
                for cls in ['Tinggi', 'Sedang', 'Rendah']
            },
            "confusion_matrix": {
                "labels": ["Tinggi", "Sedang", "Rendah"],
                "matrix": cm.tolist()
            }
        },
        "features_used": ["judul", "deskripsi", "kategori_gangguan", "kategori_pelanggan"],
        "trained_by": "Tim IT - Helpdesk"
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("[OK] Metadata saved")

    # Ringkasan hasil
    print("\n" + "=" * 50)
    print("HASIL TRAINING")
    print("=" * 50)
    print(f"  Train Accuracy  : {train_acc:.4f}")
    print(f"  Test Accuracy   : {test_acc:.4f}")
    print(f"  Overfit Gap     : {gap:.4f}  →  {overfit_status}")
    print(f"  CV Mean ± Std   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix (Tinggi / Sedang / Rendah):")
    print(cm)
    print("=" * 50)

if __name__ == "__main__":
    train_model()