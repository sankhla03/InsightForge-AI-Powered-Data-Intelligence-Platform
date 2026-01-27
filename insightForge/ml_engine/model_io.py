import joblib
from pathlib import Path

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

def save_model(model, name="model.pkl"):
    path = MODEL_DIR / name
    joblib.dump(model, path)
    return path

def load_model(name="model.pkl"):
    return joblib.load(MODEL_DIR / name)