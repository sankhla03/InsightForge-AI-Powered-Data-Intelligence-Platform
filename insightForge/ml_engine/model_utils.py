import joblib
import os
from django.conf import settings
import pandas as pd

MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_engine", "trained_model.pkl")

def load_model():
    return joblib.load(MODEL_PATH)

def make_prediction(input_data: dict, feature_columns: list):
    model = load_model()
    df = pd.DataFrame([input_data], columns=feature_columns)
    return model.predict(df)[0]