import sys
import os
from app.training_pipeline import RareClassGroupingTransformer
from app.config import TRAINING_PIPELINE_PATH, MODEL_DIR, LABEL_ENCODER_PATH, MODEL_PATH
import joblib
import pandas as pd
import numpy as np


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    #Load pipeline & label encoder
    pipeline = joblib.load(TRAINING_PIPELINE_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    #Transform features using pipeline (no fitting)
    X = pipeline.transform(df)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    #Predict
    y_pred_indices = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    #Convert predictions to labels and confidence
    y_pred_labels = le.inverse_transform(y_pred_indices)
    y_pred_conf = np.max(y_pred_proba, axis=1)

    #Return results
    result_df = df.copy()
    result_df['Predicted Category'] = y_pred_labels
    result_df['Confidence Score'] = y_pred_conf

    return result_df[['clean_description', 'Predicted Category', 'Confidence Score']]
