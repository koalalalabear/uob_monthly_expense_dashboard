import os

# BASE_DIR = os.path.abspath(os.pa÷th.join(os.path.dirname(__file__), '..', '..'))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# Directories
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Training file
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'transactions.csv')

# Model artifact paths
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'onehot.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')

TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
SVD_PATH = os.path.join(MODEL_DIR, 'svd_transformer.joblib')
MERCHANT_VOCAB_PATH = os.path.join(MODEL_DIR, 'vocab_dict.joblib')
MERCHANT_EMBED_PATH = os.path.join(MODEL_DIR, 'merchant_embedding.joblib')

TRAINING_PIPELINE_PATH = os.path.join(MODEL_DIR, 'training_pipeline.joblib')
