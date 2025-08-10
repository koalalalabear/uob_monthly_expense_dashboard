import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from app.config import MODEL_PATH, LABEL_ENCODER_PATH, ENCODER_PATH, MODEL_DIR
from app.utils import *
from app.utils import prepare_features, prepare_numeric_features, prepare_categorical_features
from app.transformers import RareClassGroupingTransformer

# Define column lists
numeric_cols = [
    'withdrawal', 'deposit', 'day_of_month', 'month', 'week_of_year',
    'day_of_month_sin', 'day_of_month_cos', 'week_of_year_sin', 'week_of_year_cos',
    'month_sin', 'month_cos',
    'transaction_type_count_3m', 'transaction_type_count_6m',
    'merchant_type_count_1m', 'merchant_type_count_3m', 'merchant_type_count_6m',
    'first_time_merchant',
    'merchant_tfidf_0', 'merchant_tfidf_1', 'merchant_tfidf_2',
    'merchant_tfidf_3', 'merchant_tfidf_4'
]
categorical_cols = ['transaction_type']

# === Define Transformers ===
from sklearn.base import BaseEstimator, TransformerMixin

# class RareClassGroupingTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold=0.01, target_col='category', new_col='category_grouped'):
#         self.threshold = threshold
#         self.target_col = target_col
#         self.new_col = new_col
#         self.rare_classes_ = None

#     def fit(self, X, y=None):
#         freq = X[self.target_col].value_counts(normalize=True)
#         self.rare_classes_ = freq[freq < self.threshold].index.tolist()
#         return self

#     def transform(self, X):
#         X = X.copy()
#         X[self.new_col] = X[self.target_col].apply(lambda x: x if x not in self.rare_classes_ else 'unknown')
#         return X



class CleanDescriptionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['clean_description'] = X['clean_description'].apply(clean_text)
        return X

class ExtractMerchantTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['clean_merchant'] = X['clean_description'].apply(show_merchant)
        return X

class CleanTransactionTypeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['transaction_type'] = X['transaction_type'].apply(clean_text).str.replace(" ", "")
        return X

class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return date_handler(X.copy())

class TransactionTypeCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, windows=[3,6]): self.windows = windows
    def fit(self, X, y=None): return self
    def transform(self, X): return add_transaction_type_counts(X.copy(), self.windows)

class MerchantTypeCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, windows=[1,3,6]): self.windows = windows
    def fit(self, X, y=None): return self
    def transform(self, X): return add_merchant_type_counts(X.copy(), self.windows)

class MerchantTFIDFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf_path=TFIDF_PATH, svd_path=SVD_PATH, n_components=5):
        self.tfidf_path = tfidf_path
        self.svd_path = svd_path
        self.n_components = n_components
    def fit(self, X, y=None):
        self.tfidf_vectorizer = joblib.load(self.tfidf_path)
        self.svd_transformer = joblib.load(self.svd_path)
        return self
    def transform(self, X):
        df = X.copy()
        df['clean_merchant'] = df['clean_merchant'].fillna("").astype(str)
        tfidf = self.tfidf_vectorizer.transform(df['clean_merchant'])
        svd = self.svd_transformer.transform(tfidf)
        for i in range(self.n_components):
            df[f'merchant_tfidf_{i}'] = svd[:, i]
        return df

class MerchantEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_path=MERCHANT_VOCAB_PATH, embed_path=MERCHANT_EMBED_PATH):
        self.vocab_path = vocab_path
        self.embed_path = embed_path
    def fit(self, X, y=None):
        self.vocab = joblib.load(self.vocab_path)
        self.embedding_matrix = joblib.load(self.embed_path)
        return self
    def transform(self, X):
        df = X.copy()
        def get_vec(text):
            tokens = text.split() if isinstance(text, str) else []
            seq = tokens_to_seq(tokens, self.vocab)
            return embed_text(seq, self.embedding_matrix)
        embeddings = df['clean_merchant'].fillna("").apply(get_vec)
        for i in range(self.embedding_matrix.shape[1]):
            df[f'merchant_emb_{i}'] = embeddings.apply(lambda x: x[i])
        return df

class FeaturePreparationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, categorical_cols, encoder_path=ENCODER_PATH):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.encoder_path = encoder_path
    def fit(self, X, y=None):
        self.encoder = joblib.load(self.encoder_path)
        return self
    def transform(self, X):
        df = X.copy()
        # Safe handling of first_time_merchant
        if 'first_time_merchant' not in df.columns:
            df['first_time_merchant'] = 0
        else:
            df['first_time_merchant'] = df['first_time_merchant'].fillna(0)

        X_numeric = prepare_numeric_features(df, self.numeric_cols)
        X_cat = self.encoder.transform(df[self.categorical_cols])
        emb_cols = [c for c in df.columns if c.startswith('merchant_emb_')]
        X_emb = df[emb_cols].values if emb_cols else np.empty((df.shape[0], 0))
        return np.hstack([X_numeric, X_cat, X_emb])

# === Load data ===
# df = pd.read_csv('/Users/yvellenah/citicredit-dashboard/notebooks/.ipynb_checkpoints/transactions.csv')

# === Create pipeline ===
pipeline = Pipeline([
    ('rare_class_grouping', RareClassGroupingTransformer()),
    ('clean_desc', CleanDescriptionTransformer()),
    ('extract_merchant', ExtractMerchantTransformer()),
    ('clean_txn_type', CleanTransactionTypeTransformer()),
    ('date_features', DateFeatureTransformer()),
    ('txn_type_counts', TransactionTypeCountTransformer()),
    ('merchant_type_counts', MerchantTypeCountTransformer()),
    ('merchant_tfidf', MerchantTFIDFTransformer()),
    ('merchant_embedding', MerchantEmbeddingTransformer()),
    ('feature_prep', FeaturePreparationTransformer(numeric_cols, categorical_cols)),
])

# === Preprocess features ===
X = pipeline.fit_transform(df)

# === Encode target ===
le = LabelEncoder()
df = pipeline.named_steps['rare_class_grouping'].transform(df)
y = le.fit_transform(df['category_grouped'])

# === Train model ===
model = XGBClassifier(n_jobs=-1, eval_metric='mlogloss')
model.fit(X, y)

# === Save artifacts ===
joblib.dump(model, MODEL_PATH)
joblib.dump(le, LABEL_ENCODER_PATH)
joblib.dump(pipeline.named_steps['feature_prep'].encoder, ENCODER_PATH)
joblib.dump(pipeline, os.path.join(MODEL_DIR, 'training_pipeline.joblib'))
