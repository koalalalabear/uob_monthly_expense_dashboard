# utils.py
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import joblib

from app.config import TFIDF_PATH, SVD_PATH,MERCHANT_VOCAB_PATH,MERCHANT_EMBED_PATH,ENCODER_PATH

# utils.py

# --- TEXT CLEANING ---
def clean_text(text):

    if not isinstance(text, str):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove punctuation & special chars
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    text = text.strip()
    return text

def show_merchant(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # remove punctuation, special chars, and numbers
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def should_clean(text):
    if not isinstance(text, str):
        return False
    tokens = text.split()
    if all(token.isdigit() for token in tokens):  # if all tokens are numbers
        return False
    return True

def clean_text_conditionally(text):
    return show_merchant(text) if should_clean(text) else text

def merchant_handler():
    df = df.copy()  # optional: avoids modifying the original DataFrame
    df['clean_merchant'] = df['clean_description'].apply(show_merchant)
    df.drop(columns=["clean_description"], inplace=True)
    return df


# --- DATE FEATURES ---
def create_cyclical_features(df, col_name, max_val):
    """
    inner function within date_handler to create cyclical features for a given column.
    Adds two columns to df: col_name_sin and col_name_cos,
    representing the cyclical encoding of col_name.
    """
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_val)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_val)


def date_handler(df):
    df["datetime"] = pd.to_datetime(df["date"])
    df["day_of_month"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["week_of_year"] = df["datetime"].dt.isocalendar().week
    df.drop(columns=["date"], inplace=True)
    
    # Create cyclical features
    create_cyclical_features(df, 'day_of_month', 31)
    create_cyclical_features(df, 'week_of_year', 52)
    create_cyclical_features(df, 'month', 12)

    return df


# --- TRANSACTION / MERCHANT COUNT FEATURES ---
def get_transaction_type_counts(df, window_months):
    """
    Returns a dictionary of transaction_type counts in the past `window_months`.
    """
    # Make sure 'date' is datetime
    df = df.copy()

    end_date = df['datetime'].max()
    start_date = end_date - pd.DateOffset(months=window_months)
    
    df_window = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    type_counts = df_window['transaction_type'].value_counts().to_dict()
    
    return type_counts


def add_transaction_type_counts(df, windows=[3, 6]):
    """
    Adds transaction_type count features to the dataframe for each time window in `windows`.
    """
    df = df.copy()
    
    for window in windows:
        counts = get_transaction_type_counts(df, window)
        col_name = f'transaction_type_count_{window}m'
        df[col_name] = df['transaction_type'].map(counts).fillna(0).astype(int)
    
    return df

def get_merchant_type_counts(df, window_months):
    """
    Returns a dictionary of transaction_type counts in the past `window_months`.
    """
    # Make sure 'date' is datetime
    df = df.copy()

    end_date = df['datetime'].max()
    start_date = end_date - pd.DateOffset(months=window_months)
    
    df_window = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    type_counts = df_window['clean_merchant'].value_counts().to_dict()
    
    return type_counts


def add_merchant_type_counts(df, windows=[1,3,6]):
    """
    Adds transaction_type count features to the dataframe for each time window in `windows`.
    """
    df = df.copy()
    
    for window in windows:
        counts = get_merchant_type_counts(df, window)
        col_name = f'merchant_type_count_{window}m'
        df[col_name] = df['clean_merchant'].map(counts).fillna(0).astype(int)
    
    return df

# # --- TEXT FEATURES (TF-IDF + SVD) ---

def add_merchant_tfidf_features(
    df, 
    tfidf_path=TFIDF_PATH, 
    svd_path=SVD_PATH,
    n_components=5
):
    """
    Loads TF-IDF and SVD models, transforms 'clean_merchant' column,
    and adds reduced tfidf features to the DataFrame.

    Ensures consistent columns even if input is empty or malformed.

    Returns:
        df (pd.DataFrame): Modified DataFrame with merchant_tfidf_ columns added
    """
    tfidf_vectorizer = joblib.load(tfidf_path)
    svd_transformer = joblib.load(svd_path)

    # Handle missing or non-string clean_merchant
    if 'clean_merchant' not in df.columns:
        df['clean_merchant'] = ""

    df['clean_merchant'] = df['clean_merchant'].fillna("").astype(str)

    try:
        merchant_tfidf = tfidf_vectorizer.transform(df["clean_merchant"])
        merchant_svd = svd_transformer.transform(merchant_tfidf)
    except Exception as e:
        print(f"⚠️ Warning: TF-IDF transformation failed: {e}")
        merchant_svd = np.zeros((df.shape[0], n_components))

    # Add the tfidf_svd components as columns
    for i in range(n_components):
        df[f"merchant_tfidf_{i}"] = merchant_svd[:, i] if merchant_svd.shape[1] > i else 0

    return df

# # --- EMBEDDING / VOCAB ---
# def build_vocab(tokens_list, min_freq): ...
# def tokens_to_seq(tokens, vocab, unk_token=0): ...
# def embed_text(seq, embedding_matrix): ...

def tokens_to_seq(tokens, vocab, unk_token=0):
    """
    Converts a list of tokens to a list of indices using a vocab dictionary.
    Unknown tokens are assigned the `unk_token` index.
    """
    return [vocab.get(token, unk_token) for token in tokens]

def embed_text(seq, embedding_matrix):
    """
    Converts a sequence of token indices into a single embedding vector
    by averaging their corresponding vectors from the embedding matrix.
    """
    if not seq:
        return np.zeros(embedding_matrix.shape[1])
    vectors = [embedding_matrix[idx] for idx in seq if idx < embedding_matrix.shape[0]]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_matrix.shape[1])

def add_merchant_embeddings(
    df, 
    vocab_path=MERCHANT_VOCAB_PATH, 
    embed_path=MERCHANT_EMBED_PATH):

    vocab = joblib.load(vocab_path)
    embedding_matrix = np.load(embed_path, allow_pickle=True)

    def get_embedding_vector(text):
        tokens = text.split() if isinstance(text, str) else []
        seq = tokens_to_seq(tokens, vocab)
        return embed_text(seq, embedding_matrix)

    # Apply embedding function
    embeddings = df["clean_merchant"].fillna("").apply(get_embedding_vector)
    embedding_dim = embedding_matrix.shape[1]

    # Add embedding dimensions as new columns
    for i in range(embedding_dim):
        df[f"merchant_tfidf_{i}"] = embeddings.apply(lambda x: x[i])

    return df

# # ---  NUMERICAL PROCESSING ---

def prepare_numeric_features(df, numeric_feature_list):
    """
    Select numeric columns from df based on numeric_feature_list.
    Fills NaNs with 0 (or any other strategy you prefer).
    
    Args:
        df (pd.DataFrame): Input dataframe.
        numeric_feature_list (list[str]): List of numeric feature column names expected.
        
    Returns:
        np.ndarray: Numeric feature matrix ready for model input.
    """
    # Check all required features exist in df
    missing_cols = [col for col in numeric_feature_list if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing numeric features in dataframe: {missing_cols}")
    
    # Select and fill NaNs if any
    X_numeric = df[numeric_feature_list].fillna(0).values
    
    return X_numeric


# # --- CATEGORICAL PROCESSING ---
def prepare_categorical_features(df, categorical_feature_list, encoder):
    """
    Encodes categorical features using a pre-trained encoder.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        categorical_feature_list (list[str]): List of categorical feature column names expected.
        encoder_path (str): Path to the pre-trained encoder model.
        
    Returns:
        np.ndarray: Encoded categorical feature matrix ready for model input.
    """
    # Check all required features exist in df
    missing_cols = [col for col in categorical_feature_list if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing categorical features in dataframe: {missing_cols}")
    
    # Load the pre-trained encoder
    encoder = ENCODER_PATH
    
    # Encode the categorical features
    X_cat_encoded = encoder.transform(df[categorical_feature_list])
    
    return X_cat_encoded

# # --- COMBINE ---
def prepare_features(df):
    """
    Prepare numeric and categorical features and combine them.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        np.ndarray: Combined feature matrix ready for model input
    """
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
    encoder_path = ENCODER_PATH
    
    X_numeric = prepare_numeric_features(df, numeric_cols)
    X_cat_encoded = prepare_categorical_features(df, categorical_cols, encoder_path)
    
    # Combine numeric and categorical features horizontally
    X_combined = np.hstack([X_numeric, X_cat_encoded])
    
    return X_combined

