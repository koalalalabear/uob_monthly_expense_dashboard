# test_inference.py

import pandas as pd
from app.inference_pipeline import run_inference

# Load the real transactions CSV
csv_path = "app/pages/data/transactions.csv"
df = pd.read_csv(csv_path)

# Run inference
result_df = run_inference(df)

# Display results
print(result_df.head())
