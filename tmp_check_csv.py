import pandas as pd
import os

if os.path.exists('extracted_labels.csv'):
    df = pd.read_csv('extracted_labels.csv')
    print("Found in root. Total rows:", len(df))
elif os.path.exists('price_estimator/data/extracted_labels.csv'):
    df = pd.read_csv('price_estimator/data/extracted_labels.csv')
    print("Found in price_estimator/data. Total rows:", len(df))
else:
    print("File not found.")
