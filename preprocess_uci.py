import pandas as pd
import os
import numpy as np

def clean_and_split():
    # 1. Load the raw data
    raw_path = "Dataset/household_power_consumption.txt"
    print(f"--- Loading data from {raw_path} ---")
    
    # Reading with semi-colon separator, treating '?' as NaN
    df = pd.read_csv(raw_path, sep=';', low_memory=False, na_values='?', 
                     parse_dates={'datetime': ['Date', 'Time']}, index_col='datetime')

    # 2. Basic Cleaning
    print("Cleaning missing values...")
    df = df.ffill() # Forward fill missing energy values (Standard for time-series)

    # 3. Create 4 "Virtual Households" (Clients)
    # We will split the dataset into 4 non-overlapping chunks
    os.makedirs("processed_data", exist_ok=True)
    n_samples = len(df)
    chunk_size = n_samples // 4

    for i in range(4):
        client_id = i + 1
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != 3 else n_samples
        
        client_df = df.iloc[start:end]
        save_path = f"processed_data/Client_{client_id}.csv"
        client_df.to_csv(save_path)
        print(f"[OK] Saved Client {client_id} data to {save_path} ({len(client_df)} rows)")

if __name__ == "__main__":
    clean_and_split()