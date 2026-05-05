# ============================================
# UPDATED: preprocess_behavioral.py
# NON-IID + SCALABLE CLIENT GENERATION
# ============================================

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = "Dataset/household_power_consumption.txt"
OUTPUT_DIR = "processed_data"
MINUTES_PER_DAY = 1440


# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--clients", type=int, default=4)
args = parser.parse_args()

NUM_CLIENTS = args.clients


def load_and_clean_data():
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep=';',
        na_values='?',
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True
    )
    df.set_index("datetime", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)
    df = df.ffill()

    return df


def create_daily_profiles(df):
    daily_groups = []

    for date, day_df in df.groupby(df.index.date):
        if len(day_df) == MINUTES_PER_DAY:
            daily_groups.append({
                "date": date,
                "mean_load": day_df["Global_active_power"].mean(),
                "max_load": day_df["Global_active_power"].max(),
                "std_load": day_df["Global_active_power"].std(),
                "evening_ratio": (
                    day_df.between_time("18:00", "23:59")["Global_active_power"].mean()
                    / day_df["Global_active_power"].mean()
                )
            })

    return pd.DataFrame(daily_groups)


def cluster_days(daily_features):
    feature_cols = ["mean_load", "max_load", "std_load", "evening_ratio"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(daily_features[feature_cols])

    kmeans = KMeans(
        n_clusters=min(NUM_CLIENTS, len(daily_features)),
        random_state=42
    )

    daily_features["client_id"] = kmeans.fit_predict(X_scaled) + 1

    return daily_features


def assign_days_to_clients(df, daily_features):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # DELETE old files
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("Client_"):
            os.remove(os.path.join(OUTPUT_DIR, f))

    df_dates = pd.Series(df.index.date, index=df.index)

    for client_id in range(1, NUM_CLIENTS + 1):

        client_days = daily_features[daily_features["client_id"] == client_id]

        # 🚨 Ensure no empty clients
        if len(client_days) == 0:
            continue

        client_dates = set(client_days["date"])
        client_df = df[df_dates.isin(client_dates)].copy()

        # -------- NON-IID ENHANCEMENT --------
        # Add slight bias per client
        noise_scale = 0.01 * client_id
        client_df.iloc[:, :7] += np.random.normal(
            0, noise_scale, client_df.iloc[:, :7].shape
        )

        # -------- NORMALIZATION --------
        scaler = StandardScaler()
        client_df.iloc[:, :7] = scaler.fit_transform(client_df.iloc[:, :7])

        # -------- STATIC METADATA --------
        meta = client_days[
            ["mean_load", "max_load", "std_load", "evening_ratio"]
        ].mean()

        for col in meta.index:
            client_df[col] = meta[col]

        save_path = f"{OUTPUT_DIR}/Client_{client_id}.csv"
        client_df.to_csv(save_path)

        print(f"[OK] Client {client_id} → {len(client_df)} samples")


def main():
    print(f"\nGenerating NON-IID data for {NUM_CLIENTS} clients...\n")

    df = load_and_clean_data()
    daily_features = create_daily_profiles(df)
    daily_features = cluster_days(daily_features)
    assign_days_to_clients(df, daily_features)

    print("\n✅ Done. Non-IID clients created.")


if __name__ == "__main__":
    main()