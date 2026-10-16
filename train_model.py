"""
Offline Training Pipeline for Earthquake Risk Prediction Model.
Generates earthquake_model_v1.pth and scaler.pkl.
"""
import os
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

from app import EarthquakeNN, EarthquakeModel, FEATURE_NAMES, MODEL_FILE, SCALER_FILE, DEVICE

MIN_TRAINING_QUAKES = 50
TRAINING_MONTHS = 6
REQUEST_TIMEOUT = 45
CHUNK_SIZE_DAYS = 60


def fetch_training_data():
    """Fetches historical USGS earthquake data in chunks."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30 * TRAINING_MONTHS)

    all_features = []
    current_start = start_date

    print(f"📡 Fetching USGS data from {start_date.date()} to {end_date.date()}...")

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=CHUNK_SIZE_DAYS), end_date)

        try:
            url = (
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                f"format=geojson&starttime={current_start.strftime('%Y-%m-%d')}"
                f"&endtime={current_end.strftime('%Y-%m-%d')}"
                f"&minmagnitude=2.5&orderby=time"
            )

            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            features = response.json().get('features', [])
            all_features.extend(features)
            print(f"   Fetched {len(features)} records up to {current_end.date()}")

        except Exception as e:
            print(f"❌ Fetch error for chunk {current_start.date()} to {current_end.date()}: {e}")

        current_start = current_end + timedelta(days=1)
        time.sleep(1)  # Respect USGS rate limits

    return all_features


def process_training_data(model_mgr, quakes):
    """Processes USGS GeoJSON features into training DataFrame."""
    data = []
    for q in quakes:
        try:
            props = q.get('properties') or {}
            coords = q.get('geometry', {}).get('coordinates') or []
            if len(coords) < 3:
                continue

            mag = props.get('mag')
            if mag is None:
                continue

            data.append({
                'latitude': float(coords[1]),
                'longitude': float(coords[0]),
                'depth': float(coords[2]),
                'mag': float(mag),
                'significant_quake': int(float(mag) >= 5.0)
            })
        except Exception:
            continue

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).dropna()
    df = model_mgr.compute_density(df, radius=1.0)
    df = model_mgr.compute_plate_distance(df)
    return df


def train():
    """Executes the full offline training workflow."""
    print("🚀 Initializing training pipeline...")
    model_mgr = EarthquakeModel()

    quakes = fetch_training_data()
    if not quakes or len(quakes) < MIN_TRAINING_QUAKES:
        print(f"❌ Insufficient data ({len(quakes)} records fetched). Minimum required: {MIN_TRAINING_QUAKES}")
        return False

    print("📊 Preprocessing and calculating spatial features...")
    df = process_training_data(model_mgr, quakes)
    if df.empty:
        print("❌ Dataset preprocessing yielded empty DataFrame.")
        return False

    sig_count = len(df[df['significant_quake'] == 1])
    print(f"   Processed {len(df)} samples with {sig_count} significant earthquakes (M >= 5.0).")

    if sig_count < 5:
        print("❌ Not enough positive samples for training.")
        return False

    df = df.sample(frac=1, random_state=42).fillna(0)
    X = df[FEATURE_NAMES].values
    y = df['significant_quake'].values.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    model = EarthquakeNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    best_val_loss = float('inf')
    patience = 5
    counter = 0
    epochs = 60
    start_time = time.time()

    print(f"🧠 Training neural network for up to {epochs} epochs on {DEVICE}...")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = loss_fn(out, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = loss_fn(val_out, y_val_t).item()
            train_eval = model(X_train_t)
            train_acc = ((train_eval > 0.5) == y_train_t).float().mean().item()
            val_acc = ((val_out > 0.5) == y_val_t).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            status_msg = "💾 Saved best"
        else:
            counter += 1
            status_msg = ""

        if (epoch + 1) % 5 == 0 or counter >= patience:
            print(f"[Epoch {epoch+1:02d}/{epochs}] Loss: {loss.item():.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | Val Loss: {val_loss:.4f} {status_msg}")

        if counter >= patience:
            print("🛑 Early stopping triggered.")
            break

    print(f"✅ Training completed in {time.time() - start_time:.1f}s.")
    print(f"   Model saved: {MODEL_FILE}")
    print(f"   Scaler saved: {SCALER_FILE}")
    return True


if __name__ == '__main__':
    train()
