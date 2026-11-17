"""Data Taobao UserBehavior: mirror publik + featurisasi kausal."""

import os

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction import FeatureHasher

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PATH = os.path.join(DATA, "taobao_userbehavior_sample.csv")
URL = ("https://raw.githubusercontent.com/OrzPiggy/UserBehavior/"
       "master/UserBehavior.csv")
COLUMNS = ["user", "item", "cat", "behavior", "ts"]
T = 6000


def ensure_data():
    if os.path.exists(PATH) and os.path.getsize(PATH) > 1e6:
        return
    os.makedirs(DATA, exist_ok=True)
    r = requests.get(URL, timeout=180)
    r.raise_for_status()
    with open(PATH, "wb") as f:
        f.write(r.content)


def load_stream():
    ensure_data()
    df = pd.read_csv(PATH, header=None, names=COLUMNS)
    df = df.sort_values("ts").reset_index(drop=True)
    days = pd.to_datetime(df.ts, unit="s").dt.normalize()
    counts = days.value_counts()
    full_days = counts[counts >= 3000].index
    df = df[days.isin(full_days)].reset_index(drop=True)
    df = df.iloc[np.linspace(0, len(df) - 1, T).astype(int)].reset_index(drop=True)
    days = pd.to_datetime(df.ts, unit="s").dt.normalize()
    buckets = (days - days.min()).dt.days.values.astype(int)
    return df, buckets


def build_features(df):
    beh = df.behavior.map({"pv": 0, "cart": 1, "fav": 2, "buy": 3})
    beh = beh.fillna(0).astype(int).values
    ts = pd.to_datetime(df.ts, unit="s")
    hour = ts.dt.hour.values + ts.dt.minute.values / 60.0

    records = [
        {"u": f"u{u}", "i": f"i{i}", "b": f"b{b}"}
        for u, i, b in zip(df.user.values, df.item.values, beh)
    ]
    hasher = FeatureHasher(n_features=48, input_type="dict", alternate_sign=True)
    Z = hasher.transform(records).toarray()

    item_seen = df.groupby("item").cumcount().values
    user_seen = df.groupby("user").cumcount().values
    time_feats = np.column_stack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        beh,
        np.log1p(item_seen),
        np.log1p(user_seen),
    ])
    return np.hstack([Z, time_feats]).astype(np.float64)


def build_labels(df, top_k=20):
    top = df.cat.value_counts().index[:top_k]
    mapping = {c: i for i, c in enumerate(top)}
    return df.cat.map(mapping).fillna(top_k).astype(int).values


if __name__ == "__main__":
    df, buckets = load_stream()
    X = build_features(df)
    y = build_labels(df)
    print(f"{len(df)} events, {len(np.unique(buckets))} hari, X={X.shape}, "
          f"kelas={len(np.unique(y))}")
    print(pd.Series(buckets).value_counts().sort_index().to_string())
