"""Fase 7 — Amazon Product Reviews (SNAP reviews_Software_5)."""

import sys
import os
import gzip
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from experiments.run_fase5_clear10 import run_config


RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(RESULTS, exist_ok=True)
URL = ("https://snap.stanford.edu/data/amazon/productGraph/"
       "categoryFiles/reviews_Software_5.json.gz")
PATH = os.path.join(DATA, "reviews_Software_5.json.gz")
T = 6000
N_SEG = 10


def ensure_data():
    if not os.path.exists(PATH) or os.path.getsize(PATH) < 1e6:
        r = requests.get(URL, timeout=300)
        r.raise_for_status()
        with open(PATH, "wb") as f:
            f.write(r.content)
        print("Downloaded", os.path.getsize(PATH) / 1e6, "MB")


def load_stream():
    ensure_data()
    rows = []
    with gzip.open(PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                js = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = js.get("reviewText") or js.get("summary") or ""
            ts = js.get("unixReviewTime")
            rating = js.get("overall")
            if txt and ts and rating:
                rows.append((int(ts), txt, int(round(rating))))
    df = pd.DataFrame(rows, columns=["ts", "text", "rating"]).sort_values("ts")
    df = df.reset_index(drop=True)
    if len(df) > T:
        idx = np.linspace(0, len(df) - 1, T).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    seg = np.minimum((np.arange(len(df)) * N_SEG) // len(df), N_SEG - 1)
    return df, seg


def build_features(df, seg, d=64):
    first = seg == 0
    vec = TfidfVectorizer(max_features=3000, stop_words="english",
                          min_df=2).fit(df.text[first])
    Z = vec.transform(df.text)
    svd = TruncatedSVD(n_components=d, random_state=0).fit(Z[first])
    return svd.transform(Z).astype(np.float64)


def main():
    t0 = time.time()
    reviews, seg = load_stream()
    print(f"Amazon Software: {len(reviews)} ulasan, {reviews.ts.nunique()} timestamp unik")
    X = build_features(reviews, seg)
    y = reviews.rating.values
    print(f"Fitur: {X.shape}, kelas rating: {sorted(set(y))}")

    rows = []
    for seed in [0, 1, 2]:
        for mode in ["sgnn_agb_v2", "static"]:
            df, acc, bounds, delays, row = run_config(X, y, seg, mode, seed)
            rows.append(row)
            print(f"[{mode:12s} seed{seed}] PE={row['mean_pe']:.2f} "
                  f"PEakhir={row['mean_pe_last2buckets']:.2f} "
                  f"n={row['neuron_final']} grow={row['grow']} "
                  f"prune={row['prune']} det={row['delay_frac_detected']:.2f} "
                  f"acc={row['probe_acc_mean']:.3f}")
            if seed == 0 and mode == "sgnn_agb_v2":
                df.to_csv(os.path.join(RESULTS, "fase7_amazon_v2_seed0.csv"),
                          index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "fase7_amazon_summary.csv"), index=False)
    print("\n=== RINGKASAN Amazon ===")
    print(summary.groupby("mode")[
        ["mean_pe", "mean_pe_last2buckets", "neuron_final", "n_max",
         "grow", "prune", "delay_frac_detected", "delay_mean",
         "probe_acc_mean", "probe_acc_last"]].mean().round(3).to_string())
    print(f"Selesai {time.time() - t0:.0f} detik")


if __name__ == "__main__":
    main()
