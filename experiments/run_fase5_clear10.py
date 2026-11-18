"""Fase 5 — Clear-10 (real-world image stream, 2004-2014, 10 bucket temporal)."""

import sys
import os
import zipfile
import glob
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

from src.core.sgnn_core import SGNNCore
from src.controller.agc_v2 import AdaptiveGrowthControllerV2


RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(RESULTS, exist_ok=True)
ZIP = os.path.join(DATA, "clear10-test.zip")
EXTRACT = os.path.join(DATA, "clear10_test")


def ensure_extracted():
    if os.path.exists(EXTRACT) and glob.glob(os.path.join(EXTRACT, "**", "*.jpg"),
                                             recursive=True):
        return
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(EXTRACT)


def build_index(limit_per_bucket=600):
    """Kumpulkan (path, bucket, class) dari struktur folder Clear-10."""
    files = glob.glob(os.path.join(EXTRACT, "**", "*.jpg"), recursive=True)
    items = []
    for p in files:
        parts = os.path.basename(os.path.dirname(p)), \
            os.path.basename(os.path.dirname(os.path.dirname(p)))
        cls, bucket = parts
        if not bucket.isdigit() or int(bucket) == 0:
            continue
        items.append((p, int(bucket), cls))
    by_bucket = {}
    for p, b, c in sorted(items, key=lambda x: (x[1], x[0])):
        by_bucket.setdefault(b, []).append((p, c))
    order = sorted(by_bucket)
    rng = np.random.default_rng(0)
    X_raw, y, buckets = [], [], []
    for b in order:
        lst = by_bucket[b][:limit_per_bucket]
        perm = rng.permutation(len(lst))
        for j in perm:
            p, c = lst[j]
            X_raw.append(p)
            y.append(c)
            buckets.append(b)
    return X_raw, np.array(y), np.array(buckets), order


def img_features(path):
    im = Image.open(path).convert("L").resize((16, 16))
    return np.asarray(im, dtype=np.float64).ravel() / 255.0


def build_features(paths, buckets, d_pca=64, seed=0):
    """Fitur dari backbone beku ResNet18 (sesuai pseudocode proposal:
    x_t <- backbone_beku(frame)). Di-cache, tetapi cache hanya dipakai bila
    urutan path identik (mencegah fitur salah-sejajar dengan label)."""
    cache = os.path.join(DATA, "clear10_features_resnet18.npy")
    order_file = os.path.join(DATA, "clear10_features_order.json")
    Z = None
    if os.path.exists(cache) and os.path.exists(order_file):
        with open(order_file, encoding="utf-8") as f:
            cached_paths = json.load(f)
        if cached_paths == paths:
            Z = np.load(cache)
    if Z is None:
        Z = _extract_backbone(paths)
        np.save(cache, Z)
        with open(order_file, "w", encoding="utf-8") as f:
            json.dump(paths, f)

    first = buckets == buckets[0]
    mu, sd = Z[first].mean(0), Z[first].std(0) + 1e-6
    Zs = (Z - mu) / sd
    pca = PCA(n_components=d_pca, random_state=seed).fit(Zs[first])
    return pca.transform(Zs).astype(np.float64)


def _extract_backbone(paths, batch=32):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision import transforms

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk])
            out.append(model(xb).numpy())
            if (i // batch) % 20 == 0:
                print(f"  fitur {i}/{len(paths)}", flush=True)
    return np.vstack(out).astype(np.float64)


def probe_acc_per_bucket(H, y, buckets):
    """kNN probe per bucket (label hanya untuk evaluasi); dimensi hidden
    yang berubah di tengah bucket ditangani dengan memilih dimensi mayoritas."""
    out = {}
    for b in np.unique(buckets):
        idx = np.where(buckets == b)[0]
        dims = np.array([np.asarray(H[i]).shape[0] for i in idx])
        if len(dims) == 0:
            continue
        dim = int(np.bincount(dims).argmax())
        idx = idx[dims == dim]
        h = np.stack([np.asarray(H[i]) for i in idx])
        lab = y[idx]
        half = len(idx) // 2
        if half < 5:
            continue
        correct = 0
        for i in range(half, len(idx)):
            dists = np.linalg.norm(h[:half] - h[i], axis=1)
            nn = np.argsort(dists)[:3]
            pred = pd.Series(lab[:half][nn]).mode().iloc[0]
            correct += int(pred == lab[i])
        out[b] = correct / (len(idx) - half)
    return out


def run_config(X, y, buckets, mode, model_seed):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=model_seed)
    agc = AdaptiveGrowthControllerV2(seed=model_seed) \
        if mode == "sgnn_agb_v2" else None
    log, H = [], []
    for t in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[t], X[t + 1], lr=1e-2)
        h, _ = sgnn.forward(X[t])
        H.append(h.copy())
        if mode == "sgnn_agb_v2":
            agc.update_replay(X[t], X[t + 1])
            info = agc.step(X[t], pe, sgnn)
        else:
            info = {"status_drift": "NORMAL", "action": "STOP"}
        info.update(t=t, pe_t=pe, n_neurons=sgnn.N, bucket=buckets[t])
        log.append(info)
    df = pd.DataFrame(log)
    acc = probe_acc_per_bucket(H, y[: len(H)], buckets[: len(H)])

    boundaries = [int(np.where(buckets == b)[0][0]) for b in np.unique(buckets)[1:]]
    delays = []
    for bp in boundaries:
        w = df[(df.t >= bp) & (df.t < bp + 500)]
        fd = w[w.status_drift.isin(["WARNING", "DRIFT"])]["t"]
        delays.append(int(fd.iloc[0] - bp) if len(fd) else -1)

    per_bucket_pe = df.groupby("bucket")["pe_t"].mean().to_dict()
    return df, acc, boundaries, delays, {
        "mode": mode, "model_seed": model_seed,
        "mean_pe": df["pe_t"].mean(),
        "mean_pe_last2buckets": df[df.bucket >= df.bucket.max() - 1].pe_t.mean(),
        "neuron_final": int(df.n_neurons.iloc[-1]),
        "n_max": int(df.n_neurons.max()),
        "grow": int((df.action == "GROW").sum()),
        "prune": int((df.action == "PRUNE").sum()),
        "delay_mean": np.mean([d for d in delays if d >= 0]) if any(
            d >= 0 for d in delays) else -1,
        "delay_frac_detected": np.mean([d >= 0 for d in delays]),
        "probe_acc_mean": np.mean(list(acc.values())) if acc else np.nan,
        "probe_acc_first": acc.get(int(np.unique(buckets)[0]), np.nan),
        "probe_acc_last": acc.get(int(np.unique(buckets)[-1]), np.nan),
        "per_bucket_pe": str({int(k): round(v, 2) for k, v in per_bucket_pe.items()}),
    }


def main():
    t0 = time.time()
    ensure_extracted()
    paths, y, buckets, order = build_index()
    print(f"Stream Clear-10: {len(paths)} gambar, bucket {order}, "
          f"{len(np.unique(y))} kelas")
    X = build_features(paths, buckets, seed=0)
    print(f"Fitur: {X.shape} (PCA dari fitur backbone beku)")
    base_acc = probe_acc_per_bucket(X, y, buckets)
    print(f"Baseline kNN pada fitur (tanpa model): "
          f"acc={np.mean(list(base_acc.values())):.3f}")

    rows = []
    for seed in [0, 1, 2]:
        for mode in ["sgnn_agb_v2", "static"]:
            df, acc, bounds, delays, row = run_config(X, y, buckets, mode, seed)
            rows.append(row)
            print(f"[{mode:12s} seed{seed}] PE={row['mean_pe']:.2f} "
                  f"PEakhir={row['mean_pe_last2buckets']:.2f} "
                  f"n={row['neuron_final']} grow={row['grow']} "
                  f"prune={row['prune']} det={row['delay_frac_detected']:.2f} "
                  f"delay~{row['delay_mean']:.0f} "
                  f"acc={row['probe_acc_mean']:.3f}")
            if seed == 0 and mode == "sgnn_agb_v2":
                df.to_csv(os.path.join(RESULTS, "fase5_clear10_v2_seed0.csv"),
                          index=False)
                plot(df, bounds)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "fase5_clear10_summary.csv"),
                   index=False)
    print("\n=== RINGKASAN Clear-10 ===")
    print(summary.groupby("mode")[
        ["mean_pe", "mean_pe_last2buckets", "neuron_final", "n_max",
         "grow", "prune", "delay_frac_detected", "delay_mean",
         "probe_acc_mean", "probe_acc_last"]].mean().round(3).to_string())
    print(f"Selesai {time.time() - t0:.0f} detik")


def plot(df, bounds):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(df.t, df.n_neurons)
    ax[1].plot(df.t, df.pe_t.rolling(50, min_periods=1).mean())
    for b in bounds:
        ax[0].axvline(b, color="red", ls="--", alpha=0.4)
        ax[1].axvline(b, color="red", ls="--", alpha=0.4)
    ax[0].set_ylabel("Jumlah neuron")
    ax[0].set_title("AGC v2 pada Clear-10 (batas bucket = drift)")
    ax[1].set_ylabel("PE(t) smoothed")
    ax[1].set_xlabel("timestep t")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fase5_clear10.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
