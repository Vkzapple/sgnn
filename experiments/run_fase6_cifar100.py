"""Fase 6 — Split-CIFAR100 (class-incremental, benchmark CL standar)."""

import sys, os, tarfile, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from sklearn.decomposition import PCA

from src.core.sgnn_core import SGNNCore
from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from experiments.run_fase5_clear10 import run_config, plot

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(RESULTS, exist_ok=True)
TAR = os.path.join(DATA, "cifar-100-python.tar.gz")
SRC = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

def ensure_data():
    if not os.path.exists(os.path.join(DATA, "cifar-100-python")):
        if not os.path.exists(TAR):
            print("Download CIFAR-100 (~160MB)...")
            with requests.get(SRC, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(TAR, "wb") as f:
                    for ch in r.iter_content(1 << 20):
                        f.write(ch)
        with tarfile.open(TAR) as t:
            t.extractall(DATA)

def load_split(per_task=600, n_tasks=10, classes_per_task=10):
    ensure_data()
    with open(os.path.join(DATA, "cifar-100-python", "test"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    y = np.asarray(d[b"fine_labels"])
    rng = np.random.default_rng(0)

    paths_placeholder = []
    imgs, labels, tasks = [], [], []
    for task in range(n_tasks):
        cls_set = set(range(task * classes_per_task,
                            (task + 1) * classes_per_task))
        idx = np.where(np.isin(y, list(cls_set)))[0]
        idx = rng.permutation(idx)[:per_task]
        imgs.append(X[idx])
        tasks.append(np.full(len(idx), task))
    return (np.vstack(imgs).astype(np.float64), np.concatenate(labels),
            np.concatenate(tasks))

def features_from_rgb(Xrgb, tasks, d_pca=64):
    """Fitur dari backbone beku ResNet18 (sesuai pseudocode proposal),
    lalu PCA per-sumbu dari statistik task pertama. Di-cache agar rerun cepat."""
    cache = os.path.join(DATA, "cifar100_features_resnet18.npy")
    if os.path.exists(cache):
        Z = np.load(cache)
        if len(Z) != len(Xrgb):
            Z = None
    else:
        Z = None
    if Z is None:
        Z = _extract_backbone(Xrgb)
        np.save(cache, Z)
    first = tasks == tasks[0]
    mu, sd = Z[first].mean(0), Z[first].std(0) + 1e-6
    Z = (Z - mu) / sd
    pca = PCA(n_components=d_pca, random_state=0).fit(Z[first])
    return pca.transform(Z)

def _extract_backbone(Xrgb):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    feats = []
    with torch.no_grad():
        for i in range(0, len(Xrgb), 64):
            chunk = Xrgb[i:i + 64].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            chunk = (chunk / 255.0 - mean) / std
            xb = torch.from_numpy(chunk.transpose(0, 3, 1, 2)).float()
            xb = torch.nn.functional.interpolate(
                xb, size=(224, 224), mode="bilinear", align_corners=False)
            feats.append(model(xb).numpy())
            if (i // 64) % 20 == 0:
                print(f"  fitur {i}/{len(Xrgb)}", flush=True)
    return np.vstack(feats).astype(np.float64)

def main():
    t0 = time.time()
    Xrgb, y, tasks = load_split(per_task=600)
    print(f"Split-CIFAR100: {len(Xrgb)} sampel, {len(np.unique(tasks))} task")
    X = features_from_rgb(Xrgb, tasks)

    rows = []
    for seed in [0, 1, 2]:
        for mode in ["sgnn_agb_v2", "static"]:
            df, acc, bounds, delays, row = run_config(X, y, tasks, mode, seed)
            rows.append(row)
            print(f"[{mode:12s} seed{seed}] PE={row['mean_pe']:.2f} "
                  f"PEakhir={row['mean_pe_last2buckets']:.2f} "
                  f"n={row['neuron_final']} grow={row['grow']} "
                  f"prune={row['prune']} det={row['delay_frac_detected']:.2f} "
                  f"delay~{row['delay_mean']:.0f} acc={row['probe_acc_mean']:.3f}")
            if seed == 0 and mode == "sgnn_agb_v2":
                df.to_csv(os.path.join(RESULTS, "fase6_cifar100_v2_seed0.csv"),
                          index=False)
                plot(df, bounds)
                os.replace(os.path.join(RESULTS, "fase5_clear10.png"),
                           os.path.join(RESULTS, "fase6_cifar100.png"))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "fase6_cifar100_summary.csv"),
                   index=False)
    print("\n=== RINGKASAN Split-CIFAR100 ===")
    print(summary.groupby("mode")[
        ["mean_pe", "mean_pe_last2buckets", "neuron_final", "n_max",
         "grow", "prune", "delay_frac_detected", "delay_mean",
         "probe_acc_mean", "probe_acc_last"]].mean().round(3).to_string())
    print(f"Selesai {time.time() - t0:.0f} detik")

if __name__ == "__main__":
    main()
