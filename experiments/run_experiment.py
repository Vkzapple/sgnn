

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.sgnn_core import SGNNCore
from src.controller.agc import AdaptiveGrowthController
from src.eval.synthetic_stream import generate_stream


def hidden_accuracy_probe(H_hist, labels_hist, window=200, k=3):
    """
    Probe OFFLINE saja (bukan bagian AGC): kNN classifier sederhana pada
    representasi hidden h_t vs label tersembunyi, dievaluasi per window.
    Ini satu-satunya tempat label dipakai, murni untuk mengukur apakah
    PE(t) adalah proxy yang valid untuk kualitas representasi.
    """
    accs = []
    idxs = []
    for start in range(0, len(H_hist) - window, window):
        train_end = start + window // 2
        test_end = start + window
        H_train, y_train = H_hist[start:train_end], labels_hist[start:train_end]
        H_test, y_test = H_hist[train_end:test_end], labels_hist[train_end:test_end]
        if len(H_train) < k or len(H_test) == 0:
            continue
        correct = 0
        for h, y in zip(H_test, y_test):
            dists = np.linalg.norm(H_train - h, axis=1)
            nn_idx = np.argsort(dists)[:k]
            pred = np.bincount(y_train[nn_idx]).argmax()
            correct += int(pred == y)
        accs.append(correct / len(H_test))
        idxs.append(test_end)
    return np.array(idxs), np.array(accs)


def main():
    T, d = 6000, 16
    X, labels, drift_points, novelty_points = generate_stream(T=T, d=d, seed=42)

    sgnn = SGNNCore(d_in=d, n0=64, seed=0)
    agc = AdaptiveGrowthController(seed=0)

    log_rows = []
    H_hist = []

    for t in range(T - 1):
        x_t, x_next = X[t], X[t + 1]

        pe_t = sgnn.backward_and_step(x_t, x_next, lr=1e-2)
        H_hist.append(x_t.copy())  # dim input tetap (h_t berubah dimensi tiap grow/prune)

        agc.update_replay(x_t, x_next)
        info = agc.step(x_t, pe_t, sgnn)
        info["t"] = t
        info["pe_t"] = pe_t
        log_rows.append(info)

    df = pd.DataFrame(log_rows)
    df.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "fase1_log.csv"), index=False)

    # ---- Probe offline hidden accuracy ----
    H_hist = np.array(H_hist)
    idxs, accs = hidden_accuracy_probe(H_hist, labels[: len(H_hist)])

    pe_smooth = df["pe_t"].rolling(50, min_periods=1).mean().values
    pe_at_probe = pe_smooth[np.clip(idxs, 0, len(pe_smooth) - 1)]
    if len(accs) > 2:
        corr = np.corrcoef(pe_at_probe, accs)[0, 1]
    else:
        corr = float("nan")

    print(f"Korelasi Pearson PE(t) (smoothed) vs hidden accuracy: {corr:.3f}")
    print(f"Total aksi GROW : {(df['action'] == 'GROW').sum()}")
    print(f"Total aksi PRUNE: {(df['action'] == 'PRUNE').sum()}")
    print(f"Neuron akhir    : {df['n_neurons'].iloc[-1]} (mulai dari 64)")

    for dp in drift_points:
        window_after = df[(df["t"] >= dp) & (df["t"] < dp + 300)]
        first_drift_idx = window_after[window_after["status_drift"] == "DRIFT"]["t"]
        delay = (first_drift_idx.iloc[0] - dp) if len(first_drift_idx) > 0 else None
        print(f"Titik drift t={dp}: delay deteksi = {delay}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(df["t"], df["n_neurons"])
    for dp in drift_points:
        axes[0].axvline(dp, color="red", linestyle="--", alpha=0.6, label="drift" if dp == drift_points[0] else None)
    for npnt in novelty_points:
        axes[0].axvline(npnt, color="green", linestyle=":", alpha=0.6, label="novelty" if npnt == novelty_points[0] else None)
    axes[0].set_ylabel("Jumlah neuron")
    axes[0].legend()
    axes[0].set_title("Bounded growth: jumlah neuron seiring waktu")

    axes[1].plot(df["t"], pe_smooth, label="PE(t) smoothed")
    axes[1].plot(idxs, pe_at_probe, "o", color="orange", label="PE saat probe")
    axes[1].set_ylabel("Predictive Error")
    axes[1].legend()

    axes[2].plot(idxs, accs, marker="o", color="purple")
    axes[2].set_ylabel("Hidden accuracy (offline probe)")
    axes[2].set_xlabel("timestep t")
    axes[2].set_title(f"Korelasi PE vs hidden accuracy: r = {corr:.3f}")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "fase1_summary.png")
    plt.savefig(out_path, dpi=130)
    print(f"Plot disimpan di: {out_path}")


if __name__ == "__main__":
    main()
