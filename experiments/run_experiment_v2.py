"""Eksperimen Fase 2 — SGNN-AGB dengan AGC v2 (redesign sesuai paper §IV.F)"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.sgnn_core import SGNNCore
from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.eval.synthetic_stream import generate_stream


RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def hidden_accuracy_probe(H_hist, labels_hist, window=200, k=3):
    """Probe pada representasi hidden h_t. Window yang melintasi aksi
    struktural (dimensi berubah) dilewati."""
    H_hist = np.asarray(H_hist, dtype=object)
    accs, idxs = [], []
    for start in range(0, len(H_hist) - window, window):
        train_end, test_end = start + window // 2, start + window
        seg = H_hist[start:test_end]
        dims = {np.asarray(h).shape[0] for h in seg}
        if len(dims) != 1:
            continue
        H = np.stack([np.asarray(h, dtype=float) for h in seg])
        y = labels_hist[start:test_end]
        H_train, y_train = H[:train_end - start], y[:train_end - start]
        H_test, y_test = H[train_end - start:], y[train_end - start:]
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


def run_config(X, labels, drift_points, novelty_points, mode, stream_seed, verbose=False):
    """mode: 'sgnn_agb_v2' | 'static' | 'grow_only' | 'prune_only'"""
    T, d = len(X) - 1, X.shape[1]
    sgnn = SGNNCore(d_in=d, n0=64, seed=0)
    agc = AdaptiveGrowthControllerV2(seed=0) if mode == "sgnn_agb_v2" else None

    log_rows, H_hist = [], []
    for t in range(T):
        x_t, x_next = X[t], X[t + 1]
        pe_t = sgnn.backward_and_step(x_t, x_next, lr=1e-2)
        h_t, _ = sgnn.forward(x_t)
        H_hist.append(h_t.copy())

        if mode == "sgnn_agb_v2":
            agc.update_replay(x_t, x_next)
            info = agc.step(x_t, pe_t, sgnn)
        elif mode == "grow_only":
            info = {"S_t": np.nan, "B_t": np.nan, "P_t": np.nan,
                    "status_drift": "NORMAL", "action": "MAINTAIN", "g_m": None}
            if t in list(drift_points) + list(novelty_points) and sgnn.N < 128:
                trial = sgnn.clone()
                trial.grow(8, seed=stream_seed + t)
                for _ in range(50):
                    j = np.random.default_rng(t).integers(0, max(t, 1))
                    trial.backward_and_step(X[j], X[j + 1], lr=1e-2)
                if trial.eval_pe_batch([(X[t], X[t + 1])]) < \
                        sgnn.eval_pe_batch([(X[t], X[t + 1])]):
                    sgnn.grow(8, seed=stream_seed + t)
                    info["action"] = "GROW"
        elif mode == "prune_only":
            info = {"S_t": np.nan, "B_t": np.nan, "P_t": np.nan,
                    "status_drift": "NORMAL", "action": "MAINTAIN", "g_m": None}
            if t > 0 and t % 1000 == 0 and sgnn.N > 16:
                sgnn.prune(4)
                info["action"] = "PRUNE"
        else:
            info = {"S_t": np.nan, "B_t": np.nan, "P_t": np.nan,
                    "status_drift": "NORMAL", "action": "MAINTAIN", "g_m": None}

        info["t"] = t
        info["pe_t"] = pe_t
        info["n_neurons"] = sgnn.N
        log_rows.append(info)

    df = pd.DataFrame(log_rows)

    idxs, accs = hidden_accuracy_probe(H_hist, labels[: len(H_hist)])
    pe_smooth = df["pe_t"].rolling(50, min_periods=1).mean().values
    pe_at_probe = pe_smooth[np.clip(idxs, 0, len(pe_smooth) - 1)]
    corr = np.corrcoef(pe_at_probe, accs)[0, 1] if len(accs) > 2 else np.nan

    tail = df[df["t"] >= df["t"].max() - 1000]
    row = {
        "mode": mode, "stream_seed": stream_seed,
        "mean_pe_all": df["pe_t"].mean(),
        "mean_pe_last1000": tail["pe_t"].mean(),
        "n_neurons_final": int(df["n_neurons"].iloc[-1]),
        "grow_events": int((df["action"] == "GROW").sum()),
        "prune_events": int((df["action"] == "PRUNE").sum()),
        "corr_pe_acc": corr,
        "probe_acc_final": accs[-1] if len(accs) else np.nan,
    }
    delays = []
    for dp in drift_points + novelty_points:
        w = df[(df["t"] >= dp) & (df["t"] < dp + 500)]
        fd = w[w["status_drift"].isin(["WARNING", "DRIFT"])]["t"]
        delays.append((fd.iloc[0] - dp) if len(fd) else np.nan)
    row["drift_delays"] = str(delays)

    if verbose:
        print(f"  [{mode} seed={stream_seed}] "
              f"PE(all)={row['mean_pe_all']:.2f} PE(last1k)={row['mean_pe_last1000']:.2f} "
              f"neuron={row['n_neurons_final']} grow={row['grow_events']} "
              f"prune={row['prune_events']} corr={row['corr_pe_acc']:.3f}")
    return df, idxs, accs, pe_smooth, pe_at_probe, row


def main():
    t0 = time.time()
    seeds = [42, 43, 44]
    T, d = 6000, 16
    all_rows = []

    for seed in seeds:
        X, labels, drift_points, novelty_points = generate_stream(T=T, d=d, seed=seed)
        print(f"\n=== Stream seed {seed} (drift@{drift_points}, novelty@{novelty_points}) ===")
        for mode in ["sgnn_agb_v2", "static", "grow_only", "prune_only"]:
            df, idxs, accs, pe_smooth, pe_at_probe, row = run_config(
                X, labels, drift_points, novelty_points, mode, seed, verbose=True)
            df.to_csv(os.path.join(RESULTS, f"fase2_{mode}_seed{seed}.csv"), index=False)
            if mode == "sgnn_agb_v2":
                plot_run(df, idxs, accs, pe_smooth, pe_at_probe,
                         drift_points, novelty_points, seed)
            row["drift_delays"] = row["drift_delays"]
            all_rows.append(row)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(RESULTS, "fase2_summary.csv"), index=False)

    print("\n=== RINGKASAN (mean over 3 seeds) ===")
    agg = summary.groupby("mode")[["mean_pe_all", "mean_pe_last1000",
                                   "n_neurons_final", "grow_events", "prune_events",
                                   "corr_pe_acc", "probe_acc_final"]].mean()
    print(agg.round(3).to_string())
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik. "
          f"Output di folder results/.")


def plot_run(df, idxs, accs, pe_smooth, pe_at_probe, drift_points, novelty_points, seed):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(df["t"], df["n_neurons"])
    for dp in drift_points:
        axes[0].axvline(dp, color="red", linestyle="--", alpha=0.6)
    for npnt in novelty_points:
        axes[0].axvline(npnt, color="green", linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Jumlah neuron")
    axes[0].set_title(f"AGC v2 — neuron vs waktu (stream seed {seed})")
    axes[1].plot(df["t"], pe_smooth)
    axes[1].set_ylabel("PE(t) smoothed")
    axes[2].plot(idxs, accs, marker="o", color="purple")
    axes[2].set_ylabel("Hidden accuracy (probe)")
    axes[2].set_xlabel("timestep t")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"fase2_sgnn_agb_v2_seed{seed}.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
