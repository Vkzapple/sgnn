"""Fase 3 — Menuntut future work paper (Section V):"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.core.sgnn_core import SGNNCore
from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.eval.synthetic_stream import generate_stream_v2
from experiments.run_experiment_v2 import hidden_accuracy_probe

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

MODES = ["sgnn_agb_v2", "static", "grow_only", "prune_only"]

def run_config(X, labels, mode, stream_seed):
    d = X.shape[1]
    sgnn = SGNNCore(d_in=d, n0=64, seed=0)
    agc = AdaptiveGrowthControllerV2(seed=0) if mode == "sgnn_agb_v2" else None
    novelty_points = [4800, 5600]

    log_rows = []
    H_hist = []
    for t in range(len(X) - 1):
        x_t, x_next = X[t], X[t + 1]
        pe_t = sgnn.backward_and_step(x_t, x_next, lr=1e-2)
        _, h_t = sgnn.forward(x_t)
        H_hist.append(h_t.copy())

        if mode == "sgnn_agb_v2":
            agc.update_replay(x_t, x_next)
            info = agc.step(x_t, pe_t, sgnn)
        elif mode == "grow_only":
            info = {"status_drift": "NORMAL", "action": "STOP"}
            if t in list([1500, 3500]) + novelty_points and sgnn.N < 128:
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
            info = {"status_drift": "NORMAL", "action": "STOP"}
            if t > 0 and t % 1000 == 0 and sgnn.N > 16:
                sgnn.prune(4)
                info["action"] = "PRUNE"
        else:
            info = {"status_drift": "NORMAL", "action": "STOP"}

        info["t"] = t
        info["pe_t"] = pe_t
        info["n_neurons"] = sgnn.N
        log_rows.append(info)

    df = pd.DataFrame(log_rows)

    idxs, accs = hidden_accuracy_probe(np.asarray(H_hist, dtype=object),
                                       labels[: len(H_hist)])
    pe_smooth = df["pe_t"].rolling(50, min_periods=1).mean().values
    pe_at_probe = pe_smooth[np.clip(idxs, 0, len(pe_smooth) - 1)]

    corr_raw = np.corrcoef(pe_at_probe, accs)[0, 1] if len(accs) > 2 else np.nan
    x_time = np.arange(len(accs), dtype=float)
    def _resid(v):
        b, a = np.polyfit(x_time, v, 1)
        return v - (a + b * x_time)
    corr_dt = np.corrcoef(_resid(pe_at_probe), _resid(accs))[0, 1] \
        if len(accs) > 2 and np.std(pe_at_probe) > 0 else np.nan

    tail = df[df["t"] >= df["t"].max() - 1000]
    return df, pe_smooth, idxs, accs, {
        "mode": mode, "stream_seed": stream_seed,
        "mean_pe_all": df["pe_t"].mean(),
        "mean_pe_last1000": tail["pe_t"].mean(),
        "n_neurons_final": int(df["n_neurons"].iloc[-1]),
        "max_neurons": int(df["n_neurons"].max()),
        "grow_events": int((df["action"] == "GROW").sum()),
        "prune_events": int((df["action"] == "PRUNE").sum()),
        "drift_or_warning_events": int(df["status_drift"].isin(
            ["WARNING", "DRIFT"]).sum()),
        "corr_raw": corr_raw,
        "corr_detrended": corr_dt,
        "probe_acc_final": accs[-1] if len(accs) else np.nan,
    }

def main():
    t0 = time.time()
    T, d = 6000, 16
    rows = []
    seeds = list(range(42, 52))

    for seed in seeds:
        X, labels, drift_points, novelty_points = generate_stream_v2(
            T=T, d=d, seed=seed)
        print(f"\n=== Stream v2 seed {seed} "
              f"(drift@{drift_points}, novelty@{novelty_points}) ===")
        for mode in MODES:
            df, pe_smooth, idxs, accs, row = run_config(X, labels, mode, seed)
            rows.append(row)
            print(f"  [{mode}] PE={row['mean_pe_all']:.1f} "
                  f"PElast={row['mean_pe_last1000']:.1f} "
                  f"neuron={row['n_neurons_final']} "
                  f"(max {row['max_neurons']}) "
                  f"grow={row['grow_events']} prune={row['prune_events']} "
                  f"corrDT={row['corr_detrended']:.2f}")
            if mode == "sgnn_agb_v2" and seed == 42:
                df.to_csv(os.path.join(RESULTS, "fase3_sgnn_agb_v2_seed42.csv"),
                          index=False)
                plot_run(df, pe_smooth, idxs, accs, drift_points,
                         novelty_points)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "fase3_summary.csv"), index=False)

    lines = []
    piv = summary.pivot(index="stream_seed", columns="mode",
                        values="mean_pe_all")
    for base in ["static", "grow_only", "prune_only"]:
        a = piv["sgnn_agb_v2"].values
        b = piv[base].values
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        lines.append(
            f"PE(all): sgnn_agb_v2 ({a.mean():.2f}) vs {base} ({b.mean():.2f}) | "
            f"paired t-test p={t_p:.4g}, Wilcoxon p={w_p:.4g}")

    piv_t = summary.pivot(index="stream_seed", columns="mode",
                          values="mean_pe_last1000")
    for base in ["static", "grow_only", "prune_only"]:
        a = piv_t["sgnn_agb_v2"].values
        b = piv_t[base].values
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        lines.append(
            f"PE(last1k): sgnn_agb_v2 ({a.mean():.2f}) vs {base} ({b.mean():.2f}) | "
            f"paired t-test p={t_p:.4g}, Wilcoxon p={w_p:.4g}")

    report = "\n".join(lines)
    print("\n=== UJI SIGNIFIKANSI (n=10 seed, paired) ===")
    print(report)
    with open(os.path.join(RESULTS, "fase3_stats.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n=== RINGKASAN (mean 10 seed) ===")
    agg = summary.groupby("mode")[["mean_pe_all", "mean_pe_last1000",
                                   "n_neurons_final", "max_neurons",
                                   "grow_events", "prune_events",
                                   "drift_or_warning_events",
                                   "corr_raw", "corr_detrended",
                                   "probe_acc_final"]].mean()
    print(agg.round(3).to_string())
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")

def plot_run(df, pe_smooth, idxs, accs, drift_points, novelty_points):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df["t"], df["n_neurons"])
    for dp in drift_points:
        axes[0].axvline(dp, color="red", linestyle="--", alpha=0.6)
    for npnt in novelty_points:
        axes[0].axvline(npnt, color="green", linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Jumlah neuron")
    axes[0].set_title("AGC v2 pada stream kompleksitas meningkat (seed 42)")
    axes[1].plot(df["t"], pe_smooth)
    axes[1].set_ylabel("PE(t) smoothed")
    axes[1].set_xlabel("timestep t")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fase3_sgnn_agb_v2_seed42.png"), dpi=130)
    plt.close(fig)

if __name__ == "__main__":
    main()
