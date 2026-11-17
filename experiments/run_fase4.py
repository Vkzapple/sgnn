"""Fase 4 — AGC v2 pada dataset sintetis standar MOA (SEA, RandomRBF)."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.core.sgnn_core import SGNNCore
from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.eval.stream_generators import generate_sea, generate_random_rbf
from experiments.run_experiment_v2 import hidden_accuracy_probe


RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
T = 6000
DRIFT = (1500, 3500)


def run_config(X, labels, mode, seed):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=0)
    agc = AdaptiveGrowthControllerV2(seed=0) if mode == "sgnn_agb_v2" else None
    log, H = [], []
    for t in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[t], X[t + 1], lr=1e-2)
        _, h = sgnn.forward(X[t])
        H.append(h.copy())
        if mode == "sgnn_agb_v2":
            agc.update_replay(X[t], X[t + 1])
            info = agc.step(X[t], pe, sgnn)
        else:
            info = {"status_drift": "NORMAL", "action": "STOP"}
        info.update(t=t, pe_t=pe, n_neurons=sgnn.N)
        log.append(info)
    df = pd.DataFrame(log)

    idxs, accs = hidden_accuracy_probe(np.asarray(H, dtype=object),
                                       labels[: len(H)])
    pe_smooth = df["pe_t"].rolling(50, min_periods=1).mean().values
    pe_at_probe = pe_smooth[np.clip(idxs, 0, len(pe_smooth) - 1)]

    x_t = np.arange(len(accs), dtype=float)

    def resid(v):
        b, a = np.polyfit(x_t, v, 1)
        return v - (a + b * x_t)
    corr_dt = np.corrcoef(resid(pe_at_probe), resid(accs))[0, 1] \
        if len(accs) > 2 and np.std(pe_at_probe) > 0 else np.nan

    rec = {}
    for k, dp in enumerate(DRIFT):
        pre = df.loc[(df.t >= dp - 300) & (df.t < dp), "pe_t"].mean()
        post = df.loc[(df.t >= dp) & (df.t < dp + 300), "pe_t"].mean()
        rec[f"recovery_ratio_d{k}"] = post / max(pre, 1e-9)

    delays = []
    for dp in DRIFT:
        w = df[(df.t >= dp) & (df.t < dp + 500)]
        fd = w[w.status_drift.isin(["WARNING", "DRIFT"])]["t"]
        delays.append(int(fd.iloc[0] - dp) if len(fd) else -1)

    return df, pe_smooth, idxs, accs, {
        "mode": mode,
        "mean_pe_all": df["pe_t"].mean(),
        "mean_pe_last1000": df["pe_t"].iloc[-1000:].mean(),
        "neuron_final": int(df["n_neurons"].iloc[-1]),
        "n_max": int(df["n_neurons"].max()),
        "grow": int((df.action == "GROW").sum()),
        "prune": int((df.action == "PRUNE").sum()),
        "drift_events": int(df.status_drift.isin(["WARNING", "DRIFT"]).sum()),
        "delay_d0": delays[0], "delay_d1": delays[1],
        "corr_detrended": corr_dt,
        "probe_acc_final": accs[-1] if len(accs) else np.nan,
        **rec,
    }


def main():
    t0 = time.time()
    streams = {
        "SEA-abrupt": lambda s: generate_sea(T, seed=s, drift_points=DRIFT),
        "SEA-gradual": lambda s: generate_sea(T, seed=s, drift_points=DRIFT,
                                              gradual_steps=500),
        "RandomRBF": lambda s: generate_random_rbf(T, seed=s,
                                                   drift_points=DRIFT),
    }
    rows = []
    for sname, gen in streams.items():
        for seed in range(42, 47):
            X, labels, _ = gen(seed)
            for mode in ["sgnn_agb_v2", "static"]:
                df, pe_s, idxs, accs, row = run_config(X, labels, mode, seed)
                row.update(stream=sname, seed=seed)
                rows.append(row)
                print(f"[{sname} seed{seed} {mode:12s}] "
                      f"PE={row['mean_pe_all']:7.2f} "
                      f"PElast={row['mean_pe_last1000']:7.2f} "
                      f"n={row['neuron_final']:3d} grow={row['grow']:2d} "
                      f"prune={row['prune']:2d} delay={row['delay_d0']}/"
                      f"{row['delay_d1']} corrDT={row['corr_detrended']:.2f}")
                if seed == 42 and mode == "sgnn_agb_v2":
                    df.to_csv(os.path.join(
                        RESULTS, f"fase4_{sname}_sgnn_agb_v2_seed42.csv"),
                        index=False)
                    plot_run(df, pe_s, sname)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "fase4_summary.csv"), index=False)

    print("\n=== RINGKASAN per stream (mean 5 seed) ===")
    agg = summary.groupby(["stream", "mode"])[
        ["mean_pe_all", "mean_pe_last1000", "neuron_final", "n_max",
         "grow", "prune", "drift_events",
         "recovery_ratio_d0", "recovery_ratio_d1",
         "corr_detrended", "probe_acc_final"]].mean()
    print(agg.round(3).to_string())

    lines = []
    for sname in streams:
        sub = summary[summary.stream == sname]
        piv = sub.pivot(index="seed", columns="mode", values="mean_pe_all")
        a, b = piv["sgnn_agb_v2"].values, piv["static"].values
        t_p = stats.ttest_rel(a, b).pvalue
        try:
            w_p = stats.wilcoxon(a, b).pvalue
        except ValueError:
            w_p = np.nan
        lines.append(f"{sname}: v2 {a.mean():.2f} vs static {b.mean():.2f} | "
                     f"t-test p={t_p:.4g}, Wilcoxon p={w_p:.4g}")
    rep = "\n".join(lines)
    print("\n=== SIGNIFIKANSI v2 vs static (n=5 seed) ===")
    print(rep)
    with open(os.path.join(RESULTS, "fase4_stats.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


def plot_run(df, pe_s, sname):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(df.t, df.n_neurons)
    for dp in DRIFT:
        ax[0].axvline(dp, color="red", ls="--", alpha=0.6)
    ax[0].set_ylabel("Jumlah neuron")
    ax[0].set_title(f"AGC v2 — {sname} (seed 42)")
    ax[1].plot(df.t, pe_s)
    for dp in DRIFT:
        ax[1].axvline(dp, color="red", ls="--", alpha=0.6)
    ax[1].set_ylabel("PE(t) smoothed")
    ax[1].set_xlabel("timestep t")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"fase4_{sname}.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
