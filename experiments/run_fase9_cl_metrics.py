"""Fase 9 — forgetting/BWT pada Split-CIFAR100 + NCDR pada stream novelty."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.core.sgnn_core import SGNNCore
from src.eval.synthetic_stream import generate_stream_v2
from experiments.cl_metrics import evaluate_seen_tasks, ncdr_from_log, summarize
from experiments.run_fase6_cifar100 import features_from_rgb, load_split

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
N_TASKS = 10


def run_cifar(X, y, tasks, seed, mode):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=seed)
    agc = AdaptiveGrowthControllerV2(seed=seed) if mode == "sgnn_agb_v2" else None
    bounds = [int(np.where(tasks == t)[0][0]) for t in range(1, N_TASKS)]
    z = np.full(len(X) - 1, np.nan)
    R = {t: {} for t in range(N_TASKS)}
    done = 0

    for i in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[i], X[i + 1], lr=1e-2)
        if agc is not None:
            agc.update_replay(X[i], X[i + 1])
            z[i] = agc.step(X[i], pe, sgnn)["z_novelty"]
        if done < len(bounds) and i == bounds[done] - 1:
            scores = evaluate_seen_tasks(sgnn, X, y, tasks, range(done + 1))
            for t, v in scores.items():
                R[t][done] = v
            done += 1

    scores = evaluate_seen_tasks(sgnn, X, y, tasks, range(N_TASKS))
    for t, v in scores.items():
        R[t][N_TASKS - 1] = v

    metrics = summarize(R, N_TASKS)
    det = ncdr_from_log(z, bounds) if agc is not None else (np.nan, np.nan, np.nan)
    return R, metrics, det


def run_novelty(seed):
    X, _, drift, novelty = generate_stream_v2(
        T=6000, d=16, seed=seed, drift_points=(1500, 3500),
        novelty_points=(4800, 5600))
    sgnn = SGNNCore(d_in=16, n0=64, seed=seed)
    agc = AdaptiveGrowthControllerV2(seed=seed)
    z = np.empty(len(X) - 1)
    for i in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[i], X[i + 1], lr=1e-2)
        agc.update_replay(X[i], X[i + 1])
        z[i] = agc.step(X[i], pe, sgnn)["z_novelty"]
    return ncdr_from_log(z, novelty, exclude=drift)


def main():
    t0 = time.time()
    Xrgb, y, tasks = load_split(per_task=600)
    X = features_from_rgb(Xrgb, tasks)
    print(f"Split-CIFAR100: {X.shape}, {N_TASKS} task")

    rows = []
    for seed in [0, 1, 2]:
        for mode in ["sgnn_agb_v2", "static"]:
            R, metrics, det = run_cifar(X, y, tasks, seed, mode)
            pd.DataFrame(R).T.to_csv(
                os.path.join(RESULTS, f"fase9_cifar_R_{mode}_seed{seed}.csv"))
            rows.append(dict(dataset="cifar100", mode=mode, seed=seed,
                             **metrics, ncdr=det[0], ncdr_delay=det[1],
                             ncdr_fpr=det[2]))
            print(f"[cifar {mode:12s} seed{seed}] acc={metrics['acc_final']:.3f} "
                  f"forget={metrics['forgetting']:.3f} bwt={metrics['bwt']:.3f} "
                  f"ncdr={det[0]:.2f} fpr={det[2]:.3f}")

    for seed in [42, 43, 44, 45, 46]:
        det = run_novelty(seed)
        rows.append(dict(dataset="synthetic_novelty", mode="sgnn_agb_v2",
                         seed=seed, ncdr=det[0], ncdr_delay=det[1],
                         ncdr_fpr=det[2]))
        print(f"[novelty seed{seed}] ncdr={det[0]:.2f} delay={det[1]} "
              f"fpr={det[2]:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "fase9_cl_metrics.csv"), index=False)
    print("\n=== RINGKASAN ===")
    print(summary.groupby(["dataset", "mode"]).mean(numeric_only=True)
          .round(3).to_string())
    print(f"Selesai {time.time() - t0:.0f} detik")


if __name__ == "__main__":
    main()
