"""Fase 13 — mitigasi forgetting dengan replay online pada Split-CIFAR100."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.controller.agc import ReservoirBuffer
from src.controller.agc_v2 import AdaptiveGrowthControllerV2
from src.core.sgnn_core import SGNNCore
from experiments.cl_metrics import evaluate_seen_tasks, summarize
from experiments.run_fase6_cifar100 import features_from_rgb, load_split

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
N_TASKS = 10
MODES = ["sgnn_agb_v2", "sgnn_agb_v2_replay", "static"]


def run_cifar(X, y, tasks, seed, mode):
    sgnn = SGNNCore(d_in=X.shape[1], n0=64, seed=seed)
    use_agc = mode.startswith("sgnn")
    agc = AdaptiveGrowthControllerV2(seed=seed) if use_agc else None
    replay = ReservoirBuffer(200, seed + 1) if mode.endswith("replay") else None
    rng = np.random.default_rng(seed + 2)

    bounds = [int(np.where(tasks == t)[0][0]) for t in range(1, N_TASKS)]
    R = {t: {} for t in range(N_TASKS)}
    done = 0

    for i in range(len(X) - 1):
        pe = sgnn.backward_and_step(X[i], X[i + 1], lr=1e-2)
        if agc is not None:
            agc.update_replay(X[i], X[i + 1])
            agc.step(X[i], pe, sgnn)
        if replay is not None and len(replay) >= 20:
            j = int(rng.integers(0, len(replay)))
            xa, xb = replay.data[j]
            sgnn.backward_and_step(xa, xb, lr=0.5e-2)
        if replay is not None:
            replay.add((X[i].copy(), X[i + 1].copy()))

        if done < len(bounds) and i == bounds[done] - 1:
            scores = evaluate_seen_tasks(sgnn, X, y, tasks, range(done + 1))
            for t, v in scores.items():
                R[t][done] = v
            done += 1

    scores = evaluate_seen_tasks(sgnn, X, y, tasks, range(N_TASKS))
    for t, v in scores.items():
        R[t][N_TASKS - 1] = v
    return R, summarize(R, N_TASKS)


def main():
    t0 = time.time()
    Xrgb, y, tasks = load_split(per_task=600)
    X = features_from_rgb(Xrgb, tasks)
    print(f"Split-CIFAR100: {X.shape}, {N_TASKS} task")

    rows = []
    for seed in [0, 1, 2]:
        for mode in MODES:
            R, metrics = run_cifar(X, y, tasks, seed, mode)
            pd.DataFrame(R).T.to_csv(
                os.path.join(RESULTS, f"fase13_cifar_R_{mode}_seed{seed}.csv"))
            rows.append(dict(mode=mode, seed=seed, **metrics))
            print(f"[{mode:20s} seed{seed}] acc={metrics['acc_final']:.3f} "
                  f"learn={metrics['learning_acc']:.3f} "
                  f"forget={metrics['forgetting']:.3f} "
                  f"bwt={metrics['bwt']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "fase13_forgetting.csv"), index=False)
    summary = df.groupby("mode")[["acc_final", "learning_acc", "forgetting",
                                  "bwt"]].mean().round(3)
    summary.to_csv(os.path.join(RESULTS, "fase13_forgetting_summary.csv"))
    print("\n=== RINGKASAN (mean 3 seed) ===")
    print(summary.to_string())
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
