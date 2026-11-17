"""Metrik continual learning: forgetting, BWT, dan NCDR."""

import numpy as np


def hidden_matrix(sgnn, X):
    H = np.empty((len(X), sgnn.N))
    for i, x in enumerate(X):
        h, _ = sgnn.forward(x)
        H[i] = h
    return H


def knn_task_acc(H_ref, y_ref, H_qry, y_qry, k=5):
    correct = 0
    for h, y in zip(H_qry, y_qry):
        dists = np.linalg.norm(H_ref - h, axis=1)
        nn = np.argsort(dists)[:k]
        pred = np.bincount(y_ref[nn]).argmax()
        correct += int(pred == y)
    return correct / max(len(H_qry), 1)


def evaluate_seen_tasks(sgnn, X, y, tasks, seen):
    scores = {}
    for t in seen:
        idx = np.where(tasks == t)[0]
        half = len(idx) // 2
        H = hidden_matrix(sgnn, X[idx])
        scores[int(t)] = knn_task_acc(H[:half], y[idx][:half],
                                      H[half:], y[idx][half:])
    return scores


def summarize(R, n_tasks):
    final = np.array([R[t][n_tasks - 1] for t in range(n_tasks)])
    forgetting = np.array([
        max(R[t][b] for b in range(t, n_tasks)) - R[t][n_tasks - 1]
        for t in range(n_tasks)
    ])
    bwt = np.array([
        R[t][n_tasks - 1] - R[t][t] for t in range(n_tasks - 1)
    ])
    learned = np.array([R[t][t] for t in range(n_tasks)])
    return {
        "acc_final": final.mean(),
        "learning_acc": learned.mean(),
        "forgetting": forgetting.mean(),
        "bwt": bwt.mean(),
    }


def ncdr_from_log(z, points, window=200, thr=3.0, exclude=()):
    z = np.asarray(z, dtype=float)
    detected, delays = 0, []
    for p in points:
        seg = np.where(z[p:p + window] > thr)[0]
        if len(seg):
            detected += 1
            delays.append(int(seg[0]))
    mask = np.ones(len(z), dtype=bool)
    for p in list(points) + list(exclude):
        mask[max(0, p - window):min(len(z), p + window)] = False
    valid = ~np.isnan(z) & mask
    fpr = float(np.mean(z[valid] > thr)) if valid.any() else np.nan
    delay = float(np.mean(delays)) if delays else np.nan
    return detected / max(len(points), 1), delay, fpr
