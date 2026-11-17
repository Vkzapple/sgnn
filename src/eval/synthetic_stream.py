import numpy as np


def generate_stream(T=6000, d=16, n_classes_init=4, seed=0,
                    drift_points=(1500, 3500), novelty_points=(4800,)):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(n_classes_init, d))
    n_classes = n_classes_init

    X, labels = [], []
    for t in range(T):
        if t in drift_points:
            centers = centers + rng.normal(0, 4, size=centers.shape)
        if t in novelty_points:
            new_center = rng.normal(0, 3, size=(1, d))
            centers = np.vstack([centers, new_center])
            n_classes += 1

        cls = rng.integers(0, n_classes)
        x = centers[cls] + rng.normal(0, 0.5, size=d)
        X.append(x)
        labels.append(cls)

    return np.array(X), np.array(labels), list(drift_points), list(novelty_points)


def generate_stream_v2(T=6000, d=16, n_classes_init=4, seed=0,
                       drift_points=(1500, 3500), novelty_points=(4800, 5600),
                       growth_per_drift=3, sigma_cluster_growth=0.9):
    """Stream non-stasioner dengan kompleksitas MENINGKAT: tiap titik drift
    menambah komponen (kebutuhan kapasitas naik), tiap titik novelty menambah
    komponen baru. Dibuat untuk menekan jalur GROW pada AGC."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(n_classes_init, d))
    n_classes = n_classes_init

    X, labels = [], []
    for t in range(T):
        if t in drift_points:
            centers = centers + rng.normal(0, 4, size=centers.shape)
            new = rng.normal(0, 3, size=(growth_per_drift, d))
            centers = np.vstack([centers, new])
            n_classes += growth_per_drift
        if t in novelty_points:
            new_center = rng.normal(0, 3, size=(1, d))
            centers = np.vstack([centers, new_center])
            n_classes += 1

        cls = rng.integers(0, n_classes)
        x = centers[cls] + rng.normal(0, sigma_cluster_growth, size=d)
        X.append(x)
        labels.append(cls)

    return np.array(X), np.array(labels), list(drift_points), list(novelty_points)
