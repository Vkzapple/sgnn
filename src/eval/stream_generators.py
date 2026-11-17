"""Generator stream standar MOA (reimplementasi NumPy, tanpa dependensi Java):"""

import numpy as np

def generate_sea(T=6000, seed=0, drift_points=(1500, 3500),
                 thetas=(8.0, 9.0, 7.0), n_features=3, noise=0.1,
                 gradual_steps=0):
    """SEA Concepts. gradual_steps > 0 -> drift bertahap (interpolasi theta)."""
    rng = np.random.default_rng(seed)
    theta_seq = np.empty(T)
    theta_seq[:] = thetas[0]
    cur = thetas[0]
    for i, dp in enumerate(drift_points):
        nxt = thetas[(i + 1) % len(thetas)]
        if gradual_steps <= 0:
            theta_seq[dp:] = nxt
        else:
            for t in range(dp, min(dp + gradual_steps, T)):
                frac = (t - dp) / gradual_steps
                theta_seq[t] = cur + frac * (nxt - cur)
            theta_seq[min(dp + gradual_steps, T):] = nxt
        cur = nxt

    X = rng.uniform(0, 10, size=(T, n_features))
    y = (X[:, 0] + X[:, 1] <= theta_seq).astype(int)
    flip = rng.random(T) < noise
    y[flip] = 1 - y[flip]
    return X, y, list(drift_points)

def generate_random_rbf(T=6000, seed=0, drift_points=(1500, 3500),
                        n_features=10, n_centroids=50, n_classes=4,
                        drift_speed=0.05):
    """Random RBF ala MOA: centroid acak + bobot + simpangan; titik
    dibangkitkan di sekitar centroid terpilih (bobot), label = kelas centroid."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, size=(n_centroids, n_features))
    weights = rng.uniform(0, 1, size=n_centroids)
    weights /= weights.sum()
    sigmas = rng.uniform(0.1, 0.5, size=n_centroids)
    cls = rng.integers(0, n_classes, size=n_centroids)
    velocities = rng.normal(0, drift_speed, size=(n_centroids, n_features))

    X = np.empty((T, n_features))
    y = np.empty(T, dtype=int)
    for t in range(T):
        if t in drift_points:
            centers = centers + velocities * rng.uniform(50, 100)
        ci = rng.choice(n_centroids, p=weights)
        x = centers[ci] + rng.normal(0, sigmas[ci], size=n_features)
        X[t] = x
        y[t] = cls[ci]
    return X, y, list(drift_points)
