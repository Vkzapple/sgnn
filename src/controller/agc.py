"""Adaptive Growth Controller (AGC) — implementasi dari docs/agc_pseudocode.md."""

from collections import deque
import numpy as np

def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """MMD^2 tak berbias dengan kernel RBF, tanpa dependensi scipy."""
    if len(X) < 2 or len(Y) < 2:
        return 0.0
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def kernel(A, B):
        sq_a = np.sum(A ** 2, axis=1)[:, None]
        sq_b = np.sum(B ** 2, axis=1)[None, :]
        sqdist = sq_a + sq_b - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(sqdist, 0))

    Kxx, Kyy, Kxy = kernel(X, X), kernel(Y, Y), kernel(X, Y)
    m, n = len(X), len(Y)
    mmd2 = ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
            - 2 * Kxy.mean())
    return max(float(mmd2), 0.0)

class ReservoirBuffer:
    """Buffer kapasitas tetap dengan reservoir sampling sederhana."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.data = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, item):
        self.seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            j = self.rng.integers(0, self.seen)
            if j < self.capacity:
                self.data[j] = item

    def array(self):
        return np.array(self.data) if self.data else np.empty((0,))

    def __len__(self):
        return len(self.data)

class EWMAStat:
    """Mean & mean-abs-deviation via EWMA, dipakai untuk z-score per sinyal."""

    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self.mu = 0.0
        self.sigma = 1.0
        self.initialized = False

    def update(self, x: float):
        if not self.initialized:
            self.mu = x
            self.sigma = 1.0
            self.initialized = True
            return
        self.mu = self.decay * self.mu + (1 - self.decay) * x
        self.sigma = self.decay * self.sigma + (1 - self.decay) * abs(x - self.mu)

    def zscore(self, x: float, eps: float = 1e-6) -> float:
        return (x - self.mu) / (self.sigma + eps)

class AdaptiveGrowthController:
    def __init__(
        self,
        alpha=1.0, beta=1.0, gamma=1.0,
        k_sigma=2.0, lambda_prune=2.0, lambda_complex=0.01,
        loss_window=30, drift_window=50,
        ewma_decay=0.98,
        warmup_T=200, refractory_T=200,
        novelty_capacity=500, replay_capacity=100,
        delta_n_grow=8, n_prune=4, knn_k=5,
        seed=0,
    ):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.k_sigma, self.lambda_prune, self.lambda_complex = k_sigma, lambda_prune, lambda_complex
        self.warmup_T, self.refractory_T = warmup_T, refractory_T
        self.delta_n_grow, self.n_prune, self.knn_k = delta_n_grow, n_prune, knn_k

        self.pe_window = deque(maxlen=loss_window)
        self.window_recent = deque(maxlen=drift_window)
        self.window_ref = deque(maxlen=drift_window)

        self.stat_loss = EWMAStat(ewma_decay)
        self.stat_drift = EWMAStat(ewma_decay)
        self.stat_novelty = EWMAStat(ewma_decay)
        self.stat_S = EWMAStat(ewma_decay)

        self.novelty_buffer = ReservoirBuffer(novelty_capacity, seed)
        self.replay_buffer = ReservoirBuffer(replay_capacity, seed + 1)

        self.L_prev = 0.0
        self.warmup_counter = 0
        self.refractory_counter = 0
        self.status_drift = "NORMAL"
        self.B_t, self.P_t = 0.0, 0.0
        self.grow_seed = seed + 100

    def _update_loss_trend(self, pe_t):
        self.pe_window.append(pe_t)
        L_t = float(np.mean(self.pe_window))
        delta_L = L_t - self.L_prev
        self.L_prev = L_t
        return delta_L

    def _update_drift(self, x_t):
        self.window_recent.append(x_t)
        if len(self.window_ref) < self.window_ref.maxlen or len(self.window_recent) < self.window_recent.maxlen:
            if len(self.window_ref) < self.window_ref.maxlen:
                self.window_ref.append(x_t)
            return 0.0
        return _rbf_mmd2(np.array(self.window_ref), np.array(self.window_recent))

    def _update_novelty(self, x_t):
        buf = self.novelty_buffer.array()
        if len(buf) < self.knn_k:
            novelty_raw = 0.0
        else:
            dists = np.linalg.norm(buf - x_t, axis=1)
            k_nearest = np.sort(dists)[: self.knn_k]
            novelty_raw = float(np.mean(k_nearest))
        if len(self.novelty_buffer) < self.novelty_buffer.capacity or novelty_raw > self.stat_novelty.mu:
            self.novelty_buffer.add(x_t.copy())
        return novelty_raw

    def step(self, x_t: np.ndarray, pe_t: float, sgnn):
        delta_L = self._update_loss_trend(pe_t)
        drift_raw = self._update_drift(x_t)
        novelty_raw = self._update_novelty(x_t)

        in_cooldown = self.warmup_counter > 0 or self.refractory_counter > 0
        frozen_stats = in_cooldown or self.status_drift == "DRIFT"

        if not frozen_stats:
            self.stat_loss.update(delta_L)
            self.stat_drift.update(drift_raw)
            self.stat_novelty.update(novelty_raw)

        Z_loss = self.stat_loss.zscore(delta_L)
        Z_drift = self.stat_drift.zscore(drift_raw)
        Z_novelty = self.stat_novelty.zscore(novelty_raw)
        S_t = self.alpha * Z_loss + self.beta * Z_drift + self.gamma * Z_novelty

        if drift_raw >= self.stat_drift.mu + 3 * self.stat_drift.sigma:
            self.status_drift = "DRIFT"
        elif drift_raw >= self.stat_drift.mu + 2 * self.stat_drift.sigma:
            self.status_drift = "WARNING"
        else:
            self.status_drift = "NORMAL"

        if not frozen_stats:
            self.stat_S.update(S_t)
            self.B_t = self.stat_S.mu + self.k_sigma * self.stat_S.sigma
            self.P_t = self.stat_S.mu - self.lambda_prune * self.stat_S.sigma

        action = "STOP"
        g_m = None

        if in_cooldown:
            action = "STOP"
        elif S_t > self.B_t and len(self.replay_buffer) >= 5:
            g_m = self._try_grow(sgnn)
            if g_m is not None and g_m > 0:
                action = "GROW"
                self.refractory_counter = self.refractory_T
                self.warmup_counter = self.warmup_T
                self.window_ref = deque(self.window_recent, maxlen=self.window_recent.maxlen)
            else:
                action = "STOP"
        elif S_t < self.P_t:
            sgnn.prune(self.n_prune)
            action = "PRUNE"
            self.refractory_counter = self.refractory_T
            self.warmup_counter = self.warmup_T
            self.window_ref = deque(self.window_recent, maxlen=self.window_recent.maxlen)

        if self.warmup_counter > 0:
            self.warmup_counter -= 1
        if self.refractory_counter > 0:
            self.refractory_counter -= 1

        return {
            "delta_L": delta_L, "drift_raw": drift_raw, "novelty_raw": novelty_raw,
            "S_t": S_t, "B_t": self.B_t, "P_t": self.P_t,
            "status_drift": self.status_drift, "action": action,
            "g_m": g_m, "n_neurons": sgnn.N,
        }

    def _try_grow(self, sgnn):
        replay = self.replay_buffer.data
        pairs_valid = [(x, xn) for (x, xn) in replay if xn is not None]
        if len(pairs_valid) < 5:
            return None
        pe_before = sgnn.eval_pe_batch(pairs_valid)
        trial = sgnn.clone()
        trial.grow(self.delta_n_grow, seed=self.grow_seed)
        self.grow_seed += 1
        pe_after = trial.eval_pe_batch(pairs_valid)
        g_m = (pe_before - pe_after) - self.lambda_complex * (self.delta_n_grow / sgnn.N)
        if g_m > 0:
            sgnn.grow(self.delta_n_grow, seed=self.grow_seed)
        return g_m

    def update_replay(self, x_t, x_next):
        """Panggil tiap langkah setelah x_next diketahui, untuk isi replay buffer."""
        if len(self.replay_buffer) < self.replay_buffer.capacity or True:
            self.replay_buffer.add((x_t.copy(), x_next.copy()))