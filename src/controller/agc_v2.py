"""AGC v2 — controller redesigned sesuai 5 requirement paper (Section IV.F)."""

from collections import deque
import numpy as np

from src.controller.agc import ReservoirBuffer, _rbf_mmd2


class RunningStat:
    """Mean & std via Welford (robust, dipakai setelah warmup selesai)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def sigma(self) -> float:
        return float(np.sqrt(max(self.m2 / max(self.n - 1, 1), 0.0)))


def _median_heuristic_gamma(X: np.ndarray, rng: np.random.Generator,
                            max_samples: int = 200) -> float:
    """gamma = 1/median(sqdist) — sensitif terhadap skala fitur (fix MMD 'mati')."""
    if len(X) < 2:
        return 1.0 / X.shape[1] if X.ndim == 2 and X.shape[1] else 1.0
    idx = rng.choice(len(X), size=min(len(X), max_samples), replace=False)
    A = X[idx]
    sq = np.sum(A ** 2, axis=1)[:, None] + np.sum(A ** 2, axis=1)[None, :] - 2 * A @ A.T
    iu = np.triu_indices(len(A), k=1)
    med = np.median(sq[iu])
    return 1.0 / med if med > 0 else 1.0


class AdaptiveGrowthControllerV2:
    def __init__(
        self,
        alpha=0.4, beta=0.3, gamma=0.3,
        k_sigma=1.5,
        loss_window=30,
        drift_window=100,
        ref_slide_every=100,
        warmup_T=200,
        refractory_T=200,
        novelty_capacity=500, replay_capacity=100,
        delta_n_grow=8, n_prune=4, knn_k=5,
        grow_q=0.95,
        prune_pe_q=0.20,
        n_floor=16,
        lag_steps=100,
        seed=0,
    ):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.k_sigma = k_sigma
        self.loss_window = loss_window
        self.drift_window = drift_window
        self.ref_slide_every = ref_slide_every
        self.warmup_T, self.refractory_T = warmup_T, refractory_T
        self.delta_n_grow, self.n_prune, self.knn_k = delta_n_grow, n_prune, knn_k
        self.grow_q, self.prune_pe_q, self.n_floor = grow_q, prune_pe_q, n_floor
        self.lag_steps = lag_steps

        self.pe_window = deque(maxlen=loss_window)
        self.window_recent = deque(maxlen=drift_window)
        self.window_ref = deque(maxlen=drift_window)
        self._ref_snapshot = None
        self._steps_since_ref_reset = 0
        self._gamma_mmd = None

        self._s_hist = deque(maxlen=1000)
        self._pe_hist = deque(maxlen=1000)
        self._novelty_hist = deque(maxlen=1000)
        self._pesmooth_hist = deque(maxlen=1000)

        self.novelty_buffer = ReservoirBuffer(novelty_capacity, seed)
        self.replay_buffer = ReservoirBuffer(replay_capacity, seed + 1)

        self.stat_loss = RunningStat()
        self.stat_drift = RunningStat()
        self.stat_novelty = RunningStat()
        self._warmup_pool = {"delta_L": [], "drift": [], "novelty": []}

        self.L_prev = 0.0
        self.warmup_counter = warmup_T
        self.refractory_counter = 0
        self.status_drift = "NORMAL"
        self.B_t, self.P_t = np.inf, -np.inf
        self.grow_seed = seed + 100
        self.rng = np.random.default_rng(seed + 7)

    def _update_loss_trend(self, pe_t):
        self.pe_window.append(pe_t)
        L_t = float(np.mean(self.pe_window))
        delta_L = L_t - self.L_prev
        self.L_prev = L_t
        return delta_L

    def _update_drift(self, x_t):
        self.window_recent.append(x_t)
        if len(self.window_recent) < self.drift_window:
            return 0.0
        if self._ref_snapshot is None or self._steps_since_ref_reset >= self.ref_slide_every:
            self._ref_snapshot = np.array(self.window_recent)
            self._steps_since_ref_reset = 0
            self._gamma_mmd = _median_heuristic_gamma(self._ref_snapshot, self.rng)
        self._steps_since_ref_reset += 1
        return _rbf_mmd2(self._ref_snapshot, np.array(self.window_recent),
                         gamma=self._gamma_mmd)

    def _update_novelty(self, x_t):
        buf = self.novelty_buffer.array()
        if len(buf) < self.knn_k:
            novelty_raw = 0.0
        else:
            dists = np.linalg.norm(buf - x_t, axis=1)
            k_nearest = np.sort(dists)[: self.knn_k]
            novelty_raw = float(np.mean(k_nearest))
        self._novelty_hist.append(novelty_raw)
        if len(self.novelty_buffer) < self.novelty_buffer.capacity:
            self.novelty_buffer.add(x_t.copy())
        elif len(self._novelty_hist) > self.warmup_T and \
                novelty_raw > np.percentile(self._novelty_hist, 90):
            self.novelty_buffer.add(x_t.copy())
        return novelty_raw

    def step(self, x_t: np.ndarray, pe_t: float, sgnn):
        delta_L = self._update_loss_trend(pe_t)
        drift_raw = self._update_drift(x_t)
        novelty_raw = self._update_novelty(x_t)

        self._pe_hist.append(pe_t)

        action = "MAINTAIN"
        g_m = None
        S_t = np.nan
        Z_novelty = np.nan

        if self.warmup_counter > 0:
            self._warmup_pool["delta_L"].append(delta_L)
            self._warmup_pool["drift"].append(drift_raw)
            self._warmup_pool["novelty"].append(novelty_raw)
            self.warmup_counter -= 1
            if self.warmup_counter == 0:
                self.stat_loss = RunningStat()
                [self.stat_loss.update(v) for v in self._warmup_pool["delta_L"]]
                self.stat_drift = RunningStat()
                [self.stat_drift.update(v) for v in self._warmup_pool["drift"]]
                self.stat_novelty = RunningStat()
                [self.stat_novelty.update(v) for v in self._warmup_pool["novelty"]]
                self._s_hist.clear()
        else:
            self.stat_loss.update(delta_L)
            self.stat_drift.update(drift_raw)
            self.stat_novelty.update(novelty_raw)

            Z_loss = (delta_L - self.stat_loss.mean) / (self.stat_loss.sigma + 1e-6)
            Z_drift = (drift_raw - self.stat_drift.mean) / (self.stat_drift.sigma + 1e-6)
            Z_novelty = (novelty_raw - self.stat_novelty.mean) / (self.stat_novelty.sigma + 1e-6)
            S_t = self.alpha * Z_loss + self.beta * Z_drift + self.gamma * Z_novelty

            if drift_raw >= self.stat_drift.mean + 3 * self.stat_drift.sigma:
                self.status_drift = "DRIFT"
            elif drift_raw >= self.stat_drift.mean + 2 * self.stat_drift.sigma:
                self.status_drift = "WARNING"
            else:
                self.status_drift = "NORMAL"

            self._s_hist.append(S_t)

            s_arr = np.array(self._s_hist)
            pe_arr = np.array(self._pe_hist)
            if len(s_arr) > self.lag_steps:
                hist_s = s_arr[:-self.lag_steps]
                self.B_t = float(np.percentile(hist_s, self.grow_q * 100))
                self.P_t = float(np.percentile(pe_arr[:-self.lag_steps],
                                               self.prune_pe_q * 100))

            pe_smooth = float(np.mean(list(self.pe_window)))
            in_refractory = self.refractory_counter > 0

            self._pesmooth_hist.append(pe_smooth)
            pe_hist = np.array(self._pesmooth_hist)
            pe_prune_thr = float(np.percentile(pe_hist[-500:], self.prune_pe_q * 100))
            self.P_t = pe_prune_thr

            if in_refractory:
                action = "MAINTAIN"
            elif (S_t > self.B_t or self.status_drift == "DRIFT") \
                    and len(self.replay_buffer) >= 5:
                g_m = self._try_grow(sgnn)
                if g_m is not None and g_m > 0:
                    action = "GROW"
                    self.refractory_counter = self.refractory_T
                else:
                    action = "MAINTAIN"
            elif len(pe_hist) > 500 and pe_smooth < pe_prune_thr \
                    and self.status_drift == "NORMAL" \
                    and sgnn.N > self.n_floor:
                sgnn.prune(self.n_prune)
                action = "PRUNE"
                self.refractory_counter = self.refractory_T

            if self.refractory_counter > 0:
                self.refractory_counter -= 1

        return {
            "delta_L": delta_L, "drift_raw": drift_raw, "novelty_raw": novelty_raw,
            "z_novelty": Z_novelty,
            "S_t": S_t,
            "B_t": self.B_t, "P_t": self.P_t,
            "status_drift": self.status_drift, "action": action,
            "g_m": g_m, "n_neurons": sgnn.N,
        }

    def _try_grow(self, sgnn):
        """Marginal gain yang adil: kandidat lama vs model yang di-grow
        dilatih sama lama pada replay buffer sebelum dibandingkan."""
        pairs = [(x, xn) for (x, xn) in self.replay_buffer.data if xn is not None]
        if len(pairs) < 5:
            return None
        steps = 30

        base = sgnn.clone()
        base.train_batch(pairs, steps=steps, lr=1e-2)
        pe_base = base.eval_pe_batch(pairs)
        trial = sgnn.clone()
        trial.grow(self.delta_n_grow, seed=self.grow_seed, zero_out=True)
        self.grow_seed += 1
        trial.train_batch(pairs, steps=steps, lr=1e-2)
        pe_trial = trial.eval_pe_batch(pairs)

        g_m = ((pe_base - pe_trial) / (pe_base + 1e-9)) \
            - self.lambda_complex() * (self.delta_n_grow / sgnn.N)
        if g_m > 0:
            sgnn.grow(self.delta_n_grow, seed=self.grow_seed, zero_out=True)
        return g_m

    def lambda_complex(self, lam=0.05):
        return lam

    def update_replay(self, x_t, x_next):
        self.replay_buffer.add((x_t.copy(), x_next.copy()))
