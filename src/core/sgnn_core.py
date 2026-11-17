import copy
import numpy as np


class SGNNCore:
    def __init__(self, d_in: int, n0: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.d = d_in
        self.N = n0
        self.W1 = rng.normal(0, 0.1, size=(d_in, n0))
        self.b1 = np.zeros(n0)
        self.W2 = rng.normal(0, 0.1, size=(n0, d_in))
        self.b2 = np.zeros(d_in)
        self._init_adam()

    def _init_adam(self):
        self.m = {"W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
                  "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2)}
        self.v = {k: np.zeros_like(val) for k, val in self.m.items()}
        self.t = 0

    def forward(self, x: np.ndarray):
        h = np.tanh(x @ self.W1 + self.b1)
        x_pred = h @ self.W2 + self.b2
        return h, x_pred

    def backward_and_step(self, x, x_next, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        h, x_pred = self.forward(x)
        d_pred = 2 * (x_pred - x_next)
        dW2 = np.outer(h, d_pred)
        db2 = d_pred
        dh = d_pred @ self.W2.T
        dh_pre = dh * (1 - h ** 2)
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        self.t += 1
        for k, g in grads.items():
            p = getattr(self, k)
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        pe = float(np.sum((x_next - x_pred) ** 2))
        return pe

    def grow(self, delta_n: int, seed=None, zero_out: bool = False):
        """zero_out=True: baris W2 neuron baru di-nol-kan, sehingga grow
        TIDAK mendisturb output (PE tidak melonjak sesaat setelah GROW)."""
        rng = np.random.default_rng(seed)
        new_W1 = rng.normal(0, 0.1, size=(self.d, delta_n))
        new_b1 = np.zeros(delta_n)
        new_W2 = np.zeros((delta_n, self.d)) if zero_out else \
            rng.normal(0, 0.1, size=(delta_n, self.d))

        self.W1 = np.hstack([self.W1, new_W1])
        self.b1 = np.concatenate([self.b1, new_b1])
        self.W2 = np.vstack([self.W2, new_W2])
        self.N += delta_n

        self.m["W1"] = np.hstack([self.m["W1"], np.zeros((self.d, delta_n))])
        self.v["W1"] = np.hstack([self.v["W1"], np.zeros((self.d, delta_n))])
        self.m["b1"] = np.concatenate([self.m["b1"], np.zeros(delta_n)])
        self.v["b1"] = np.concatenate([self.v["b1"], np.zeros(delta_n)])
        self.m["W2"] = np.vstack([self.m["W2"], np.zeros((delta_n, self.d))])
        self.v["W2"] = np.vstack([self.v["W2"], np.zeros((delta_n, self.d))])

    def prune(self, n_remove: int):
        if n_remove <= 0 or self.N - n_remove < 4:
            return
        contrib = np.sum(np.abs(self.W1), axis=0) * np.sum(np.abs(self.W2), axis=1)
        idx_remove = np.argsort(contrib)[:n_remove]
        keep = np.setdiff1d(np.arange(self.N), idx_remove)

        self.W1 = self.W1[:, keep]
        self.b1 = self.b1[keep]
        self.W2 = self.W2[keep, :]
        self.N = len(keep)

        self.m["W1"] = self.m["W1"][:, keep]
        self.v["W1"] = self.v["W1"][:, keep]
        self.m["b1"] = self.m["b1"][keep]
        self.v["b1"] = self.v["b1"][keep]
        self.m["W2"] = self.m["W2"][keep, :]
        self.v["W2"] = self.v["W2"][keep, :]

    def clone(self):
        return copy.deepcopy(self)

    def eval_pe_batch(self, X_pairs):
        """X_pairs: list of (x_t, x_next) unlabeled -> rata-rata predictive error."""
        if len(X_pairs) == 0:
            return 0.0
        total = 0.0
        for x, x_next in X_pairs:
            _, x_pred = self.forward(x)
            total += np.sum((x_next - x_pred) ** 2)
        return total / len(X_pairs)

    def train_batch(self, X_pairs, steps: int = 20, lr: float = 1e-2):
        """Beberapa langkah update online pada replay buffer (untuk uji
        marginal gain yang adil: kedua kandidat dilatih sama lama)."""
        if len(X_pairs) == 0:
            return
        rng = np.random.default_rng(self.t)
        for _ in range(steps):
            x, x_next = X_pairs[rng.integers(0, len(X_pairs))]
            self.backward_and_step(x, x_next, lr=lr)
