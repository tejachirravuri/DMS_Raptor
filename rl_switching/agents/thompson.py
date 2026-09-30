"""
Thompson Sampling contextual bandit with Bayesian linear regression.

Each arm maintains a posterior N(mu_a, v^2 * B_a^{-1}) over weight
vectors.  At decision time, we sample theta ~ posterior and pick the
arm with the highest predicted reward x^T theta.

Exploration decays naturally as the posterior tightens with more data.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseAgent
from ..config import NUM_ACTIONS, FEATURE_DIM


class ThompsonSamplingAgent(BaseAgent):
    """Bayesian Linear Thompson Sampling for contextual bandits.

    Parameters
    ----------
    d : int
        Context dimension.
    v : float
        Observation noise standard deviation.  Controls initial
        exploration breadth.  Typical: 0.5 – 2.0.
    """

    name = "thompson"

    def __init__(self, d: int = FEATURE_DIM, v: float = 1.0):
        self.d = d
        self.v = v

        # Per-arm: B = X^T X + I,  f = X^T r
        # Posterior mean: mu = B^{-1} f
        # Posterior cov:  Sigma = v^2 * B^{-1}
        self.B = {a: np.eye(d, dtype=np.float64) for a in range(NUM_ACTIONS)}
        self.f = {a: np.zeros(d, dtype=np.float64) for a in range(NUM_ACTIONS)}

        self._B_inv = {a: np.eye(d, dtype=np.float64) for a in range(NUM_ACTIONS)}
        self._dirty = {a: False for a in range(NUM_ACTIONS)}

        self.total_updates = 0
        self.arm_counts = {a: 0 for a in range(NUM_ACTIONS)}

    def _get_inv(self, a: int) -> np.ndarray:
        if self._dirty[a]:
            self._B_inv[a] = np.linalg.solve(
                self.B[a], np.eye(self.d, dtype=np.float64)
            )
            self._dirty[a] = False
        return self._B_inv[a]

    def choose(self, context: np.ndarray) -> int:
        """Sample theta from posterior, pick arm with max x^T theta."""
        x = context.astype(np.float64).ravel()
        best_action = 0
        best_score = -np.inf

        for a in range(NUM_ACTIONS):
            B_inv = self._get_inv(a)
            mu = B_inv @ self.f[a]
            # Sample from posterior
            try:
                theta = np.random.multivariate_normal(
                    mu, self.v ** 2 * B_inv
                )
            except np.linalg.LinAlgError:
                # Fallback: diagonal approximation if covariance is singular
                diag_var = self.v ** 2 * np.diag(B_inv)
                theta = np.random.normal(mu, np.sqrt(np.maximum(diag_var, 1e-8)))

            score = float(x @ theta)
            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def greedy_choose(self, context: np.ndarray) -> int:
        """Greedy: pick arm with highest posterior mean prediction."""
        x = context.astype(np.float64).ravel()
        best_action = 0
        best_pred = -np.inf

        for a in range(NUM_ACTIONS):
            B_inv = self._get_inv(a)
            mu = B_inv @ self.f[a]
            pred = float(x @ mu)
            if pred > best_pred:
                best_pred = pred
                best_action = a

        return best_action

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = context.astype(np.float64).ravel()
        self.B[action] += np.outer(x, x)
        self.f[action] += reward * x
        self._dirty[action] = True
        self.total_updates += 1
        self.arm_counts[action] += 1

    def get_params(self) -> Dict[str, Any]:
        params = {}
        for a in range(NUM_ACTIONS):
            B_inv = self._get_inv(a)
            mu = B_inv @ self.f[a]
            arm_name = "n" if a == 0 else "s"
            params[f"mu_{arm_name}"] = mu.tolist()
            params[f"uncertainty_{arm_name}"] = float(
                np.sqrt(np.diag(self.v ** 2 * B_inv)).mean()
            )
        params["total_updates"] = self.total_updates
        return params

    def save(self, path: str) -> None:
        np.savez(
            path,
            d=self.d, v=self.v,
            B_0=self.B[0], B_1=self.B[1],
            f_0=self.f[0], f_1=self.f[1],
            total_updates=self.total_updates,
            arm_count_0=self.arm_counts[0],
            arm_count_1=self.arm_counts[1],
        )

    def load(self, path: str) -> "ThompsonSamplingAgent":
        data = np.load(path)
        self.d = int(data["d"])
        self.v = float(data["v"])
        self.B = {0: data["B_0"], 1: data["B_1"]}
        self.f = {0: data["f_0"], 1: data["f_1"]}
        self._dirty = {0: True, 1: True}
        self._B_inv = {a: np.eye(self.d) for a in range(NUM_ACTIONS)}
        self.total_updates = int(data["total_updates"])
        self.arm_counts = {
            0: int(data["arm_count_0"]),
            1: int(data["arm_count_1"]),
        }
        return self
