"""
LinUCB contextual bandit agent for model switching.

Reference: Li et al., "A Contextual-Bandit Approach to Personalized
News Article Recommendation", WWW 2010.

Each arm (n-model, s-model) maintains a ridge regression model.
The UCB bonus encourages exploration of under-sampled context regions.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from .base import BaseAgent
from ..config import NUM_ACTIONS, FEATURE_DIM


class LinUCBAgent(BaseAgent):
    """Linear Upper Confidence Bound contextual bandit.

    Parameters
    ----------
    d : int
        Context dimension (number of features).
    alpha : float
        Exploration parameter.  Higher = more exploration.
        Typical range: 0.1 – 3.0.  Default 1.0.
    """

    name = "linucb"

    def __init__(self, d: int = FEATURE_DIM, alpha: float = 1.0):
        self.d = d
        self.alpha = alpha

        # Per-arm sufficient statistics for ridge regression.
        # A_a = X_a^T X_a + I  (d × d)
        # b_a = X_a^T r_a      (d,)
        self.A = {a: np.eye(d, dtype=np.float64) for a in range(NUM_ACTIONS)}
        self.b = {a: np.zeros(d, dtype=np.float64) for a in range(NUM_ACTIONS)}

        # Cache inverse for efficiency (recomputed lazily).
        self._A_inv = {a: np.eye(d, dtype=np.float64) for a in range(NUM_ACTIONS)}
        self._dirty = {a: False for a in range(NUM_ACTIONS)}

        # Statistics
        self.total_updates = 0
        self.arm_counts = {a: 0 for a in range(NUM_ACTIONS)}

    def _get_inv(self, a: int) -> np.ndarray:
        if self._dirty[a]:
            self._A_inv[a] = np.linalg.solve(
                self.A[a], np.eye(self.d, dtype=np.float64)
            )
            self._dirty[a] = False
        return self._A_inv[a]

    def choose(self, context: np.ndarray) -> int:
        """Select action with highest UCB score."""
        x = context.astype(np.float64).ravel()
        assert x.shape == (self.d,), f"Expected ({self.d},), got {x.shape}"

        best_action = 0
        best_score = -np.inf

        for a in range(NUM_ACTIONS):
            A_inv = self._get_inv(a)
            theta = A_inv @ self.b[a]       # learned weights
            pred = float(x @ theta)
            ucb = self.alpha * np.sqrt(float(x @ A_inv @ x))
            score = pred + ucb

            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def greedy_choose(self, context: np.ndarray) -> int:
        """Greedy: pick arm with highest predicted reward (no UCB bonus)."""
        x = context.astype(np.float64).ravel()
        best_action = 0
        best_pred = -np.inf

        for a in range(NUM_ACTIONS):
            A_inv = self._get_inv(a)
            theta = A_inv @ self.b[a]
            pred = float(x @ theta)
            if pred > best_pred:
                best_pred = pred
                best_action = a

        return best_action

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Update arm parameters after observing reward."""
        x = context.astype(np.float64).ravel()
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        self._dirty[action] = True
        self.total_updates += 1
        self.arm_counts[action] += 1

    def get_params(self) -> Dict[str, Any]:
        """Return learned weight vectors for each arm."""
        params = {}
        for a in range(NUM_ACTIONS):
            A_inv = self._get_inv(a)
            theta = A_inv @ self.b[a]
            arm_name = "n" if a == 0 else "s"
            params[f"theta_{arm_name}"] = theta.tolist()
        params["total_updates"] = self.total_updates
        params["arm_counts"] = dict(self.arm_counts)
        return params

    def get_feature_importance(self) -> Dict[str, float]:
        """Compute per-feature importance as |theta_s - theta_n|.

        Larger values indicate features that most strongly differentiate
        the switching decision.
        """
        from ..config import FEATURE_NAMES
        A_inv_n = self._get_inv(0)
        A_inv_s = self._get_inv(1)
        theta_n = A_inv_n @ self.b[0]
        theta_s = A_inv_s @ self.b[1]
        diff = np.abs(theta_s - theta_n)
        return {name: float(diff[i]) for i, name in enumerate(FEATURE_NAMES)}

    def save(self, path: str) -> None:
        np.savez(
            path,
            d=self.d, alpha=self.alpha,
            A_0=self.A[0], A_1=self.A[1],
            b_0=self.b[0], b_1=self.b[1],
            total_updates=self.total_updates,
            arm_count_0=self.arm_counts[0],
            arm_count_1=self.arm_counts[1],
        )

    def load(self, path: str) -> "LinUCBAgent":
        data = np.load(path)
        self.d = int(data["d"])
        self.alpha = float(data["alpha"])
        self.A = {0: data["A_0"], 1: data["A_1"]}
        self.b = {0: data["b_0"], 1: data["b_1"]}
        self._dirty = {0: True, 1: True}
        self._A_inv = {a: np.eye(self.d) for a in range(NUM_ACTIONS)}
        self.total_updates = int(data["total_updates"])
        self.arm_counts = {
            0: int(data["arm_count_0"]),
            1: int(data["arm_count_1"]),
        }
        return self
