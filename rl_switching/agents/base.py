"""Abstract base class for all switching agents."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..config import NUM_ACTIONS, FEATURE_DIM


class BaseAgent(ABC):
    """Interface that all switching agents (bandit & RL) must implement."""

    name: str = "base"

    @abstractmethod
    def choose(self, context: np.ndarray) -> int:
        """Select an action given the current context/state.

        Parameters
        ----------
        context : np.ndarray
            Feature vector of shape (d,).

        Returns
        -------
        int
            Action index: 0 = n-model, 1 = s-model.
        """

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Update the agent after observing a reward.

        Parameters
        ----------
        context : np.ndarray
            Feature vector used when choosing the action.
        action : int
            Action that was taken.
        reward : float
            Observed reward.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent's learned parameters to disk."""

    @abstractmethod
    def load(self, path: str) -> "BaseAgent":
        """Restore the agent from a saved checkpoint."""

    def get_params(self) -> Dict[str, Any]:
        """Return a dict of learned parameters for inspection."""
        return {}

    def greedy_choose(self, context: np.ndarray) -> int:
        """Greedy action selection (no exploration). Default: same as choose."""
        return self.choose(context)
