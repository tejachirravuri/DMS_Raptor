"""
Experience replay buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions
and supports uniform random sampling.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    state_dim : int
        Dimension of the state vector.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Sample a random batch of transitions.

        Returns
        -------
        states : (B, state_dim)
        actions : (B,)
        rewards : (B,)
        next_states : (B, state_dim)
        dones : (B,)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size
