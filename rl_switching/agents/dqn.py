"""
Deep Q-Network (DQN) agent for temporal model switching.

Unlike bandit agents, DQN models the switching decision as a sequential
problem: the state includes temporal context (dwell time, recent switch
history), and the agent optimizes discounted future reward.

Architecture: MLP with configurable depth/width.
Techniques: experience replay, target network, linear epsilon decay.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseAgent
from ..config import NUM_ACTIONS, FEATURE_DIM


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "DQN agent requires PyTorch.  Install with:\n"
            "  pip install torch\n"
            "or:\n"
            "  conda install pytorch -c pytorch"
        )


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    """DQN agent with experience replay and target network.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector (features + temporal context).
    hidden : int
        Hidden layer width.
    n_layers : int
        Number of hidden layers.
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon_start, epsilon_end, epsilon_decay : float
        Linear epsilon-greedy schedule.
    target_update : int
        Steps between target network syncs.
    device : str
        "cpu" or "cuda".
    """

    name = "dqn"

    def __init__(
        self,
        state_dim: int = FEATURE_DIM,
        hidden: int = 128,
        n_layers: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 50_000,
        target_update: int = 500,
        device: str = "cpu",
    ):
        _check_torch()
        self.state_dim = state_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.device = torch.device(device)

        # Networks
        self.q_net = QNetwork(state_dim, NUM_ACTIONS, hidden, n_layers).to(self.device)
        self.target_net = QNetwork(state_dim, NUM_ACTIONS, hidden, n_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Step counter for epsilon schedule and target updates
        self.steps = 0
        self.train_steps = 0

        # Training mode flag
        self._training = True

    @property
    def epsilon(self) -> float:
        """Current epsilon value (linear decay)."""
        progress = min(1.0, self.steps / max(1, self.epsilon_decay))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def choose(self, context: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        self.steps += 1
        if self._training and np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        return self.greedy_choose(context)

    def greedy_choose(self, context: np.ndarray) -> int:
        """Greedy action: argmax Q(s, a)."""
        with torch.no_grad():
            state = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            return int(q_values.argmax(dim=1).item())

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Single training step on a batch from replay buffer.

        Parameters
        ----------
        states : (B, state_dim)
        actions : (B,) int
        rewards : (B,)
        next_states : (B, state_dim)
        dones : (B,) bool

        Returns
        -------
        float
            Training loss value.
        """
        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones.astype(np.float32)).to(self.device)

        # Current Q-values for taken actions
        q_current = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1).values
            q_target = r + self.gamma * q_next * (1.0 - d)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1

        # Sync target network
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Not used directly — DQN trains via train_step with batches."""
        pass

    def eval_mode(self) -> None:
        """Switch to evaluation mode (no exploration)."""
        self._training = False
        self.q_net.eval()

    def train_mode(self) -> None:
        """Switch to training mode (with exploration)."""
        self._training = True
        self.q_net.train()

    def get_params(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "train_steps": self.train_steps,
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
        }

    def save(self, path: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "train_steps": self.train_steps,
            "state_dim": self.state_dim,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "target_update": self.target_update,
        }, path)

    def load(self, path: str) -> "DQNAgent":
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.state_dim = checkpoint["state_dim"]
        self.hidden = checkpoint["hidden"]
        self.n_layers = checkpoint["n_layers"]
        self.gamma = checkpoint["gamma"]
        self.epsilon_start = checkpoint["epsilon_start"]
        self.epsilon_end = checkpoint["epsilon_end"]
        self.epsilon_decay = checkpoint["epsilon_decay"]
        self.target_update = checkpoint["target_update"]
        self.steps = checkpoint["steps"]
        self.train_steps = checkpoint["train_steps"]

        # Reconstruct networks with correct architecture
        self.q_net = QNetwork(
            self.state_dim, NUM_ACTIONS, self.hidden, self.n_layers
        ).to(self.device)
        self.target_net = QNetwork(
            self.state_dim, NUM_ACTIONS, self.hidden, self.n_layers
        ).to(self.device)

        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer = optim.Adam(self.q_net.parameters())
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        return self
