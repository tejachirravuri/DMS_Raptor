"""Agent implementations for RL-based model switching."""
from .base import BaseAgent
from .linucb import LinUCBAgent
from .thompson import ThompsonSamplingAgent

__all__ = ["BaseAgent", "LinUCBAgent", "ThompsonSamplingAgent"]

# DQN imported separately to avoid hard torch dependency:
#   from rl_switching.agents.dqn import DQNAgent
