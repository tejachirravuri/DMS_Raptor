"""
rl_switching — Reinforcement Learning & Contextual Bandit model switching.

Replaces hand-crafted threshold policies with learned switching strategies
for the DMS-Raptor UAV inspection framework.

Modules
-------
config      : RL-specific configuration dataclasses.
features    : Per-frame feature extraction (wraps existing proxy functions).
rewards     : Reward functions for bandit and RL training.
agents/     : LinUCB, Thompson Sampling, DQN agent implementations.
data/       : Data collection and dataset management.
training/   : Offline bandit and online DQN training loops.
evaluation/ : Metrics, evaluation, and comparison framework.
engine_rl   : Drop-in RL-based StreamingEngine replacement.
"""
