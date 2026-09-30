"""Evaluation and comparison framework."""
from .metrics import SwitchingMetrics

# PolicyEvaluator requires h5py (via SwitchingDataset).
# Import explicitly when needed:
#   from rl_switching.evaluation.evaluator import PolicyEvaluator
