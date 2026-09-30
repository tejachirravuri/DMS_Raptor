"""Data collection, datasets, and replay buffers for RL training."""
from .replay_buffer import ReplayBuffer

# Lazy imports: collector and dataset require h5py which may not be
# installed on all machines.  Import them explicitly when needed:
#   from rl_switching.data.collector import DualModelCollector
#   from rl_switching.data.dataset import SwitchingDataset
