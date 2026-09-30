# RL Switching Module for DMS-Raptor

Replaces hand-crafted threshold policies with learned switching strategies
using Contextual Bandits (LinUCB, Thompson Sampling) and Deep RL (DQN).

## Architecture

```
rl_switching/
├── config.py              # Feature names, hyperparameters, RLConfig
├── features.py            # Per-frame feature extraction + normalization
├── rewards.py             # Oracle IoU, confidence, hybrid reward functions
├── engine_rl.py           # Drop-in StreamingEngine replacement
├── agents/
│   ├── base.py            # Abstract BaseAgent interface
│   ├── linucb.py          # LinUCB contextual bandit
│   ├── thompson.py        # Thompson Sampling bandit
│   └── dqn.py             # DQN with target network + replay
├── data/
│   ├── collector.py       # Dual-model data collection → HDF5
│   ├── dataset.py         # SwitchingDataset for training
│   └── replay_buffer.py   # Experience replay for DQN
├── training/
│   ├── bandit_trainer.py  # Offline bandit training loop
│   └── dqn_trainer.py     # DQN training with episodes
└── evaluation/
    ├── metrics.py         # SwitchingMetrics computation
    └── evaluator.py       # RL vs heuristic comparison
```

## Quick Start

### Step 1: Collect training data (run both models on all frames)
```bash
# GPU recommended (~52 min for 78K frames)
python research/12_collect_rl_data.py --device cuda

# Quick test (500 frames)
python research/12_collect_rl_data.py --videos 633-01 --max-frames 500
```

### Step 2: Train bandit agents
```bash
# Both LinUCB and Thompson Sampling
python research/13_train_bandits.py

# With leave-one-out cross-validation
python research/13_train_bandits.py --loo

# LinUCB only with custom exploration
python research/13_train_bandits.py --agent linucb --alpha 0.5
```

### Step 3: Train DQN agent (requires PyTorch + GPU)
```bash
python research/14_train_dqn.py --device cuda

# Custom architecture
python research/14_train_dqn.py --device cuda --hidden 256 --layers 3 --lr 5e-4
```

### Step 4: Compare RL vs heuristic policies
```bash
python research/15_evaluate_rl.py
```

## How It Works

### Problem Formulation
Each frame is a decision point:
- **Context**: 12-dim feature vector (image proxies + temporal state)
- **Action**: {use YOLOv8n, use YOLOv8s}
- **Reward**: detection quality - latency penalty - switch penalty

### Agents

| Agent | Type | Training | Overhead | Exploration |
|-------|------|----------|----------|-------------|
| LinUCB | Contextual Bandit | Offline, O(d²)/frame | ~0.01ms | UCB bound |
| Thompson Sampling | Contextual Bandit | Offline, O(d²)/frame | ~0.01ms | Posterior sampling |
| DQN | Deep RL | Batch, GPU | ~0.05ms | ε-greedy |

### Feature Vector (12 dimensions)
1. Laplacian variance (sharpness)
2. Shannon entropy (texture complexity)
3. Tenengrad (gradient energy)
4. Color entropy (HSV)
5. NR-IQA score (image quality)
6. Brightness (mean intensity)
7. Edge density (Canny)
8. Local contrast (pixel std)
9. Previous frame confidence
10. Previous detection count
11. Previous action (0=n, 1=s)
12. Dwell time (frames since switch)

### Reward Function (oracle_iou, used during training)
- **n chosen, n sufficient (IoU≥0.5)**: +1.0 + latency bonus
- **n chosen, n insufficient**: -1.0
- **s chosen, s needed**: +1.0
- **s chosen, s wasteful**: +0.3
- **Switch penalty**: -0.05

## Integration with Existing Pipeline

The RL engine is a drop-in replacement for StreamingEngine:

```python
from rl_switching.agents.linucb import LinUCBAgent
from rl_switching.features import FeatureNormalizer
from rl_switching.engine_rl import RLStreamingEngine

agent = LinUCBAgent().load("checkpoints/linucb_agent.npz")
normalizer = FeatureNormalizer().load("checkpoints/normalizer.npz")

engine = RLStreamingEngine(
    source_path="video.mp4",
    cfg=cfg, model_n=model_n, model_s=model_s,
    inf=inf, agent=agent, normalizer=normalizer,
)
for result in engine.run():
    # Same FrameResult as StreamingEngine
    print(result.choice, result.T_total_ms)
summary = engine.get_summary()  # Same RunSummary format
```

## Expected Results

Based on the proxy correlation analysis (Phase 6):
- Laplacian: r=0.39 with disagreement
- Confidence: r=0.67 with disagreement
- A learned combination should achieve r>0.7

Expected improvement over conf_ema (86% coverage, 27% s-usage):
- **LinUCB**: 89-92% coverage, 15-25% s-usage
- **DQN**: 90-94% coverage, 12-20% s-usage

## Dependencies

```bash
pip install -r requirements_rl.txt
# For DQN: install PyTorch matching your CUDA version
```
