#!/usr/bin/env python3
"""
Phase 14: Train DQN agent for temporal model switching.

Requires PyTorch.  Trains on pre-collected data from Phase 12.
Uses experience replay and target network for stability.

Usage:
    # Default training (GPU recommended)
    python research/14_train_dqn.py --device cuda

    # Quick test
    python research/14_train_dqn.py --episodes 10 --device cpu

    # Custom hyperparameters
    python research/14_train_dqn.py --device cuda --lr 5e-4 --gamma 0.99 --hidden 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Phase 14: Train DQN agent")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--epsilon-decay", type=int, default=50_000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--lambda-latency", type=float, default=0.005)
    parser.add_argument("--switch-penalty", type=float, default=0.05)
    args = parser.parse_args()

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
        if args.device.startswith("cuda") and torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: PyTorch required for DQN training.")
        print("Install: pip install torch  or  conda install pytorch -c pytorch")
        sys.exit(1)

    data_dir = args.data_dir or os.path.join(
        PROJECT_ROOT, "research_results", "rl_switching", "data_cache"
    )
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "research_results", "rl_switching", "dqn"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 14: Train DQN Agent")
    print("=" * 70)
    print(f"  Data:          {data_dir}")
    print(f"  Output:        {output_dir}")
    print(f"  Device:        {args.device}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Architecture:  {args.hidden} x {args.layers} layers")
    print(f"  LR:            {args.lr}")
    print(f"  Gamma:         {args.gamma}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Buffer:        {args.buffer_size}")
    print(f"  Eps decay:     {args.epsilon_decay} steps")
    print()

    from rl_switching.config import RLConfig, FEATURE_DIM
    from rl_switching.agents.dqn import DQNAgent
    from rl_switching.data.dataset import SwitchingDataset
    from rl_switching.training.dqn_trainer import DQNTrainer

    # ── Load manifest ────────────────────────────────────────────────
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: No manifest at {manifest_path}")
        print("Run Phase 12 first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)
    all_video_ids = sorted(manifest.keys())

    # Train/eval split: 75/25
    split = max(1, int(len(all_video_ids) * 0.75))
    train_ids = all_video_ids[:split]
    eval_ids = all_video_ids[split:]
    if not eval_ids:
        eval_ids = [train_ids[-1]]
        train_ids = train_ids[:-1]

    print(f"Train videos ({len(train_ids)}): {train_ids}")
    print(f"Eval videos  ({len(eval_ids)}):  {eval_ids}\n")

    # ── Load data ────────────────────────────────────────────────────
    train_data = SwitchingDataset(data_dir, video_ids=train_ids, normalize=True)
    eval_data = SwitchingDataset(data_dir, video_ids=eval_ids, normalize=True)
    eval_data.normalizer = train_data.normalizer

    print(f"Train frames: {len(train_data)}")
    print(f"Eval frames:  {len(eval_data)}\n")

    # Save normalizer
    norm_path = os.path.join(output_dir, "normalizer.npz")
    train_data.normalizer.save(norm_path)

    # ── Configure ────────────────────────────────────────────────────
    cfg = RLConfig(
        iou_threshold=args.iou_threshold,
        lambda_latency=args.lambda_latency,
        switch_penalty=args.switch_penalty,
        dqn_hidden=args.hidden,
        dqn_layers=args.layers,
        dqn_lr=args.lr,
        dqn_gamma=args.gamma,
        dqn_batch_size=args.batch_size,
        dqn_buffer_size=args.buffer_size,
        dqn_epsilon_decay=args.epsilon_decay,
        dqn_target_update=args.target_update,
        dqn_state_dim=FEATURE_DIM,
    )

    # ── Create agent ─────────────────────────────────────────────────
    agent = DQNAgent(
        state_dim=FEATURE_DIM,
        hidden=args.hidden,
        n_layers=args.layers,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        device=args.device,
    )

    # ── Train ────────────────────────────────────────────────────────
    trainer = DQNTrainer(agent, cfg, train_data, eval_data)

    print("Starting DQN training...\n")
    t_start = time.time()
    result = trainer.train(num_episodes=args.episodes, verbose=True)
    elapsed = time.time() - t_start

    # ── Save ─────────────────────────────────────────────────────────
    trainer.save_results(output_dir)

    print(f"\n{'=' * 70}")
    print(f"DQN training complete in {elapsed/60:.1f} minutes")
    print(f"  Best eval reward: {result['best_eval_reward']:.4f} "
          f"(episode {result['best_episode']})")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Results: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
