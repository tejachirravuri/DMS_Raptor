#!/usr/bin/env python3
"""
Phase 13: Train contextual bandit agents (LinUCB + Thompson Sampling).

Trains on pre-collected data from Phase 12.  Uses leave-one-video-out
cross-validation for robust evaluation.

Usage:
    # Default: train both agents, 10 epochs
    python research/13_train_bandits.py

    # LinUCB only with custom alpha
    python research/13_train_bandits.py --agent linucb --alpha 0.5

    # Quick test
    python research/13_train_bandits.py --epochs 2 --agent linucb

    # Custom data path
    python research/13_train_bandits.py --data-dir path/to/data_cache
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

from rl_switching.config import RLConfig, FEATURE_DIM, FEATURE_NAMES
from rl_switching.agents.linucb import LinUCBAgent
from rl_switching.agents.thompson import ThompsonSamplingAgent
from rl_switching.data.dataset import SwitchingDataset
from rl_switching.training.bandit_trainer import BanditTrainer


def main():
    parser = argparse.ArgumentParser(description="Phase 13: Train bandit agents")
    parser.add_argument("--data-dir", default=None,
                        help="Path to data_cache from Phase 12")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for trained agents")
    parser.add_argument("--agent", choices=["linucb", "thompson", "both"],
                        default="both", help="Which agent(s) to train")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per agent")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="LinUCB exploration parameter")
    parser.add_argument("--thompson-v", type=float, default=1.0,
                        help="Thompson Sampling noise variance")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for n-sufficiency")
    parser.add_argument("--lambda-latency", type=float, default=0.005,
                        help="Latency penalty weight in reward")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loo", action="store_true",
                        help="Leave-one-out cross-validation across videos")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(
        PROJECT_ROOT, "research_results", "rl_switching", "data_cache"
    )
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "research_results", "rl_switching", "bandits"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 13: Train Contextual Bandit Agents")
    print("=" * 70)
    print(f"  Data:       {data_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Agent(s):   {args.agent}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Alpha:      {args.alpha}")
    print(f"  IoU thresh: {args.iou_threshold}")
    print()

    # ── Load manifest to get video IDs ───────────────────────────────
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: No manifest found at {manifest_path}")
        print("Run Phase 12 first: python research/12_collect_rl_data.py")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)
    all_video_ids = sorted(manifest.keys())
    print(f"Available videos: {all_video_ids}\n")

    cfg = RLConfig(
        iou_threshold=args.iou_threshold,
        lambda_latency=args.lambda_latency,
        linucb_alpha=args.alpha,
        thompson_v=args.thompson_v,
        linucb_epochs=args.epochs,
        thompson_epochs=args.epochs,
    )

    if args.loo:
        # Leave-one-out cross-validation
        _run_loo(cfg, data_dir, output_dir, all_video_ids, args)
    else:
        # Simple 75/25 split
        _run_simple_split(cfg, data_dir, output_dir, all_video_ids, args)


def _run_simple_split(cfg, data_dir, output_dir, all_video_ids, args):
    """Train on first 75% of videos, evaluate on last 25%."""
    split = max(1, int(len(all_video_ids) * 0.75))
    train_ids = all_video_ids[:split]
    eval_ids = all_video_ids[split:]

    if not eval_ids:
        eval_ids = [train_ids[-1]]
        train_ids = train_ids[:-1]

    print(f"Train videos ({len(train_ids)}): {train_ids}")
    print(f"Eval videos  ({len(eval_ids)}):  {eval_ids}\n")

    train_data = SwitchingDataset(data_dir, video_ids=train_ids, normalize=True)
    eval_data = SwitchingDataset(data_dir, video_ids=eval_ids, normalize=True)

    # Share normalizer (fit on train data)
    eval_data.normalizer = train_data.normalizer

    print(f"Train frames: {len(train_data)}")
    print(f"Eval frames:  {len(eval_data)}\n")

    # Save normalizer
    norm_path = os.path.join(output_dir, "normalizer.npz")
    train_data.normalizer.save(norm_path)
    print(f"Normalizer saved: {norm_path}\n")

    agents_to_train = []
    if args.agent in ("linucb", "both"):
        agents_to_train.append(("linucb", LinUCBAgent(FEATURE_DIM, cfg.linucb_alpha)))
    if args.agent in ("thompson", "both"):
        agents_to_train.append(("thompson", ThompsonSamplingAgent(FEATURE_DIM, cfg.thompson_v)))

    all_results = {}
    for agent_name, agent in agents_to_train:
        print(f"\n{'─' * 50}")
        print(f"Training: {agent_name}")
        print(f"{'─' * 50}")

        trainer = BanditTrainer(agent, cfg, train_data, eval_data)
        result = trainer.train(
            num_epochs=args.epochs,
            seed=args.seed,
            verbose=True,
        )

        agent_dir = os.path.join(output_dir, agent_name)
        trainer.save_results(agent_dir)
        all_results[agent_name] = result

        # Print feature importance for LinUCB
        if hasattr(agent, "get_feature_importance"):
            imp = agent.get_feature_importance()
            print(f"\nFeature importance ({agent_name}):")
            for name, val in sorted(imp.items(), key=lambda x: -x[1]):
                bar = "█" * int(val * 50 / max(imp.values()))
                print(f"  {name:<20s} {val:.4f} {bar}")

    # Save combined results
    summary_path = os.path.join(output_dir, "bandit_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Training complete. Results in: {output_dir}")
    print(f"{'=' * 70}")


def _run_loo(cfg, data_dir, output_dir, all_video_ids, args):
    """Leave-one-out cross-validation."""
    loo_dir = os.path.join(output_dir, "loo")
    os.makedirs(loo_dir, exist_ok=True)

    agents_to_eval = []
    if args.agent in ("linucb", "both"):
        agents_to_eval.append("linucb")
    if args.agent in ("thompson", "both"):
        agents_to_eval.append("thompson")

    loo_results = {a: [] for a in agents_to_eval}

    for hold_idx, hold_id in enumerate(all_video_ids):
        train_ids = [v for v in all_video_ids if v != hold_id]
        eval_ids = [hold_id]

        print(f"\n{'=' * 50}")
        print(f"LOO fold {hold_idx+1}/{len(all_video_ids)}: hold out {hold_id}")
        print(f"{'=' * 50}")

        train_data = SwitchingDataset(data_dir, video_ids=train_ids, normalize=True)
        eval_data = SwitchingDataset(data_dir, video_ids=eval_ids, normalize=True)
        eval_data.normalizer = train_data.normalizer

        for agent_name in agents_to_eval:
            if agent_name == "linucb":
                agent = LinUCBAgent(FEATURE_DIM, cfg.linucb_alpha)
            else:
                agent = ThompsonSamplingAgent(FEATURE_DIM, cfg.thompson_v)

            trainer = BanditTrainer(agent, cfg, train_data, eval_data)
            result = trainer.train(num_epochs=args.epochs, seed=args.seed, verbose=False)

            eval_metrics = result.get("eval_log", [{}])
            final_eval = eval_metrics[-1] if eval_metrics else {}
            fold_result = {
                "hold_out": hold_id,
                "eval_reward": final_eval.get("mean_reward", 0),
                "eval_accuracy": final_eval.get("accuracy", 0),
                "eval_s_usage": final_eval.get("s_usage", 0),
            }
            loo_results[agent_name].append(fold_result)
            print(f"  {agent_name}: reward={fold_result['eval_reward']:.4f}, "
                  f"acc={fold_result['eval_accuracy']:.3f}, "
                  f"s%={fold_result['eval_s_usage']:.3f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("LOO Cross-Validation Summary")
    print(f"{'=' * 70}")
    for agent_name, folds in loo_results.items():
        rewards = [f["eval_reward"] for f in folds]
        accs = [f["eval_accuracy"] for f in folds]
        print(f"\n{agent_name}:")
        print(f"  Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        print(f"  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # Save
    loo_path = os.path.join(loo_dir, "loo_results.json")
    with open(loo_path, "w") as f:
        json.dump(loo_results, f, indent=2)
    print(f"\nSaved: {loo_path}")


if __name__ == "__main__":
    main()
