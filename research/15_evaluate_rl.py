#!/usr/bin/env python3
"""
Phase 15: Evaluate RL agents vs heuristic policies.

Loads trained agents from Phases 13/14 and compares them against
heuristic baselines (n_only, s_only, conf_ema, oracle) on the same
pre-collected data.

Produces:
  - Comparison table (printed + saved as JSON)
  - Per-video metrics
  - Pareto front analysis (latency vs accuracy)
  - Feature importance visualization data

Usage:
    # Full evaluation
    python research/15_evaluate_rl.py

    # Evaluate specific agents
    python research/15_evaluate_rl.py --agents linucb dqn

    # Custom data/model paths
    python research/15_evaluate_rl.py --data-dir path/to/data_cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Phase 15: Evaluate RL vs heuristic")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--bandit-dir", default=None,
                        help="Directory with trained bandit agents")
    parser.add_argument("--dqn-dir", default=None,
                        help="Directory with trained DQN agent")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--agents", nargs="*", default=["linucb", "thompson", "dqn"],
                        help="Which RL agents to evaluate")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Specific videos to evaluate on (default: all)")
    args = parser.parse_args()

    base_dir = os.path.join(PROJECT_ROOT, "research_results", "rl_switching")
    data_dir = args.data_dir or os.path.join(base_dir, "data_cache")
    bandit_dir = args.bandit_dir or os.path.join(base_dir, "bandits")
    dqn_dir = args.dqn_dir or os.path.join(base_dir, "dqn")
    output_dir = args.output_dir or os.path.join(base_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 15: RL vs Heuristic Policy Evaluation")
    print("=" * 70)

    from rl_switching.config import RLConfig, FEATURE_DIM
    from rl_switching.data.dataset import SwitchingDataset
    from rl_switching.features import FeatureNormalizer
    from rl_switching.evaluation.evaluator import PolicyEvaluator

    # ── Load data ────────────────────────────────────────────────────
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: No manifest at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.videos:
        video_ids = [v for v in manifest.keys()
                     if any(sel in v for sel in args.videos)]
    else:
        video_ids = sorted(manifest.keys())

    print(f"  Data dir:   {data_dir}")
    print(f"  Videos:     {video_ids}")
    print(f"  Agents:     {args.agents}")
    print()

    dataset = SwitchingDataset(data_dir, video_ids=video_ids, normalize=True)
    print(f"  Total frames: {len(dataset)}\n")

    cfg = RLConfig(iou_threshold=args.iou_threshold)
    evaluator = PolicyEvaluator(cfg, dataset)

    # ── Load trained agents ──────────────────────────────────────────
    if "linucb" in args.agents:
        linucb_path = os.path.join(bandit_dir, "linucb", "linucb_agent.npz")
        if os.path.exists(linucb_path):
            from rl_switching.agents.linucb import LinUCBAgent
            agent = LinUCBAgent(FEATURE_DIM).load(linucb_path)
            evaluator.add_agent("linucb", agent)
            print(f"  Loaded LinUCB: {linucb_path}")

            # Print feature importance
            imp = agent.get_feature_importance()
            print("  Feature importance:")
            for name, val in sorted(imp.items(), key=lambda x: -x[1])[:5]:
                print(f"    {name}: {val:.4f}")
        else:
            print(f"  LinUCB not found at {linucb_path}, skipping")

    if "thompson" in args.agents:
        ts_path = os.path.join(bandit_dir, "thompson", "thompson_agent.npz")
        if os.path.exists(ts_path):
            from rl_switching.agents.thompson import ThompsonSamplingAgent
            agent = ThompsonSamplingAgent(FEATURE_DIM).load(ts_path)
            evaluator.add_agent("thompson", agent)
            print(f"  Loaded Thompson: {ts_path}")
        else:
            print(f"  Thompson not found at {ts_path}, skipping")

    if "dqn" in args.agents:
        dqn_path = os.path.join(dqn_dir, "dqn_agent.pt")
        if os.path.exists(dqn_path):
            from rl_switching.agents.dqn import DQNAgent
            agent = DQNAgent(FEATURE_DIM, device="cpu").load(dqn_path)
            agent.eval_mode()
            evaluator.add_agent("dqn", agent)
            print(f"  Loaded DQN: {dqn_path}")
        else:
            print(f"  DQN not found at {dqn_path}, skipping")

    print()

    # ── Evaluate ─────────────────────────────────────────────────────
    print("Running evaluation...\n")
    results = evaluator.evaluate_all(video_ids=video_ids)

    # ── Print comparison table ───────────────────────────────────────
    print()
    table = evaluator.print_comparison_table(results)

    # ── Pareto analysis ──────────────────────────────────────────────
    print("\n\nPareto Front (Latency vs Routing Accuracy):")
    print("-" * 50)
    pareto_points = []
    for policy_name, metrics_list in results.items():
        agg = evaluator.aggregate_metrics(metrics_list)
        if agg:
            pareto_points.append({
                "policy": policy_name,
                "latency_ms": agg["mean_latency_ms"],
                "accuracy": agg["mean_routing_accuracy"],
                "s_usage": agg["mean_s_usage"],
                "reward": agg["mean_reward"],
            })

    # Sort by latency
    pareto_points.sort(key=lambda x: x["latency_ms"])
    for p in pareto_points:
        marker = " ★" if p["policy"] in ("linucb", "thompson", "dqn") else ""
        print(f"  {p['policy']:<16} lat={p['latency_ms']:.1f}ms  "
              f"acc={p['accuracy']:.3f}  s%={p['s_usage']:.3f}{marker}")

    # ── Per-video breakdown ──────────────────────────────────────────
    print("\n\nPer-Video Results:")
    print("-" * 70)
    for policy_name in results:
        if policy_name not in ("linucb", "thompson", "dqn", "conf_ema"):
            continue
        print(f"\n  {policy_name}:")
        for m in results[policy_name]:
            print(f"    {m.video_id}: acc={m.routing_accuracy:.3f} "
                  f"s%={m.s_usage:.3f} lat={m.mean_latency_ms:.1f}ms "
                  f"sw/100={m.sw_per_100:.1f}")

    # ── Save results ─────────────────────────────────────────────────
    result_path = evaluator.save_results(results, output_dir)
    print(f"\n{'=' * 70}")
    print(f"Evaluation complete. Results saved to: {result_path}")

    # Save Pareto data
    pareto_path = os.path.join(output_dir, "pareto_analysis.json")
    with open(pareto_path, "w") as f:
        json.dump(pareto_points, f, indent=2)

    # Save comparison table
    table_path = os.path.join(output_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table)

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
