"""Train an active-explorer agent on the ActiveExplorerMNISTEnv using PPO (stable-baselines3).

This script creates a single-process vectorized env and trains a PPO agent with an MLP policy.

Example:
    python train_explorer_ppo.py --classifier ./mnist_cnn.pth --timesteps 20000 --threshold 0.90

Notes:
- Requires the dependencies in requirements.txt (stable-baselines3, gymnasium, torch, torchvision, opencv-python).
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from active_mnist_env import ActiveExplorerMNISTEnv


def make_env_fn(classifier_path: str, threshold: float, seed: int | None = None) -> Callable:
    def _f():
        # Use flat observations for MLP policy compatibility
        env = ActiveExplorerMNISTEnv(classifier_path=classifier_path, confidence_threshold=threshold, seed=seed)
        return Monitor(env)

    return _f


def evaluate(policy, env, n_episodes: int = 10):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total = 0.0
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            total += rew
        rewards.append(total)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default="./mnist_cnn.pth", help="Path to classifier state dict")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--save-path", type=str, default="./ppo_explorer")
    parser.add_argument("--resume-path", type=str, default=None, help="Path to saved PPO .zip to resume from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # build env
    env = DummyVecEnv([make_env_fn(args.classifier, args.threshold, seed=args.seed)])

    # If a resume path is provided and exists, load it and attach the env.
    if args.resume_path is not None and os.path.exists(args.resume_path):
        print(f"Resuming model from {args.resume_path}")
        model = PPO.load(args.resume_path, env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)

    print(f"Starting training for {args.timesteps} timesteps")
    # Train in chunks so we can display a reliable tqdm progress bar and ETA.
    total = int(args.timesteps)
    chunk = max(2000, total // 50)  # at least 2000 steps per chunk, ~50 updates
    learned = 0
    with tqdm(total=total, desc='PPO training', unit='steps') as pbar:
        while learned < total:
            this_chunk = min(chunk, total - learned)
            model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)
            learned += this_chunk
            pbar.update(this_chunk)
    model_path = os.path.join(args.save_path, "ppo_explorer.zip")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Simple evaluation using the non-vectorized env for easier interaction
    eval_env = ActiveExplorerMNISTEnv(classifier_path=args.classifier, confidence_threshold=args.threshold, seed=args.seed)
    mean_reward = np.mean(evaluate(model, eval_env, n_episodes=10))
    print(f"Evaluation mean reward over 10 episodes: {mean_reward}")


if __name__ == "__main__":
    main()
