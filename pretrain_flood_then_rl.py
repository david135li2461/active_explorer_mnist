"""Behavioral-clone FloodPolicy to initialize PPO policy, then continue RL training.

This script:
- Collects observation->action pairs by running the `FloodPolicy` for many steps.
- Performs supervised behavioral cloning on the SB3 policy network to imitate FloodPolicy.
- Continues RL training from the cloned initialization.

Usage (dry-run):
    python pretrain_flood_then_rl.py --classifier ./mnist_cnn.pth --bc-samples 10000 --bc-epochs 5 --rl-timesteps 50000
"""

from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from active_mnist_env import ActiveExplorerMNISTEnv
from test_policy_runner import FloodPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def collect_bc_data(env: ActiveExplorerMNISTEnv, n_samples: int) -> (np.ndarray, np.ndarray):
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    policy = FloodPolicy(env)
    collected = 0
    while collected < n_samples:
        o, _ = env.reset()
        done = False
        while not done and collected < n_samples:
            a = policy.act(o)
            obs_list.append(o.copy())
            act_list.append(int(a))
            o, rew, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            collected += 1
    return np.stack(obs_list, axis=0), np.array(act_list, dtype=np.int64)


def behavioral_clone(model: PPO, obs: np.ndarray, acts: np.ndarray, epochs: int = 5, batch_size: int = 64, lr: float = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.policy.to(device)
    dataset = TensorDataset(torch.from_numpy(obs).float(), torch.from_numpy(acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    loss_fn = lambda logp: -logp.mean()

    for ep in range(epochs):
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # forward through policy to get latent for pi
            # Use the policy's feature extractor + mlp_extractor to obtain the
            # latent_pi that the action net expects.
            with torch.no_grad():
                features = model.policy.extract_features(xb)
            # mlp_extractor returns (latent_pi, latent_vf)
            latent_pi, _ = model.policy.mlp_extractor(features)
            dist = model.policy._get_action_dist_from_latent(latent_pi)
            # PyTorch distribution may return log_prob shape depending on action space
            logp = dist.log_prob(yb)
            if logp.dim() > 1:
                logp = logp.sum(dim=1)
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"BC epoch {ep+1}/{epochs}: loss={total_loss/n:.6f}")


def make_env_fn(classifier_path: str, threshold: float, seed: int | None = None):
    def _f():
        return ActiveExplorerMNISTEnv(classifier_path=classifier_path, confidence_threshold=threshold, seed=seed)
    return _f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, default='./mnist_cnn.pth')
    parser.add_argument('--bc-samples', type=int, default=10000)
    parser.add_argument('--bc-epochs', type=int, default=5)
    parser.add_argument('--bc-batch', type=int, default=64)
    parser.add_argument('--rl-timesteps', type=int, default=50000)
    parser.add_argument('--save-path', type=str, default='./ppo_amnist_bc')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    env0 = ActiveExplorerMNISTEnv(classifier_path=args.classifier, confidence_threshold=0.90, seed=args.seed)
    print(f"Collecting {args.bc_samples} BC samples using FloodPolicy...")
    obs, acts = collect_bc_data(env0, args.bc_samples)
    print("Collected", obs.shape[0], "samples")

    # create PPO model (untrained)
    vec = DummyVecEnv([make_env_fn(args.classifier, 0.90, seed=args.seed)])
    model = PPO('MlpPolicy', vec, verbose=1, seed=args.seed)

    print("Running behavioral cloning to initialize policy...")
    behavioral_clone(model, obs, acts, epochs=args.bc_epochs, batch_size=args.bc_batch)

    # save intermediate model
    model.save(args.save_path + '/ppo_pretrained')
    print("Saved pretrained model to", args.save_path + '/ppo_pretrained.zip')

    # continue RL
    print(f"Continuing RL for {args.rl_timesteps} timesteps")
    model.learn(total_timesteps=args.rl_timesteps)
    model.save(args.save_path + '/ppo_final')
    print("Saved final model to", args.save_path + '/ppo_final.zip')


if __name__ == '__main__':
    main()
