"""Visualize a few episodes from the saved 2M PPO policy using cv2 windows.

This script uses the PyTorch state-dict fallback to avoid cloudpickle issues.
Run with:
    conda run -n amnist-eval python3 visualize_2M.py --saved-path ./ppo_amnist_2M/ppo_explorer.zip --classifier ./mnist_cnn.pth --episodes 5

Requirements: opencv-python installed and a display available.
"""
from __future__ import annotations
import io
import zipfile
import argparse
import time
import os

import torch
from stable_baselines3 import PPO

from active_mnist_env import ActiveExplorerMNISTEnv


def load_policy_from_zip(saved_path: str, env: ActiveExplorerMNISTEnv):
    if not zipfile.is_zipfile(saved_path):
        raise RuntimeError('Not a zip file')
    with zipfile.ZipFile(saved_path, 'r') as z:
        if 'policy.pth' not in z.namelist():
            raise RuntimeError('policy.pth not found in archive')
        data = z.read('policy.pth')
    state = torch.load(io.BytesIO(data), map_location='cpu')
    model = PPO('MlpPolicy', env, verbose=0)
    try:
        model.policy.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            model.policy.load_state_dict(state['state_dict'])
        else:
            raise
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved-path', type=str, required=True)
    parser.add_argument('--classifier', type=str, default='./mnist_cnn.pth')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    env = ActiveExplorerMNISTEnv(classifier_path=args.classifier, seed=args.seed)
    policy = None
    # Try normal load first, fallback to state-dict loader
    try:
        policy = PPO.load(args.saved_path)
    except Exception as e:
        print('PPO.load failed, using state-dict fallback:', e)
        policy = load_policy_from_zip(args.saved_path, env)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(int(action))
            env.render(mode='cv2')
            time.sleep(0.02)
            steps += 1
            if steps > env.max_steps + 10:
                break
        print(f'Episode {ep}: true={info.get("true_label")} pred={info.get("classifier_pred")} conf={info.get("classifier_max_confidence"):.3f} moves={info.get("moves")}')

    print('Done')


if __name__ == '__main__':
    main()
