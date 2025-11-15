"""Test a policy (flood or saved SB3 policy) on the ActiveExplorerMNISTEnv and log episodes to CSV.

CSV columns: episode, true_label, predicted_label, confidence, moves, pixels_seen

Usage examples:
    python test_policy_runner.py --policy flood --num-episodes 100 --output results.csv
    python test_policy_runner.py --policy saved --saved-path ./ppo_explorer/ppo_explorer.zip --num-episodes 50 --render

By default rendering is off. Turn on with --render to see live cv2 windows (requires opencv-python).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import time
import zipfile
from typing import Optional

import numpy as np
import torch

from active_mnist_env import ActiveExplorerMNISTEnv

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


class FloodPolicy:
    """Simple local exploration policy.

    Rules:
    - Never move off the image.
    - If an unexplored neighbor exists, pick one at random.
    - If all neighbors are explored, pick a random valid neighbor.
    """

    def __init__(self, env: ActiveExplorerMNISTEnv):
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        # obs is a flat vector: [img_flat, mask_flat, r, c]
        img_len = self.env._img_len
        mask_flat = obs[img_len: img_len * 2]
        # flat obs stores raw integer coords as last two entries
        r = int(round(float(obs[-2])))
        c = int(round(float(obs[-1])))

        neighbors = []  # (action, nr, nc)
        # up
        if r - 1 >= 0:
            neighbors.append((0, r - 1, c))
        # down
        if r + 1 < self.env.img_h:
            neighbors.append((1, r + 1, c))
        # left
        if c - 1 >= 0:
            neighbors.append((2, r, c - 1))
        # right
        if c + 1 < self.env.img_w:
            neighbors.append((3, r, c + 1))

        unexplored = [a for (a, nr, nc) in neighbors if mask_flat[nr * self.env.img_w + nc] < 0.5]
        if len(unexplored) > 0:
            return int(random.choice(unexplored))
        # else choose any neighbor
        return int(random.choice([a for (a, nr, nc) in neighbors]))


def load_policy_fallback(saved_path: str, env: ActiveExplorerMNISTEnv):
    """Fallback loader: extract `policy.pth` from the saved SB3 zip and
    load it into a freshly-instantiated PPO('MlpPolicy', env).

    This avoids using cloudpickle to deserialize objects that reference
    environment-specific module names (e.g., `numpy._core`).
    """
    if PPO is None:
        raise RuntimeError('stable-baselines3 not available in environment')
    if not zipfile.is_zipfile(saved_path):
        raise RuntimeError('Saved policy is not a zip archive')
    with zipfile.ZipFile(saved_path, 'r') as z:
        if 'policy.pth' not in z.namelist():
            raise RuntimeError('policy.pth not found in archive; cannot fallback')
        data = z.read('policy.pth')
    state = torch.load(io.BytesIO(data), map_location='cpu')
    # instantiate model with env to create the architecture
    model = PPO('MlpPolicy', env, verbose=0)
    # try to load state directly or under common key
    try:
        model.policy.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            model.policy.load_state_dict(state['state_dict'])
        else:
            raise
    return model


def run_episode(env: ActiveExplorerMNISTEnv, policy, render: bool = False):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    moves = 0
    # loop until either termination (confidence threshold met) or truncation (time limit)
    while not (terminated or truncated):
        if isinstance(policy, FloodPolicy):
            action = policy.act(obs)
        else:
            # SB3 model
            action, _ = policy.predict(obs, deterministic=True)
            # when using SB3 the observation shape expected is (obs_dim,) or (1, obs_dim)
            # stable-baselines3 may accept 1D obs
        obs, rew, terminated, truncated, info = env.step(int(action))
        moves += 1
        if render:
            env.render(mode='cv2')
            # slow down so humans can see
            time.sleep(0.02)
        # safety cap
        if moves > env.max_steps + 10:
            break

    # prefer info-provided terminal metadata; fall back to env internals
    pixels_seen = int(info.get('pixels_seen', int(env._mask.sum()) if getattr(env, '_mask', None) is not None else 0))
    moves_taken = int(info.get('moves', moves))
    return {
        'true_label': int(info.get('true_label', -1)),
        'predicted_label': int(info.get('classifier_pred', -1)),
        'confidence': float(info.get('classifier_max_confidence', 0.0)),
        'moves': moves_taken,
        'pixels_seen': pixels_seen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["flood", "saved"], required=True)
    parser.add_argument("--saved-path", type=str, default=None, help="Path to saved SB3 model (.zip)")
    parser.add_argument("--classifier", type=str, default="./mnist_cnn.pth")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="policy_results.csv")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.policy == 'saved' and (args.saved_path is None or not os.path.exists(args.saved_path)):
        raise FileNotFoundError("Saved policy path must be provided and exist when --policy saved")

    env = ActiveExplorerMNISTEnv(classifier_path=args.classifier, confidence_threshold=args.threshold, seed=args.seed)

    if args.policy == 'flood':
        policy = FloodPolicy(env)
    else:
        if PPO is None:
            raise RuntimeError("stable-baselines3 not available; install it to use a saved policy")
        # First try normal SB3 loader (may fail if cloudpickle expects different modules)
        try:
            policy = PPO.load(args.saved_path)
        except Exception as e:
            print('PPO.load failed, falling back to PyTorch state dict loader:', type(e).__name__, e)
            policy = load_policy_fallback(args.saved_path, env)

    # run episodes and log
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['episode', 'true_label', 'predicted_label', 'confidence', 'moves', 'pixels_seen'])
        writer.writeheader()
        for ep in range(args.num_episodes):
            res = run_episode(env, policy, render=args.render)
            row = {'episode': ep, **res}
            writer.writerow(row)
            print(f"Episode {ep}: true={res['true_label']} pred={res['predicted_label']} conf={res['confidence']:.3f} moves={res['moves']} pixels={res['pixels_seen']}")

    print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
