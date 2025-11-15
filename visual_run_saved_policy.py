"""Run and visualize a saved PPO policy for a few episodes and print results.

This script does NOT write any files; it prints episode summaries to stdout
and shows the env render window (cv2). Use Ctrl-C to stop early.

Example:
    python visual_run_saved_policy.py --saved ./ppo_explorer_flat.zip --classifier ./mnist_cnn.pth --episodes 5
"""

from __future__ import annotations

import argparse
import time

import torch
from stable_baselines3 import PPO

from active_mnist_env import ActiveExplorerMNISTEnv


def run(saved_path: str, classifier: str, episodes: int = 5, sleep: float = 0.02, seed: int | None = None):
    env = ActiveExplorerMNISTEnv(classifier_path=classifier, confidence_threshold=0.90, seed=seed)

    model = PPO.load(saved_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        moves = 0
        print(f"=== Episode {ep} ===")
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(int(action))
            moves += 1
            # render (cv2)
            try:
                env.render(mode='cv2')
            except Exception as e:
                # if cv2 not available, fall back to human/text
                env.render(mode='human')
            time.sleep(sleep)

            # safety
            if moves > env.max_steps + 10:
                break

        pixels_seen = int(info.get('pixels_seen', int(env._mask.sum()) if getattr(env, '_mask', None) is not None else 0))
        moves_taken = int(info.get('moves', moves))
        print(f"true={info.get('true_label', -1)} pred={info.get('classifier_pred', -1)} conf={info.get('classifier_max_confidence', 0.0):.3f} moves={moves_taken} pixels={pixels_seen}")
        # small pause between episodes so windows update
        time.sleep(0.5)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved', type=str, required=True, help='Path to saved PPO archive (.zip)')
    parser.add_argument('--classifier', type=str, default='./mnist_cnn.pth')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--sleep', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    run(args.saved, args.classifier, episodes=args.episodes, sleep=args.sleep, seed=args.seed)


if __name__ == '__main__':
    main()
