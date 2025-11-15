"""Active exploration MNIST Gymnasium environment

Environment summary
- Grid: 28x28 (MNIST image). Agent starts at a random pixel.
- Actions: 0=up,1=down,2=left,3=right
- At each step the agent "reveals" the pixel at its current position (i.e., the pixel value becomes visible in the partial observation).
- After each step the environment sends the current partial image to a pretrained full-image MNIST classifier.
- If the classifier's max softmax probability >= confidence_threshold, the episode ends and the agent receives reward=1 (regardless of correctness).
- Otherwise reward=0. Episode also ends at a configurable max number of steps with no reward.

Design choices and justification for the threshold
- We use a softmax maximum probability as the classifier "confidence". For a 10-way classifier, random chance is 0.1; we choose a threshold default of 0.90.
  Reason: 0.90 is high enough to require strong evidence for a single class while still being reachable after revealing a small, informative subset of pixels for many digits.
  Lower thresholds lead to earlier stopping but much higher false-positive rates; higher thresholds force more reveals and slow learning.
  The threshold is configurable in the constructor so you can experiment.

Notes
- This file implements the environment only. It expects you to provide a pretrained classifier path (a PyTorch state_dict compatible with the SmallCNN class below) via the `classifier_path` argument or place a suitable model file in the repository and pass its path.
- I provide helper code to build the classifier model architecture and load weights. I intentionally do NOT auto-download a model to avoid silent network actions; you can download one and pass its path.

Usage (example):
    from active_mnist_env import ActiveExplorerMNISTEnv
    env = ActiveExplorerMNISTEnv(classifier_path='/path/to/mnist_cnn.pth')

    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, rew, done, trunc, info = env.step(action)
        if done:
            break

You can wrap the env for SB3/PPO training; see README.md for example command-lines.
"""

from __future__ import annotations

import os
import typing as t
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    """Small CNN for MNIST classification.

    The architecture is intentionally flexible: some pretrained checkpoints use
    conv layers with padding=1 (keeps 28->28 before pooling) while others use
    padding=0 (reduces spatial dims before pooling). We accept a `padding`
    argument and compute the fully-connected input size accordingly so that
    we can attempt to load slightly different checkpoints.
    """

    def __init__(self, num_classes: int = 10, padding: int = 0):
        super().__init__()
        self.padding = int(padding)
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=self.padding)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=self.padding)
        self.pool = nn.MaxPool2d(2, 2)
        # compute spatial size after two convs and one pooling
        s = 28
        # after two convs with kernel=3, padding=p each: s -> s + 2*p - 2 (per conv)
        s = s + 2 * self.padding - 2
        s = s + 2 * self.padding - 2
        # after pooling /2
        s = s // 2
        fc_in = 64 * s * s
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActiveExplorerMNISTEnv(gym.Env):
    """Gymnasium environment for active MNIST exploration.

    Observation space
    - A flattened vector of length 28*28 + 2 (image flattened followed by agent row and col as integers scaled to [0,1]).
      Revealed pixels contain their normalized grayscale values in [0,1]. Unrevealed pixels are 0.0.

    Action space
    - Discrete(4): up, down, left, right

    Important arguments
    - classifier_path: path to a PyTorch state_dict for SmallCNN. If None, the environment will attempt to locate
      a file named 'mnist_cnn.pth' in nearby repo folders. If not found, initialization raises an informative error.
    - confidence_threshold: softmax probability threshold to stop and award reward=1 when met or exceeded.
    - max_steps: maximum number of steps before truncation.
    """

    metadata = {"render_modes": ["human", "cv2"]}

    def __init__(
        self,
        classifier_path: t.Optional[str] = None,
        confidence_threshold: float = 0.5,
        max_steps: int = 500,
        seed: t.Optional[int] = None,
        mnist_root: str = "./data",
    ) -> None:
        super().__init__()

        assert 0.0 < confidence_threshold < 1.0, "confidence_threshold must be in (0,1)"
        self.confidence_threshold = float(confidence_threshold)
        # justification briefly: see module docstring

        self.max_steps = int(max_steps)
        self.seed(seed)

        # action space: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # observation: flattened image (28*28) + flattened mask (28*28) + raw row + raw col
        # Use a flat vector observation (most compatible with stable-baselines3 PPO MlpPolicy)
        self.img_h = 28
        self.img_w = 28
        self._img_len = self.img_h * self.img_w
        self._obs_len = self._img_len * 2 + 2
        low = np.concatenate([
            np.zeros(self._img_len, dtype=np.float32),
            np.zeros(self._img_len, dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ])
        high = np.concatenate([
            np.ones(self._img_len, dtype=np.float32),
            np.ones(self._img_len, dtype=np.float32),
            np.array([float(self.img_h - 1), float(self.img_w - 1)], dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # (observation_space already configured above depending on observation_format)

        # load MNIST dataset to sample images on reset
        self.mnist_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.mnist_root = mnist_root
        try:
            # keep dataset object to sample from when resetting
            self._mnist = datasets.MNIST(self.mnist_root, train=True, download=True, transform=self.mnist_transform)
        except Exception as e:
            warnings.warn(f"Failed to download/load MNIST dataset at {self.mnist_root}: {e}")
            self._mnist = None

        # classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = SmallCNN(num_classes=10).to(self.device)
        loaded = False
        if classifier_path is not None:
            if os.path.exists(classifier_path):
                self._load_classifier_state(classifier_path)
                loaded = True
            else:
                raise FileNotFoundError(f"classifier_path provided but file not found: {classifier_path}")
        else:
            # try to find a weight file named mnist_cnn.pth nearby (repo heuristics)
            candidate = self._find_mnist_weights()
            if candidate is not None:
                self._load_classifier_state(candidate)
                loaded = True

        if not loaded:
            raise FileNotFoundError(
                "No pretrained classifier found. Please provide classifier_path pointing to a PyTorch state_dict "
                "compatible with SmallCNN, or place 'mnist_cnn.pth' in the repository and re-run. See README.md for details."
            )

        self.classifier.eval()

        # runtime state
        self._target_image: t.Optional[np.ndarray] = None
        self._target_label: t.Optional[int] = None
        self._mask: t.Optional[np.ndarray] = None
        self._pos: t.Optional[t.Tuple[int, int]] = None
        self._steps = 0

    def _load_classifier_state(self, path: str) -> None:
        raw = torch.load(path, map_location=self.device)
        # accept either a raw state_dict or a checkpoint dict containing 'model_state_dict'
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        else:
            state = raw

        # Load the checkpoint assuming the SmallCNN(padding=0) architecture used by
        # the included/preferred `mnist_cnn.pth`. If this fails, raise an error so
        # the user can provide a compatible checkpoint.
        try:
            alt = SmallCNN(num_classes=10, padding=0).to(self.device)
            alt.load_state_dict(state)
            self.classifier = alt
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load classifier state dict from {path}: {e}")

    def _find_mnist_weights(self) -> t.Optional[str]:
        """Heuristically search a few likely repository locations for 'mnist_cnn.pth'."""
        cwd = os.path.abspath(os.getcwd())
        candidates = [
            os.path.join(cwd, "mnist_cnn.pth"),
            os.path.join(cwd, "active_mnist_env", "mnist_cnn.pth"),
            os.path.join(cwd, "act-in-dark", "mnist_cnn.pth"),
            os.path.join(cwd, "active_mnist_env", "models", "mnist_cnn.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # try walking up a couple levels looking for the file
        root = cwd
        for _ in range(3):
            for candidate in ("mnist_cnn.pth", os.path.join("active_mnist_env", "mnist_cnn.pth")):
                p = os.path.join(root, candidate)
                if os.path.exists(p):
                    return p
            root = os.path.dirname(root)
        return None

    def seed(self, seed: t.Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _sample_target(self) -> None:
        if self._mnist is None:
            raise RuntimeError("MNIST dataset not available; cannot sample target image. See README.md to download datasets.")
        i = np.random.randint(0, len(self._mnist))
        img, label = self._mnist[i]
        # img is tensor [1,28,28]
        arr = img.squeeze(0).numpy().astype(np.float32)
        # normalize to [0,1] (ToTensor already did that)
        self._target_image = arr
        self._target_label = int(label)

    def reset(self, *, seed: t.Optional[int] = None, options: t.Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self._sample_target()
        self._mask = np.zeros((self.img_h, self.img_w), dtype=bool)
        # start at random position
        r = np.random.randint(0, self.img_h)
        c = np.random.randint(0, self.img_w)
        self._pos = (r, c)
        # reveal starting pixel
        self._mask[r, c] = True
        self._steps = 0
        obs = self._build_observation()
        return obs, {}

    def _build_observation(self) -> np.ndarray:
        assert self._target_image is not None and self._mask is not None and self._pos is not None
        revealed = np.where(self._mask, self._target_image, 0.0).astype(np.float32)
        flat_img = revealed.ravel()
        flat_mask = self._mask.astype(np.float32).ravel()
        # raw integer coordinates (as floats) for row and col
        r, c = self._pos
        obs = np.concatenate([flat_img, flat_mask, np.array([float(r), float(c)], dtype=np.float32)], axis=0)
        return obs

    def step(self, action: int):
        assert self._target_image is not None and self._mask is not None and self._pos is not None
        r, c = self._pos
        if action == 0:  # up
            r = max(0, r - 1)
        elif action == 1:  # down
            r = min(self.img_h - 1, r + 1)
        elif action == 2:  # left
            c = max(0, c - 1)
        elif action == 3:  # right
            c = min(self.img_w - 1, c + 1)
        else:
            raise ValueError(f"Invalid action {action}")
        self._pos = (r, c)
        self._mask[r, c] = True
        self._steps += 1

        obs = self._build_observation()

        # compute classifier confidence on the *partial* image
        partial = np.where(self._mask, self._target_image, 0.0).astype(np.float32)
        with torch.no_grad():
            timg = torch.tensor(partial).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,28,28]
            logits = self.classifier(timg)
            probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
            maxp = float(np.max(probs))
            pred = int(np.argmax(probs))

        # Gymnasium distinguishes termination (task success/failure) from truncation
        # (time limit). Treat classifier confidence meeting threshold as termination
        # with reward=1.0. Reaching max_steps without meeting threshold is a
        # truncation (no reward).
        info = {"classifier_max_confidence": maxp, "classifier_pred": pred, "true_label": self._target_label}

        terminated = bool(maxp >= self.confidence_threshold)
        truncated = bool((self._steps >= self.max_steps) and not terminated)

        reward = 1.0 if terminated else 0.0

        # Add terminal metadata when episode ends (either terminated or truncated)
        if terminated or truncated:
            pixels_seen = int(self._mask.sum()) if self._mask is not None else 0
            info["pixels_seen"] = pixels_seen
            info["moves"] = int(self._steps)

        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human"):
        # textual rendering and an OpenCV window mode that visualizes the evolving partial image
        if self._target_image is None or self._mask is None or self._pos is None:
            print("Env not initialized")
            return

        if mode == "human":
            disp = self._target_image.copy()
            # show unrevealed pixels as -1 for print clarity
            disp[~self._mask] = -1.0
            r, c = self._pos
            disp[r, c] = 2.0
            np.set_printoptions(precision=2, suppress=True)
            print(disp)
            return

        if mode == "cv2":
            # lazy import to avoid requiring cv2 at module import time
            try:
                import cv2
            except Exception as e:
                raise RuntimeError("cv2 not available; install opencv-python to use render(mode='cv2')") from e

            # build partial grayscale image where unrevealed pixels will be
            # rendered in blue so they are visually distinct from revealed
            # black pixels.
            partial = np.where(self._mask, self._target_image, 0.0).astype(np.float32)
            img_gray = (partial * 255.0).clip(0, 255).astype(np.uint8)
            # create BGR image: for revealed pixels use grayscale; for
            # unrevealed pixels paint them blue (B=255, G=0, R=0)
            mask_bool = self._mask.astype(bool)
            b = np.where(mask_bool, img_gray, 255).astype(np.uint8)
            g = np.where(mask_bool, img_gray, 0).astype(np.uint8)
            r = np.where(mask_bool, img_gray, 0).astype(np.uint8)
            img_bgr = np.stack([b, g, r], axis=-1)

            # scale up for visibility
            scale = 20
            img_large = cv2.resize(img_bgr, (self.img_w * scale, self.img_h * scale), interpolation=cv2.INTER_NEAREST)

            # draw agent position as a red circle
            r, c = self._pos
            center = (int((c + 0.5) * scale), int((r + 0.5) * scale))
            radius = max(2, scale // 2)
            cv2.circle(img_large, center, radius, (0, 0, 255), -1)

            # show the image window
            cv2.imshow("ActiveMNIST Partial", img_large)
            # small waitKey to refresh window; non-blocking
            cv2.waitKey(1)
            return

        # unsupported mode
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        return


if __name__ == "__main__":
    # minimal smoke test when run directly; will error if classifier not found — that's intentional to avoid silent downloads.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default=None, help="Path to pretrained mnist_cnn.pth state_dict")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    env = ActiveExplorerMNISTEnv(classifier_path=args.classifier, confidence_threshold=args.threshold)
    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 200:
        a = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(a)
        print(f"step={steps} action={a} rew={rew} max_conf={info['classifier_max_confidence']:.3f} pred={info['classifier_pred']} true={info['true_label']}")
        steps += 1
    print("Done")
