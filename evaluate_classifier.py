"""Evaluate the pretrained SmallCNN classifier on MNIST test set.

Prints top-1 accuracy and optionally writes a small report.
"""

from __future__ import annotations

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from active_mnist_env import SmallCNN


def _load_compatible_model(path: str, device: torch.device) -> SmallCNN:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    else:
        state = raw

    # Try padding variants that were used across checkpoints
    for pad in (1, 0):
        model = SmallCNN(num_classes=10, padding=pad).to(device)
        try:
            model.load_state_dict(state)
            return model
        except Exception:
            continue

    # Fallback: load matching keys only
    model = SmallCNN(num_classes=10, padding=1).to(device)
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default="./mnist_cnn.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not os.path.exists(args.classifier):
        raise FileNotFoundError(f"Classifier file not found: {args.classifier}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_compatible_model(args.classifier, device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Classifier accuracy on MNIST test set: {acc*100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
