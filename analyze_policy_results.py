"""Analyze policy CSV results and produce plots.

Produces three PNGs per CSV:
- confusion matrix (true vs predicted)
- confidence histogram (with threshold line)
- scatter plot moves vs pixels_seen (with y=x line)

Usage:
    python analyze_policy_results.py ./flood_results.csv --threshold 0.90
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def analyze(csv_path: str, threshold: float = 0.90, out_dir: str | None = None):
    df = pd.read_csv(csv_path)
    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or '.'
    base = os.path.splitext(os.path.basename(csv_path))[0]

    # Confusion matrix (also show correct / total in title)
    y_true = df['true_label'].astype(int).values
    y_pred = df['predicted_label'].astype(int).values
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    correct = int((y_true == y_pred).sum())
    total = int(len(df))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {base} — correct {correct}/{total}')
    out_cm = os.path.join(out_dir, f'{base}_confusion.png')
    plt.tight_layout()
    plt.savefig(out_cm)
    plt.close()

    # Confidence histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(df['confidence'].astype(float), bins=50, kde=False)
    plt.axvline(threshold, color='red', linestyle='--', label=f'threshold={threshold}')
    plt.xlabel('Classifier Confidence')
    plt.title(f'Confidence Histogram: {base}')
    plt.legend()
    out_hist = os.path.join(out_dir, f'{base}_confidence_hist.png')
    plt.tight_layout()
    plt.savefig(out_hist)
    plt.close()

    # Moves vs pixels scatter
    plt.figure(figsize=(6, 6))
    x = df['moves'].astype(float)
    y = df['pixels_seen'].astype(float)
    plt.scatter(x, y, alpha=0.6, s=20)
    lim = max(max(x.max(), y.max()), 1)
    plt.plot([0, lim], [0, lim], color='red', linestyle='--', label='y = x')
    plt.xlabel('Moves')
    plt.ylabel('Pixels Seen')
    plt.title(f'Moves vs Pixels Seen: {base}')
    plt.legend()
    out_scatter = os.path.join(out_dir, f'{base}_moves_vs_pixels.png')
    plt.tight_layout()
    plt.savefig(out_scatter)
    plt.close()

    return out_cm, out_hist, out_scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='CSV results file')
    parser.add_argument('--threshold', type=float, default=0.90)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    outs = analyze(args.csv, threshold=args.threshold, out_dir=args.out_dir)
    print('Saved:', '\n'.join(outs))


if __name__ == '__main__':
    main()
