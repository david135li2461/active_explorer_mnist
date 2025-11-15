Active Explorer MNIST
======================

This folder contains a Gymnasium environment for active exploration of MNIST images.

Files
- `active_mnist_env.py` - the Gymnasium environment implementation. It implements `ActiveExplorerMNISTEnv`.

Concept
- An agent moves on the 28x28 MNIST image revealing pixel values.
- Allowed actions: up, down, left, right.
- After each move the environment sends the current partial image to a pretrained full-image MNIST classifier.
- If the classifier's confidence (max softmax probability) >= threshold (default 0.90), the episode ends and the agent receives reward=1 (regardless of classifier correctness).
- Otherwise the agent continues until a step limit.

Why 0.90 threshold?
- For a 10-class classification problem the chance baseline is 0.1. A 0.90 threshold provides a reasonable trade-off between early stopping and reliability; it requires the classifier to be much more confident than random guessing. The threshold is configurable at environment construction and should be tuned for your classifier.

How to provide the classifier
- The environment expects a PyTorch state_dict compatible with the `SmallCNN` class in `active_mnist_env.py`.
- If you already have a `mnist_cnn.pth` in your project, pass its path via the `classifier_path` argument. If not provided, the env will attempt a small heuristic search for `mnist_cnn.pth` in nearby repository locations.
- I do NOT auto-download models. To experiment quickly you can reuse a pretrained mnist model from elsewhere and save the `state_dict`.

Example usage

Simple smoke test (requires classifier):

```bash
python active_mnist_env.py --classifier /path/to/mnist_cnn.pth --threshold 0.90
```

Using with stable-baselines3 PPO (example sketch)
- Install dependencies in `requirements.txt`.
- Example training script (not included): create a vectorized env, wrap with `sb3.common.monitor`, and call `PPO(...).learn(...)`.

Notes & next steps
- You can extend the observation to expose the mask explicitly (currently unrevealed pixels are 0.0).
- For SB3 integration, wrap the env with a `FlattenObservation` wrapper if you wish to use the current flat-vector observation.
- Consider adding a "stop" action if you want the agent to explicitly declare the moment it is confident rather than relying on the classifier to trigger stop.
