"""Dataset generators for nonlinear SVM demonstrations."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles, make_moons


def generate_dataset(
    dataset_type: str = "circles",
    n_samples: int = 300,
    noise: float = 0.08,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-feature classification dataset."""
    dataset_type = dataset_type.lower()

    if dataset_type == "circles":
        return make_circles(
            n_samples=n_samples,
            factor=0.35,
            noise=noise,
            random_state=random_state,
        )

    if dataset_type == "moons":
        return make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
        )

    if dataset_type == "xor":
        rng = np.random.default_rng(random_state)
        X = rng.uniform(-1.2, 1.2, size=(n_samples, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        if noise > 0:
            X = X + rng.normal(0, noise, size=X.shape)
            flip_mask = rng.random(n_samples) < min(noise, 0.30)
            y[flip_mask] = 1 - y[flip_mask]
        return X, y

    raise ValueError("dataset_type must be one of: circles, moons, xor")

