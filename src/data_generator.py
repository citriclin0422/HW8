"""Dataset generators for nonlinear SVM demonstrations."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons


def generate_dataset(
    dataset_type: str = "circles",
    n_samples: int = 300,
    noise: float = 0.08,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-feature classification dataset."""
    dataset_type = dataset_type.lower()

    if dataset_type in {"linear", "linearly separable"}:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=max(0.7, 1.8 - noise * 2),
            flip_y=min(noise, 0.20),
            random_state=random_state,
        )
        return X, y

    if dataset_type == "blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=2,
            cluster_std=0.85 + noise * 2.5,
            random_state=random_state,
        )
        return X, y

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

    raise ValueError("dataset_type must be one of: linear, blobs, circles, moons, xor")
