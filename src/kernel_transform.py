"""Feature mappings used to explain the kernel trick."""

from __future__ import annotations

import numpy as np


def explicit_3d_mapping(X: np.ndarray) -> np.ndarray:
    """Map [x1, x2] to [x1, x2, x1^2 + x2^2]."""
    z = np.sum(np.square(X), axis=1)
    return np.column_stack([X[:, 0], X[:, 1], z])


def rbf_similarity_note(gamma: float | str = "scale") -> str:
    """Return the RBF kernel formula for educational text."""
    return f"K(x, z) = exp(-{gamma} * ||x - z||^2)"

