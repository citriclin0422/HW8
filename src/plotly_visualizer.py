"""Plotly figures for SVM 2D and 3D demonstrations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.kernel_transform import explicit_3d_mapping
from src.svm_model import get_support_vectors


COLOR_0 = "#2563eb"
COLOR_1 = "#f97316"
SUPPORT_COLOR = "#111827"


def create_mesh_grid(X: np.ndarray, resolution: int = 120, padding: float = 0.35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2D grid around the dataset."""
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def _class_colors(y: np.ndarray) -> list[str]:
    return [COLOR_1 if label == 1 else COLOR_0 for label in y]


def plot_2d_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "2D Decision Boundary") -> go.Figure:
    """Plot data, support vectors, and the decision boundary contour."""
    xx, yy, grid = create_mesh_grid(X)
    scores = model.decision_function(grid).reshape(xx.shape)
    support_vectors = get_support_vectors(model)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=scores,
            contours=dict(start=0, end=0, size=1, coloring="lines"),
            line=dict(color="#0f172a", width=3),
            showscale=False,
            name="decision boundary",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=scores,
            contours=dict(start=-1, end=1, size=2, coloring="lines"),
            line=dict(color="#64748b", width=1, dash="dash"),
            showscale=False,
            name="margin",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(color=_class_colors(y), size=8, line=dict(color="white", width=1)),
            name="samples",
            text=[f"class {label}" for label in y],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode="markers",
            marker=dict(symbol="circle-open", color=SUPPORT_COLOR, size=13, line=dict(width=2)),
            name="support vectors",
        )
    )
    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="x1",
        yaxis_title="x2",
    )
    return fig


def plot_3d_kernel_mapping(X: np.ndarray, y: np.ndarray, title: str = "Explicit 3D Mapping") -> go.Figure:
    """Plot z = x1^2 + x2^2 mapping."""
    X3 = explicit_3d_mapping(X)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X3[:, 0],
                y=X3[:, 1],
                z=X3[:, 2],
                mode="markers",
                marker=dict(color=_class_colors(y), size=4, opacity=0.86),
                name="mapped samples",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=620,
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="x1^2 + x2^2"),
    )
    return fig


def plot_3d_decision_surface(model, X: np.ndarray, y: np.ndarray, title: str = "3D Decision Surface") -> go.Figure:
    """Plot the SVM decision function as a surface over the original feature plane."""
    xx, yy, grid = create_mesh_grid(X, resolution=80)
    scores = model.decision_function(grid).reshape(xx.shape)
    support_vectors = get_support_vectors(model)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=scores,
            colorscale="RdBu",
            opacity=0.72,
            showscale=True,
            colorbar=dict(title="f(x)", len=0.55),
            name="decision function",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=model.decision_function(X),
            mode="markers",
            marker=dict(color=_class_colors(y), size=4, line=dict(color="white", width=0.5)),
            name="samples",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            z=model.decision_function(support_vectors),
            mode="markers",
            marker=dict(symbol="circle-open", color=SUPPORT_COLOR, size=7, line=dict(width=3)),
            name="support vectors",
        )
    )
    fig.update_layout(
        title=title,
        height=650,
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="decision f(x)"),
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Plot a compact confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            colorscale=[[0, "#eff6ff"], [1, "#1d4ed8"]],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def plot_model_comparison(linear_metrics: dict[str, object], selected_metrics: dict[str, object]) -> go.Figure:
    """Compare metric bars for linear SVM and the selected kernel."""
    labels = ["accuracy", "precision", "recall", "f1"]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(name="Linear SVM", x=labels, y=[linear_metrics[label] for label in labels], marker_color="#64748b"))
    fig.add_trace(go.Bar(name="Selected SVM", x=labels, y=[selected_metrics[label] for label in labels], marker_color="#2563eb"))
    fig.update_layout(
        barmode="group",
        height=340,
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
