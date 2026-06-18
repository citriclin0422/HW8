"""Streamlit app for an interactive SVM Kernel Trick 3D demo."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data_generator import generate_dataset
from src.kernel_transform import rbf_similarity_note
from src.plotly_visualizer import (
    plot_2d_decision_boundary,
    plot_3d_decision_surface,
    plot_3d_kernel_mapping,
    plot_confusion_matrix,
    plot_model_comparison,
)
from src.svm_model import evaluate_model, get_support_vectors, train_svm_models


PROJECT_ROOT = Path(__file__).parent
CONCEPT_IMAGE = PROJECT_ROOT / "SVM_3D教學設計概念圖表.png"
FALLBACK_GIF = PROJECT_ROOT / "Kernel_method.gif"


def find_manim_video() -> Path | None:
    """Find a rendered Manim video if one exists."""
    candidates = [
        PROJECT_ROOT / "outputs" / "videos",
        PROJECT_ROOT / "outputs" / "manim",
        PROJECT_ROOT / "media" / "videos",
    ]
    for folder in candidates:
        if folder.exists():
            videos = sorted(
                (path for path in folder.rglob("*.mp4") if "partial_movie_files" not in path.parts),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if videos:
                return videos[0]
    return None


st.set_page_config(
    page_title="SVM Kernel Trick 3D Demo",
    page_icon="SVM",
    layout="wide",
)

st.title("SVM Kernel Trick 3D Interactive Demo")
st.caption("Explore how SVMs use margins, support vectors, and kernels to separate nonlinear data.")

with st.sidebar:
    st.header("Controls")
    dataset_type = st.selectbox("Dataset", ["circles", "moons", "xor"])
    n_samples = st.slider("Samples", 100, 1000, 300, step=50)
    noise = st.slider("Noise", 0.0, 0.30, 0.08, step=0.01)
    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
    C = st.slider("C", 0.01, 100.0, 1.0, step=0.01, format="%.2f")
    gamma_mode = st.selectbox("Gamma", ["scale", "auto", "custom"], disabled=kernel == "linear")
    custom_gamma = st.slider("Custom gamma", 0.001, 10.0, 1.0, step=0.001, format="%.3f", disabled=gamma_mode != "custom" or kernel == "linear")
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

gamma = custom_gamma if gamma_mode == "custom" else gamma_mode
if kernel == "linear":
    gamma = "scale"

X, y = generate_dataset(dataset_type, n_samples, noise, int(random_state))
bundle = train_svm_models(X, y, kernel=kernel, C=C, gamma=gamma)
linear_metrics = evaluate_model(bundle.linear_model, bundle.X_test, bundle.y_test)
selected_metrics = evaluate_model(bundle.selected_model, bundle.X_test, bundle.y_test)
support_vectors = get_support_vectors(bundle.selected_model)

metric_cols = st.columns(5)
metric_cols[0].metric("Accuracy", f"{selected_metrics['accuracy']:.3f}")
metric_cols[1].metric("Precision", f"{selected_metrics['precision']:.3f}")
metric_cols[2].metric("Recall", f"{selected_metrics['recall']:.3f}")
metric_cols[3].metric("F1", f"{selected_metrics['f1']:.3f}")
metric_cols[4].metric("Support Vectors", len(support_vectors))

tab_concept, tab_animation, tab_2d, tab_3d, tab_metrics, tab_notes = st.tabs(
    ["Concept", "Manim Animation", "2D Boundary", "3D Kernel View", "Model Metrics", "Learning Notes"]
)

with tab_concept:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Teaching flow")
        st.write(
            "This demo follows the five-step story from the design chart: start with nonlinear 2D rings, "
            "map them through a feature function, separate them in a 3D feature view, project the idea back "
            "to a nonlinear 2D boundary, then compare it with the real RBF SVM decision function."
        )
        if CONCEPT_IMAGE.exists():
            st.image(str(CONCEPT_IMAGE), caption="SVM Kernel Trick 3D teaching blueprint")
    with right:
        st.subheader("Core equations")
        st.latex(r"w^T x + b = 0")
        st.latex(r"\text{margin width} = \frac{2}{\lVert w \rVert}")
        st.latex(r"\phi(x,y) = (x,\ y,\ x^2+y^2)")
        st.latex(r"K(x,z)=\phi(x)^T\phi(z)")
        st.subheader("Current setup")
        st.write(f"Dataset: `{dataset_type}`")
        st.write(f"Kernel: `{kernel}`")
        st.write(f"C: `{C:.2f}`")
        st.write(f"Gamma: `{gamma}`")
        st.write(rbf_similarity_note(gamma))

with tab_animation:
    st.subheader("Manim concept animation")
    manim_video = find_manim_video()
    if manim_video:
        st.video(str(manim_video))
        st.caption(f"Loaded rendered Manim video: `{manim_video.relative_to(PROJECT_ROOT)}`")
    elif FALLBACK_GIF.exists():
        st.image(str(FALLBACK_GIF), caption="Fallback animation preview. Render the Manim scene to replace this with MP4.")
    else:
        st.info("No rendered Manim video or fallback GIF was found yet.")

    st.write(
        "Render the MP4 once, then reload this app. The player will automatically pick up the newest `.mp4` "
        "from `outputs/videos`, `outputs/manim`, or Manim's default `media/videos` folder."
    )
    st.code(
        "manim -pql manim_scenes/svm_kernel_intro.py SVMKernelIntro --media_dir outputs/manim",
        language="powershell",
    )

with tab_2d:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_2d_decision_boundary(bundle.linear_model, X, y, "Linear SVM Reference"),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            plot_2d_decision_boundary(bundle.selected_model, X, y, f"{kernel.upper()} SVM Boundary"),
            width="stretch",
        )

with tab_3d:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_3d_kernel_mapping(X, y), width="stretch")
    with c2:
        st.plotly_chart(plot_3d_decision_surface(bundle.selected_model, X, y), width="stretch")

with tab_metrics:
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.plotly_chart(plot_model_comparison(linear_metrics, selected_metrics), width="stretch")
    with c2:
        st.plotly_chart(plot_confusion_matrix(selected_metrics["confusion_matrix"]), width="stretch")
    st.dataframe(
        {
            "metric": ["accuracy", "precision", "recall", "f1"],
            "linear_svm": [linear_metrics[key] for key in ["accuracy", "precision", "recall", "f1"]],
            "selected_svm": [selected_metrics[key] for key in ["accuracy", "precision", "recall", "f1"]],
        },
        hide_index=True,
        width="stretch",
    )

with tab_notes:
    st.subheader("Parameter intuition")
    st.write(
        "**C** controls how strongly the model penalizes classification errors. "
        "Small C usually gives a wider, smoother margin. Large C tries harder to fit the training data."
    )
    st.write(
        "**Gamma** controls how local each training point's influence is for the RBF kernel. "
        "Small gamma creates broad, smooth regions. Large gamma can make the boundary very detailed."
    )
    st.write(
        "**Support vectors** are the points that most directly shape the boundary. "
        "They sit closest to the margin or inside it, so moving them would change the classifier."
    )
    st.write(
        "The 3D mapping tab shows a simple explicit mapping, `z = x1^2 + x2^2`, for intuition. "
        "RBF SVMs use a richer implicit mapping, but the core idea is the same: nonlinear in 2D can become easier to separate in a transformed view."
    )
