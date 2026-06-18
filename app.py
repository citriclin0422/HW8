"""Streamlit app for an interactive SVM Kernel Trick 3D demo."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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


def create_webgl_svm_html(n_samples: int, noise: float, random_state: int) -> str:
    """Create a self-contained Three.js SVM kernel trick visualizer."""
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5eefb;
    }}
    #app {{
      display: grid;
      grid-template-columns: 260px 1fr 280px;
      height: 720px;
      gap: 12px;
      padding: 12px;
      box-sizing: border-box;
    }}
    .panel {{
      background: rgba(15, 23, 42, 0.84);
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 8px;
      padding: 14px;
      box-sizing: border-box;
    }}
    h2 {{ font-size: 18px; margin: 0 0 10px; color: #93c5fd; }}
    h3 {{ font-size: 14px; margin: 14px 0 8px; color: #a7f3d0; }}
    p {{ font-size: 13px; line-height: 1.45; color: #cbd5e1; }}
    .btn, .toggle {{
      width: 100%;
      display: block;
      border: 1px solid rgba(147, 197, 253, 0.35);
      background: rgba(30, 41, 59, 0.92);
      color: #e0f2fe;
      border-radius: 8px;
      padding: 10px;
      margin: 8px 0;
      text-align: left;
      cursor: pointer;
      font-size: 13px;
    }}
    .btn.active {{ background: #1d4ed8; border-color: #93c5fd; }}
    .toggle input {{ margin-right: 8px; }}
    #viewport {{
      position: relative;
      background: radial-gradient(circle at top, #1e293b, #020617);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 8px;
      overflow: hidden;
    }}
    #canvas3d {{ width: 100%; height: 100%; }}
    .overlay {{
      position: absolute;
      left: 14px;
      bottom: 14px;
      padding: 8px 10px;
      border-radius: 8px;
      background: rgba(2, 6, 23, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.28);
      font-size: 12px;
      color: #dbeafe;
    }}
    .legend-dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
    }}
    .blue {{ background: #3b82f6; }}
    .orange {{ background: #f97316; }}
    .yellow {{ background: #facc15; }}
    code {{ color: #fde68a; }}
  </style>
</head>
<body>
  <div id="app">
    <section class="panel">
      <h2>WebGL Controls</h2>
      <h3>Kernel Trick Steps</h3>
      <button class="btn active" id="step2d">Step 1: original 2D rings</button>
      <button class="btn" id="step3d">Step 2: lift to z = x² + y²</button>
      <button class="btn" id="stepProject">Step 3: projected 2D boundary</button>
      <h3>Layers</h3>
      <label class="toggle"><input type="checkbox" id="planeToggle" checked> Show separating plane</label>
      <label class="toggle"><input type="checkbox" id="svToggle" checked> Highlight support vectors</label>
      <label class="toggle"><input type="checkbox" id="marginToggle" checked> Show margin rings</label>
      <label class="toggle"><input type="checkbox" id="surfaceToggle" checked> Show mapping surface</label>
    </section>
    <section id="viewport">
      <div id="canvas3d"></div>
      <div class="overlay">Drag to rotate · Scroll to zoom · Double click to reset</div>
    </section>
    <section class="panel">
      <h2>SVM Explanation</h2>
      <div id="info">
        <p><span class="legend-dot blue"></span>Inner class</p>
        <p><span class="legend-dot orange"></span>Outer class</p>
        <p><span class="legend-dot yellow"></span>Support vectors / margin</p>
        <p>The explicit surface uses <code>z = x² + y²</code> as a teaching mapping. A real RBF kernel works implicitly through similarities.</p>
      </div>
    </section>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/geometries/ParametricGeometry.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    const config = {{ samples: {int(n_samples)}, noise: {float(noise):.4f}, seed: {int(random_state)} }};
    const container = document.getElementById('canvas3d');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(4.2, 4.2, 3.3);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0.8);

    scene.add(new THREE.AmbientLight(0xffffff, 0.72));
    const light = new THREE.DirectionalLight(0xffffff, 0.75);
    light.position.set(3, 4, 6);
    scene.add(light);
    scene.add(new THREE.GridHelper(5.2, 12, 0x334155, 0x1e293b));
    scene.add(new THREE.AxesHelper(2.8));

    function rng(seed) {{
      return function() {{
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      }};
    }}
    const random = rng(config.seed);
    function gaussian() {{
      const u = Math.max(random(), 1e-9);
      const v = Math.max(random(), 1e-9);
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }}

    const pointGroup = new THREE.Group();
    const supportGroup = new THREE.Group();
    const marginGroup = new THREE.Group();
    scene.add(pointGroup, supportGroup, marginGroup);

    const points = [];
    const innerCount = Math.floor(config.samples * 0.42);
    const outerCount = config.samples - innerCount;
    for (let i = 0; i < innerCount; i++) {{
      const a = (i / innerCount) * Math.PI * 2;
      const r = 0.55 + gaussian() * config.noise;
      points.push({{ x: Math.cos(a) * r, y: Math.sin(a) * r, label: 0 }});
    }}
    for (let i = 0; i < outerCount; i++) {{
      const a = (i / outerCount) * Math.PI * 2;
      const r = 1.55 + gaussian() * config.noise;
      points.push({{ x: Math.cos(a) * r, y: Math.sin(a) * r, label: 1 }});
    }}

    const blueMat = new THREE.MeshStandardMaterial({{ color: 0x3b82f6, roughness: 0.45 }});
    const orangeMat = new THREE.MeshStandardMaterial({{ color: 0xf97316, roughness: 0.45 }});
    const svMat = new THREE.MeshStandardMaterial({{ color: 0xfacc15, emissive: 0x4a3300, roughness: 0.25 }});
    const sphere = new THREE.SphereGeometry(0.045, 16, 16);

    const meshes = [];
    points.forEach((p) => {{
      const mesh = new THREE.Mesh(sphere, p.label === 0 ? blueMat : orangeMat);
      mesh.userData = p;
      pointGroup.add(mesh);
      meshes.push(mesh);
      const z = p.x * p.x + p.y * p.y;
      if (Math.abs(Math.sqrt(z) - 1.05) < 0.10) {{
        const sv = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 16), svMat);
        sv.userData = p;
        supportGroup.add(sv);
      }}
    }});

    const surfaceGeo = new THREE.ParametricGeometry((u, v, target) => {{
      const x = (u - 0.5) * 3.6;
      const y = (v - 0.5) * 3.6;
      target.set(x, y, x * x + y * y);
    }}, 42, 42);
    const surfaceMat = new THREE.MeshStandardMaterial({{ color: 0x60a5fa, transparent: true, opacity: 0.24, wireframe: true }});
    const surface = new THREE.Mesh(surfaceGeo, surfaceMat);
    scene.add(surface);

    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(4.2, 4.2, 1, 1),
      new THREE.MeshStandardMaterial({{ color: 0x22c55e, transparent: true, opacity: 0.28, side: THREE.DoubleSide }})
    );
    plane.rotation.x = Math.PI / 2;
    plane.position.z = 1.15;
    scene.add(plane);

    function makeRing(radius, z, color) {{
      const curve = new THREE.EllipseCurve(0, 0, radius, radius, 0, Math.PI * 2);
      const pts = curve.getPoints(180).map(p => new THREE.Vector3(p.x, p.y, z));
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      return new THREE.LineLoop(geo, new THREE.LineBasicMaterial({{ color }}));
    }}
    const decisionRing = makeRing(1.05, 0.025, 0xfacc15);
    const marginInner = makeRing(0.92, 0.03, 0xffffff);
    const marginOuter = makeRing(1.18, 0.03, 0xffffff);
    marginGroup.add(decisionRing, marginInner, marginOuter);

    let mode = '2d';
    function setMode(next) {{
      mode = next;
      document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
      if (next === '2d') document.getElementById('step2d').classList.add('active');
      if (next === '3d') document.getElementById('step3d').classList.add('active');
      if (next === 'project') document.getElementById('stepProject').classList.add('active');
      updateInfo();
    }}

    function updateInfo() {{
      const info = document.getElementById('info');
      if (mode === '2d') {{
        info.innerHTML = '<h3>Step 1: Original 2D</h3><p>The inner and outer rings are not linearly separable by one straight line.</p><p>Try rotating the view: all points are still on the flat z=0 plane.</p>';
      }} else if (mode === '3d') {{
        info.innerHTML = '<h3>Step 2: Feature Mapping</h3><p>Each point is lifted by <code>z = x² + y²</code>. The outer class moves higher, making a horizontal separating plane possible.</p>';
      }} else {{
        info.innerHTML = '<h3>Step 3: Project Back</h3><p>The 3D separating plane becomes a circular nonlinear boundary in the original 2D view.</p><p>This mirrors the kernel trick intuition.</p>';
      }}
    }}

    document.getElementById('step2d').onclick = () => setMode('2d');
    document.getElementById('step3d').onclick = () => setMode('3d');
    document.getElementById('stepProject').onclick = () => setMode('project');
    document.getElementById('planeToggle').onchange = e => plane.visible = e.target.checked;
    document.getElementById('svToggle').onchange = e => supportGroup.visible = e.target.checked;
    document.getElementById('marginToggle').onchange = e => marginGroup.visible = e.target.checked;
    document.getElementById('surfaceToggle').onchange = e => surface.visible = e.target.checked;
    renderer.domElement.ondblclick = () => {{
      camera.position.set(4.2, 4.2, 3.3);
      controls.target.set(0, 0, 0.8);
    }};

    function animate() {{
      requestAnimationFrame(animate);
      meshes.forEach((mesh) => {{
        const p = mesh.userData;
        const targetZ = mode === '3d' ? p.x * p.x + p.y * p.y : 0;
        mesh.position.z += (targetZ - mesh.position.z) * 0.075;
        mesh.position.x += (p.x - mesh.position.x) * 0.15;
        mesh.position.y += (p.y - mesh.position.y) * 0.15;
      }});
      supportGroup.children.forEach((mesh) => {{
        const p = mesh.userData;
        const targetZ = mode === '3d' ? p.x * p.x + p.y * p.y : 0.03;
        mesh.position.z += (targetZ - mesh.position.z) * 0.075;
        mesh.position.x += (p.x - mesh.position.x) * 0.15;
        mesh.position.y += (p.y - mesh.position.y) * 0.15;
      }});
      surface.visible = document.getElementById('surfaceToggle').checked && mode === '3d';
      plane.visible = document.getElementById('planeToggle').checked && mode === '3d';
      marginGroup.visible = document.getElementById('marginToggle').checked && mode !== '3d';
      controls.update();
      renderer.render(scene, camera);
    }}
    setMode('2d');
    animate();

    window.addEventListener('resize', () => {{
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }});
  </script>
</body>
</html>
"""


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

tab_concept, tab_animation, tab_webgl, tab_2d, tab_3d, tab_metrics, tab_notes = st.tabs(
    ["Concept", "Manim Animation", "WebGL 3D", "2D Boundary", "3D Kernel View", "Model Metrics", "Learning Notes"]
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

with tab_webgl:
    st.subheader("WebGL 3D kernel trick studio")
    st.write(
        "This Three.js view follows the classmate-style WebGL workflow: original 2D rings, "
        "animated lift to `z = x^2 + y^2`, separating plane, support vectors, and projected nonlinear boundary."
    )
    components.html(
        create_webgl_svm_html(n_samples=n_samples, noise=noise, random_state=int(random_state)),
        height=760,
        scrolling=False,
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
