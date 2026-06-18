"""Streamlit app for an interactive SVM Kernel Trick 3D demo."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).parent
CONCEPT_IMAGE = PROJECT_ROOT / "SVM_3D教學設計概念圖表.png"
FALLBACK_GIF = PROJECT_ROOT / "Kernel_method.gif"

CHAPTERS = [
    ("Concept", "01 Concept", "核心故事"),
    ("Manim Animation", "02 Manim", "概念動畫"),
    ("WebGL 3D", "03 WebGL", "3D 操作"),
    ("2D Boundary", "04 Boundary", "決策邊界"),
    ("3D Kernel View", "05 Surface", "3D 曲面"),
    ("Model Metrics", "06 Metrics", "模型診斷"),
    ("Learning Notes", "07 Notes", "學習筆記"),
    ("Quiz", "08 Quiz", "小測驗"),
]

TASKS = {
    "Concept": [
        "先確認 2D 圓形資料為什麼不能用直線分開。",
        "觀察公式 phi(x, y) 如何把半徑資訊變成 z 軸高度。",
    ],
    "Manim Animation": [
        "播放動畫後，找出 2D、3D、投影回 2D 三個關鍵畫面。",
        "注意最後的提醒：RBF kernel 是隱式高維映射，不只是單一 3D 圖。",
    ],
    "WebGL 3D": [
        "依序點 Step 1、Step 2、Step 3，觀察資料如何被 lift。",
        "關閉或開啟 separating plane、support vectors、margin rings，比較視覺差異。",
    ],
    "2D Boundary": [
        "把 dataset 切到 circles，再比較 linear 與 rbf 的邊界差異。",
        "提高 gamma 或 C，觀察邊界是否變得更貼近資料點。",
    ],
    "3D Kernel View": [
        "左圖看顯式 3D mapping，右圖看真實 SVM decision function。",
        "切換 kernel 後，觀察 decision surface 是否更彎曲。",
    ],
    "Model Metrics": [
        "觀察 train/test accuracy gap，判斷是否開始 overfit。",
        "比較 support-vector ratio，理解有多少資料點正在支撐邊界。",
    ],
    "Learning Notes": [
        "讀完 C 與 gamma 後，回到 2D Boundary 實際操作驗證。",
        "把 notes 和 WebGL 3D 互相對照，區分教學映射與真實 RBF。",
    ],
    "Quiz": [
        "先不看 Learning Notes 直接作答，再回去檢查錯題。",
        "答錯時讀提示，確認自己能用一句話解釋 support vectors。",
    ],
}


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


def inject_ui_css() -> None:
    """Add lightweight CSS for the teaching dashboard."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #dbe4ee;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
        }
        .svm-card {
            border: 1px solid #d7e3f3;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
        }
        .svm-card h4 {
            margin: 0 0 0.45rem 0;
            color: #0f3b78;
            font-size: 1rem;
        }
        .svm-card p, .svm-card li {
            color: #334155;
            font-size: 0.94rem;
            line-height: 1.45;
        }
        .status-good { border-left: 5px solid #16a34a; }
        .status-watch { border-left: 5px solid #d97706; }
        .status-risk { border-left: 5px solid #dc2626; }
        .param-chip {
            display: inline-block;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            margin: 0.1rem 0.18rem 0.1rem 0;
            font-size: 0.82rem;
            color: #1f2937;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chapter_buttons(current_view: str) -> str:
    """Render visual chapter navigation while keeping sidebar navigation available."""
    st.markdown("#### Chapter Navigator")
    cols = st.columns(4)
    selected = current_view
    for index, (chapter, label, caption) in enumerate(CHAPTERS):
        with cols[index % 4]:
            button_type = "primary" if chapter == current_view else "secondary"
            if st.button(
                f"{label}\n{caption}",
                key=f"chapter_btn_{chapter}",
                use_container_width=True,
                type=button_type,
            ):
                st.session_state["chapter"] = chapter
                selected = chapter
                st.rerun()
    return selected


def render_observation_tasks(view: str) -> None:
    """Show a compact teaching task card for the active chapter."""
    tasks = TASKS.get(view, [])
    if not tasks:
        return
    items = "".join(f"<li>{task}</li>" for task in tasks)
    st.markdown(
        f"""
        <div class="svm-card">
          <h4>觀察任務</h4>
          <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_parameter_insight(dataset_type: str, kernel: str, C: float, gamma: str | float, degree: int, noise: float) -> list[str]:
    """Return short teaching hints for the current parameter combination."""
    notes: list[str] = []
    if dataset_type in {"circles", "moons", "xor"} and kernel == "linear":
        notes.append("目前是非線性資料搭配 linear kernel，預期邊界會不足以完整分開資料。")
    if kernel == "rbf":
        notes.append("RBF kernel 適合彎曲邊界；請觀察 gamma 對邊界局部性的影響。")
    if kernel == "poly":
        notes.append(f"Polynomial kernel 目前 degree = {degree}，degree 越高通常邊界越複雜。")
    if C >= 20:
        notes.append("C 偏高，模型會更努力符合訓練資料，需注意 overfitting。")
    elif C <= 0.1:
        notes.append("C 偏低，模型容忍更多錯誤，邊界通常較平滑但可能 underfit。")
    if isinstance(gamma, (int, float)) and gamma >= 3:
        notes.append("gamma 偏高，每個點的影響範圍較小，RBF 邊界可能變得細碎。")
    if noise >= 0.2:
        notes.append("noise 偏高，請比較 train/test gap，避免只看訓練表現。")
    if not notes:
        notes.append("目前參數屬於穩健起點，適合先觀察資料形狀與支援向量位置。")
    return notes


def render_parameter_panel(dataset_type: str, kernel: str, C: float, gamma: str | float, degree: int, noise: float) -> None:
    """Render an explanation panel for the active settings."""
    chips = [
        f"Dataset: {dataset_type}",
        f"Kernel: {kernel}",
        f"C: {C:.2f}",
        f"Gamma: {gamma}",
        f"Noise: {noise:.2f}",
    ]
    if kernel == "poly":
        chips.append(f"Degree: {degree}")
    chip_html = "".join(f'<span class="param-chip">{chip}</span>' for chip in chips)
    notes = "".join(f"<li>{note}</li>" for note in get_parameter_insight(dataset_type, kernel, C, gamma, degree, noise))
    st.markdown(
        f"""
        <div class="svm-card">
          <h4>目前參數解讀</h4>
          <p>{chip_html}</p>
          <ul>{notes}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status(train_metrics: dict[str, object], selected_metrics: dict[str, object], support_vector_count: int, sample_count: int) -> None:
    """Show a visual model health card based on train/test gap and support-vector ratio."""
    gap = train_metrics["accuracy"] - selected_metrics["accuracy"]
    sv_ratio = support_vector_count / sample_count
    if gap > 0.15:
        status_class = "status-risk"
        title = "模型狀態：高過擬合風險"
        message = "訓練分數明顯高於測試分數。可以降低 C、降低 gamma，或增加資料雜訊觀察邊界是否更穩定。"
    elif gap > 0.06:
        status_class = "status-watch"
        title = "模型狀態：需要觀察"
        message = "train/test gap 有一點拉開。請搭配 2D 邊界與 support-vector ratio 判斷是否過度貼合。"
    else:
        status_class = "status-good"
        title = "模型狀態：泛化表現穩定"
        message = "目前 train/test gap 很小，模型表現相對穩定。可以嘗試提高 gamma 或 C 觀察何時開始 overfit。"
    st.markdown(
        f"""
        <div class="svm-card {status_class}">
          <h4>{title}</h4>
          <p>Train/Test gap: <b>{gap:+.3f}</b> ｜ Support-vector ratio: <b>{sv_ratio:.1%}</b></p>
          <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_demo_state(
    dataset_type: str,
    n_samples: int,
    noise: float,
    kernel: str,
    C: float,
    gamma: str | float,
    degree: int,
    random_state: int,
):
    """Generate data and train models only when a model view needs them."""
    from src.data_generator import generate_dataset
    from src.svm_model import evaluate_model, get_support_vectors, train_svm_models

    X, y = generate_dataset(dataset_type, n_samples, noise, random_state)
    bundle = train_svm_models(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=random_state)
    linear_metrics = evaluate_model(bundle.linear_model, bundle.X_test, bundle.y_test)
    selected_metrics = evaluate_model(bundle.selected_model, bundle.X_test, bundle.y_test)
    train_metrics = evaluate_model(bundle.selected_model, bundle.X_train, bundle.y_train)
    support_vectors = get_support_vectors(bundle.selected_model)
    return X, y, bundle, linear_metrics, selected_metrics, train_metrics, support_vectors


def render_metric_strip(
    selected_metrics: dict[str, object],
    train_metrics: dict[str, object],
    support_vector_count: int,
    sample_count: int,
) -> None:
    """Render model metrics after the selected model has been trained."""
    metric_cols = st.columns(5)
    generalization_gap = train_metrics["accuracy"] - selected_metrics["accuracy"]
    metric_cols[0].metric("Test Accuracy", f"{selected_metrics['accuracy']:.3f}")
    metric_cols[1].metric("Train Accuracy", f"{train_metrics['accuracy']:.3f}")
    metric_cols[2].metric("Gap", f"{generalization_gap:+.3f}")
    metric_cols[3].metric("F1", f"{selected_metrics['f1']:.3f}")
    metric_cols[4].metric("SV Ratio", f"{support_vector_count / sample_count:.1%}")


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
      <button class="btn" id="step3d">Step 2: lift to z = x^2 + y^2</button>
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
        <p>The explicit surface uses <code>z = x^2 + y^2</code> as a teaching mapping. A real RBF kernel works implicitly through similarities.</p>
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
        info.innerHTML = '<h3>Step 2: Feature Mapping</h3><p>Each point is lifted by <code>z = x^2 + y^2</code>. The outer class moves higher, making a horizontal separating plane possible.</p>';
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

inject_ui_css()

st.title("SVM Kernel Trick 3D Interactive Demo")
st.caption("Explore how SVMs use margins, support vectors, and kernels to separate nonlinear data.")

if "chapter" not in st.session_state:
    st.session_state["chapter"] = "Concept"

with st.sidebar:
    st.header("Controls")
    view_options = [chapter for chapter, _, _ in CHAPTERS]
    view = st.selectbox(
        "Chapter",
        view_options,
        index=view_options.index(st.session_state.get("chapter", "Concept")),
    )
    st.session_state["chapter"] = view
    st.caption("Use the visual chapter buttons in the main page or this selector to move through the lesson.")

view = render_chapter_buttons(st.session_state["chapter"])
st.caption(f"Current chapter: {view}")
render_observation_tasks(view)

st.markdown("#### 參數設定與目前參數解讀")
with st.container(border=True):
    control_cols = st.columns([1.15, 1.0, 1.0])
    with control_cols[0]:
        dataset_type = st.selectbox("Dataset", ["circles", "moons", "linear", "blobs", "xor"])
        n_samples = st.slider("Samples", 100, 500, 300, step=50)
        random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    with control_cols[1]:
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        C = st.slider("C", 0.01, 100.0, 1.0, step=0.01, format="%.2f")
        degree = st.slider("Poly degree", 2, 5, 3, disabled=kernel != "poly")
    with control_cols[2]:
        noise = st.slider("Noise", 0.0, 0.30, 0.08, step=0.01)
        gamma_mode = st.selectbox("Gamma", ["scale", "auto", "custom"], disabled=kernel == "linear")
        custom_gamma = st.slider(
            "Custom gamma",
            0.001,
            10.0,
            1.0,
            step=0.001,
            format="%.3f",
            disabled=gamma_mode != "custom" or kernel == "linear",
        )
    gamma = custom_gamma if gamma_mode == "custom" else gamma_mode
    if kernel == "linear":
        gamma = "scale"
    render_parameter_panel(dataset_type, kernel, C, gamma, degree, noise)

if view == "Concept":
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Teaching flow")
        st.write(
            "This demo follows the five-step story from the design chart: start with nonlinear 2D rings, "
            "map them through a feature function, separate them in a 3D feature view, project the idea back "
            "to a nonlinear 2D boundary, then compare it with the real RBF SVM decision function."
        )
        if CONCEPT_IMAGE.exists() and st.checkbox("Show teaching blueprint image"):
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
        if kernel == "poly":
            st.write(f"Polynomial degree: `{degree}`")
        if isinstance(gamma, (int, float)):
            st.write(f"K(x, z) = exp(-{gamma:.3f} * ||x - z||^2)")
        else:
            st.write("For `scale` or `auto`, scikit-learn chooses gamma from the data scale.")

elif view == "Manim Animation":
    st.subheader("Manim concept animation")
    st.write(
        "This short Phase-1 animation follows the teaching blueprint: 2D nonlinear rings, "
        "feature mapping, 3D lift, separating plane, and projected nonlinear boundary."
    )
    video_path = find_manim_video()
    if video_path is None:
        st.warning("No rendered MP4 was found. Render the Manim scene locally first, then refresh this page.")
    else:
        size_mb = video_path.stat().st_size / (1024 * 1024)
        st.caption(f"Rendered video: `{video_path.name}` ({size_mb:.2f} MB). It loads only after you enable playback.")
        if st.checkbox("Load and play the Manim MP4", value=False):
            st.video(str(video_path))
    st.code(
        "python -m manim -ql manim_scenes/svm_kernel_intro.py SVMKernelIntro --media_dir outputs/manim",
        language="powershell",
    )

elif view == "WebGL 3D":
    st.subheader("WebGL 3D kernel trick studio")
    st.write(
        "This Three.js view follows the classmate-style WebGL workflow: original 2D rings, "
        "animated lift to `z = x^2 + y^2`, separating plane, support vectors, and projected nonlinear boundary."
    )
    webgl_samples = min(n_samples, 420)
    if webgl_samples < n_samples:
        st.caption(f"WebGL preview uses {webgl_samples} points for smoother browser interaction.")
    components.html(
        create_webgl_svm_html(n_samples=webgl_samples, noise=noise, random_state=int(random_state)),
        height=760,
        scrolling=False,
    )

elif view == "2D Boundary":
    from src.plotly_visualizer import plot_2d_decision_boundary

    with st.spinner("Training SVM models for the 2D boundary view..."):
        X, y, bundle, linear_metrics, selected_metrics, train_metrics, support_vectors = get_demo_state(
            dataset_type, n_samples, noise, kernel, C, gamma, degree, int(random_state)
        )
    render_metric_strip(selected_metrics, train_metrics, len(support_vectors), len(X))
    render_model_status(train_metrics, selected_metrics, len(support_vectors), len(X))
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

elif view == "3D Kernel View":
    from src.plotly_visualizer import plot_3d_decision_surface, plot_3d_kernel_mapping

    with st.spinner("Training SVM models for the 3D view..."):
        X, y, bundle, linear_metrics, selected_metrics, train_metrics, support_vectors = get_demo_state(
            dataset_type, n_samples, noise, kernel, C, gamma, degree, int(random_state)
        )
    render_metric_strip(selected_metrics, train_metrics, len(support_vectors), len(X))
    render_model_status(train_metrics, selected_metrics, len(support_vectors), len(X))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_3d_kernel_mapping(X, y), width="stretch")
    with c2:
        st.plotly_chart(plot_3d_decision_surface(bundle.selected_model, X, y), width="stretch")

elif view == "Model Metrics":
    from src.plotly_visualizer import plot_confusion_matrix, plot_model_comparison

    with st.spinner("Training SVM models for metrics..."):
        X, y, bundle, linear_metrics, selected_metrics, train_metrics, support_vectors = get_demo_state(
            dataset_type, n_samples, noise, kernel, C, gamma, degree, int(random_state)
        )
    render_metric_strip(selected_metrics, train_metrics, len(support_vectors), len(X))
    render_model_status(train_metrics, selected_metrics, len(support_vectors), len(X))
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

elif view == "Learning Notes":
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

elif view == "Quiz":
    st.subheader("Quick SVM quiz")
    questions = [
        (
            "Support vectors 是什麼？",
            ["離決策邊界最近、會直接影響 margin 的資料點", "所有訓練資料點", "模型預測錯誤的點"],
            0,
            "Support vectors 是最靠近 margin 的關鍵樣本，移動它們通常會改變決策邊界。",
        ),
        (
            "C 變大通常代表什麼？",
            ["模型更不容忍錯誤分類，可能 overfit", "模型完全忽略錯誤分類", "決策邊界一定變成直線"],
            0,
            "較大的 C 會更用力懲罰訓練錯誤，訓練分數可能上升，但泛化不一定更好。",
        ),
        (
            "RBF gamma 變大時常見效果是？",
            ["單一資料點影響範圍變小，邊界更彎曲", "所有點影響範圍變大，邊界更平滑", "support vectors 必定變成 0"],
            0,
            "gamma 越大，每個點的作用越局部，容易形成細碎複雜的決策區域。",
        ),
        (
            "最大 margin 的直覺是什麼？",
            ["讓分隔線離兩類最近點都盡量遠", "讓所有點都落在同一側", "只追求訓練準確率 100%"],
            0,
            "SVM 希望找到最有緩衝空間的分隔邊界，這通常能提升泛化能力。",
        ),
        (
            "Kernel trick 主要解決什麼問題？",
            ["讓非線性資料能在轉換後的特徵空間更容易分開", "把資料量變成 0", "讓模型不需要訓練"],
            0,
            "Kernel 透過相似度計算隱式使用高維特徵，不必真的把每個高維座標算出來。",
        ),
    ]
    score = 0
    for index, (question, options, answer_index, explanation) in enumerate(questions, start=1):
        choice = st.radio(f"{index}. {question}", options, key=f"quiz_{index}")
        if choice == options[answer_index]:
            st.success(explanation)
            score += 1
        else:
            st.info(f"再想一下：{explanation}")
    st.metric("Current score", f"{score} / {len(questions)}")
