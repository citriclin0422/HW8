# SVM Kernel Trick 3D Interactive Demo

Live demo: [https://hw8svmdemo.streamlit.app/](https://hw8svmdemo.streamlit.app/)

An educational Streamlit project for learning Support Vector Machines (SVM), maximum margin, support vectors, and the kernel trick through interactive 2D/3D visualizations.

## 專案摘要

這是一個 **Support Vector Machine (SVM) Kernel Trick 3D 互動教學展示**。設計重點是讓初學者能看懂：

- SVM 如何尋找最大 margin 的分類邊界。
- Support vectors 為什麼是決定邊界的關鍵資料點。
- 非線性資料，例如 circles、moons、XOR，為什麼無法只靠一條直線分開。
- Kernel Trick 如何把原本 2D 中難分的資料，轉換成較容易分開的特徵空間。
- `C`、`gamma`、`degree` 對 overfitting / underfitting 的影響。

Streamlit App 目前包含：

- **Concept**：SVM 與 Kernel Trick 的教學流程與公式。
- **Manim Animation**：保留本機預渲染指令；Cloud 端不即時執行 Manim。
- **WebGL 3D**：使用 Three.js 展示 2D 圓形資料如何 lift 到 `z = x^2 + y^2`。
- **2D Boundary**：比較 linear SVM 與目前選擇 kernel 的決策邊界。
- **3D Kernel View**：顯示 kernel mapping 與 decision function surface。
- **Model Metrics**：顯示 test/train accuracy、generalization gap、F1、support-vector ratio 與 confusion matrix。
- **Learning Notes**：解釋 `C`、`gamma`、support vectors 與 3D mapping。
- **Quiz**：用五題小測驗複習 SVM 核心概念。

## Project Structure

```text
.
|-- streamlit_app.py
|-- app.py
|-- requirements.txt
|-- requirements-dev.txt
|-- runtime.txt
|-- README.md
|-- src/
|   |-- data_generator.py
|   |-- kernel_transform.py
|   |-- plotly_visualizer.py
|   `-- svm_model.py
|-- manim_scenes/
|   `-- svm_kernel_intro.py
|-- outputs/
|   `-- manim/
|-- docs/
|   |-- index.html
|   `-- assets/
`-- *.png / *.gif teaching assets
```

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Locally

```powershell
python -m streamlit run streamlit_app.py
```

The local development entry point also works:

```powershell
python -m streamlit run app.py
```

## Manim Animation

Manim should be rendered locally only. Streamlit Community Cloud should not install or run Manim because it requires native dependencies such as Cairo, Pango, FFmpeg, X11, and OpenGL context packages.

Install optional development dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Render locally:

```powershell
python -m manim -ql manim_scenes/svm_kernel_intro.py SVMKernelIntro --media_dir outputs/manim
```

The deployed Streamlit app currently keeps video playback disabled to preserve page smoothness. The Manim scene and render command remain in the repository for local review.

## Streamlit Cloud Deployment

1. Push this project to GitHub repository: `citriclin0422/HW8.git`.
2. Open [https://share.streamlit.io/deploy](https://share.streamlit.io/deploy).
3. Choose the GitHub repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Deploy the app.

Recommended Streamlit Cloud settings:

- Python version: `3.12`
- Do not include `manim` in `requirements.txt`
- Reboot the app after pushing changes that modify imports or cached modules

## Requirements

`requirements.txt` intentionally stays small:

```text
numpy
scikit-learn
plotly
streamlit
```

This keeps Streamlit Cloud deployment stable and avoids native build failures from Manim-related packages.

## Troubleshooting

- If the app starts but the page is blank, reboot the app in Streamlit Cloud.
- If Streamlit Cloud uses Python 3.14, change the Python version to 3.12 in **Manage App -> Settings -> Advanced settings**.
- If a new function import fails after deployment, push the latest code and reboot the app so Streamlit clears stale module state.
- If the page feels slow, reduce `Samples` to 100-300 and avoid loading video assets inside the Cloud app.

## Notes

The visible 3D mapping `z = x^2 + y^2` is a teaching simplification. A true RBF kernel does not literally map data into one visible 3D axis; it computes similarities in an implicit high-dimensional feature space.
