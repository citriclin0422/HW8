# SVM Kernel Trick 3D Interactive Demo

An educational Streamlit + Plotly project for learning Support Vector Machines (SVM), maximum margin, support vectors, and the kernel trick through interactive 2D/3D visualizations and a Manim concept animation.

GitHub repository target: [citriclin0422/HW8.git](https://github.com/citriclin0422/HW8.git)

Live demo: [https://hw8svmdemo.streamlit.app/](https://hw8svmdemo.streamlit.app/)

Streamlit deploy page: [https://share.streamlit.io/deploy](https://share.streamlit.io/deploy)

Recommended Streamlit main file:

```text
streamlit_app.py
```

## Project Summary

This project demonstrates how SVM handles nonlinear classification problems:

- Linear SVM finds a maximum-margin decision boundary.
- Support vectors are the key training points that define the margin.
- Nonlinear data such as concentric circles cannot be separated well by a single straight line.
- A teaching mapping `z = x1^2 + x2^2` helps explain the 2D to 3D intuition.
- RBF SVM uses an implicit high-dimensional feature space through the kernel trick.
- Streamlit controls let users adjust dataset type, noise, kernel, `C`, and `gamma`.
- A WebGL 3D view provides a Three.js interaction for the kernel trick workflow.

## 繁體中文說明

本專案是一個 **Support Vector Machine (SVM) Kernel Trick 3D 互動展示教材**，目標是用視覺化方式幫助學習者理解：

- SVM 如何尋找最大間隔（maximum margin）的分類邊界。
- Support vectors 如何決定分類邊界的位置。
- 為什麼同心圓、月牙形、XOR 這類非線性資料無法只靠直線分類。
- Kernel Trick 如何把原本 2D 中難以線性分割的資料，轉換成較容易分割的特徵空間。
- RBF kernel、`C`、`gamma` 對模型邊界平滑度與 overfitting 的影響。

Streamlit App 中包含：

- Manim 動畫播放器，用來觀看 SVM 概念動畫。
- WebGL 3D 互動展示，用滑鼠旋轉與縮放觀察 2D 到 3D 映射。
- 2D decision boundary 圖。
- 3D kernel mapping 圖。
- 3D decision function surface 圖。
- Accuracy、precision、recall、F1、confusion matrix 等模型指標。
- Support vectors 高亮顯示。

## Project Structure

```text
.
|-- streamlit_app.py
|-- app.py
|-- requirements.txt
|-- README.md
|-- src/
|   |-- __init__.py
|   |-- data_generator.py
|   |-- kernel_transform.py
|   |-- plotly_visualizer.py
|   `-- svm_model.py
|-- manim_scenes/
|   `-- svm_kernel_intro.py
|-- outputs/
|   `-- manim/
|       `-- videos/
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

For local Manim rendering, install the optional development requirements:

```powershell
python -m pip install -r requirements-dev.txt
```

## Run Locally

Recommended Streamlit Cloud-compatible entry point:

```powershell
python -m streamlit run streamlit_app.py
```

The original local entry point also works:

```powershell
python -m streamlit run app.py
```

## Render The Manim Animation

The repository includes a rendered MP4 target under `outputs/manim` when generated locally. Streamlit Cloud only plays the rendered MP4; it does not install Manim or render videos during deployment.

To regenerate the animation locally:

```powershell
python -m pip install -r requirements-dev.txt
python -m manim -ql manim_scenes/svm_kernel_intro.py SVMKernelIntro --media_dir outputs/manim
```

Then open the Streamlit app and go to the **Manim Animation** view. The app automatically finds the newest `.mp4` from:

- `outputs/videos`
- `outputs/manim`
- `media/videos`

If no MP4 exists, the app falls back to `Kernel_method.gif`.

## WebGL 3D Interaction

The **WebGL 3D** view embeds a Three.js visualizer inspired by dashboard-style SVM demos:

- Step 1: original 2D concentric rings.
- Step 2: animated lift to `z = x^2 + y^2`.
- Step 3: projected nonlinear 2D boundary.
- Toggles for the separating plane, support vectors, margin rings, and mapping surface.
- Mouse drag rotates the scene; mouse wheel zooms the camera.

## Streamlit Cloud Deployment

Use Streamlit Community Cloud:

1. Push this project to GitHub repository: `citriclin0422/HW8.git`.
2. Open [https://share.streamlit.io/deploy](https://share.streamlit.io/deploy).
3. Choose the GitHub repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Deploy the app.

Important: `requirements.txt` intentionally excludes `manim`. Manim depends on native Linux graphics libraries such as Cairo, X11, and OpenGL context packages that are not needed for the deployed app. The deployed app uses the pre-rendered MP4 already stored in this repository.

This repository includes `runtime.txt` to request Python 3.12 on Streamlit Cloud. This avoids using a very new Python runtime when ML/visualization packages have not fully stabilized on it.

## GitHub Push Commands

If this folder has not been initialized as a Git repository yet:

```powershell
git init
git add .
git commit -m "Build SVM Kernel Trick 3D Streamlit demo"
git branch -M main
git remote add origin https://github.com/citriclin0422/HW8.git
git push -u origin main
```

If the remote already exists:

```powershell
git add .
git commit -m "Update SVM Kernel Trick 3D demo"
git push
```

## Notes

The explicit 3D mapping `z = x1^2 + x2^2` is used for teaching intuition. A true RBF kernel does not simply map data into one visible 3D dimension; it works through similarities in an implicit high-dimensional feature space.
