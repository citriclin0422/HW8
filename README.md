# SVM Kernel Trick 3D 互動教學展示

Live Demo: [https://hw8svm618.streamlit.app/](https://hw8svm618.streamlit.app/)

GitHub Repository: [citriclin0422/HW8](https://github.com/citriclin0422/HW8.git)

本專案是一個以 **Support Vector Machine (SVM)** 為主題的互動式教學網站，使用 Streamlit、scikit-learn、Plotly 與 WebGL/Three.js 展示最大間隔、support vectors、kernel trick 與 3D 特徵映射的概念。

## 專案目標

本網站希望讓學習者透過互動方式理解：

- SVM 如何尋找最大 margin 的分類邊界。
- Support vectors 為什麼會決定分類邊界的位置。
- `C`、`gamma`、`degree` 這些參數如何影響模型複雜度。
- 線性模型為什麼無法處理 circles、moons、XOR 等非線性資料。
- Kernel Trick 如何將原本難以分割的 2D 資料轉換到較容易分割的特徵空間。
- 3D 視覺化如何幫助理解 `z = x^2 + y^2` 這類教學用 feature mapping。

## 網站流程

網站左側 sidebar 提供 **Chapter** 下拉選單，用來切換各教學章節：

主畫面上方也提供 **Chapter Navigator** 視覺化章節按鈕，方便使用者用更直覺的方式切換教學流程。每個章節會顯示「觀察任務」與「目前參數解讀」，引導使用者知道該調整哪些參數、觀察哪些圖形變化。

1. **Concept**
   - 說明 SVM 教學流程。
   - 顯示最大 margin、hyperplane、kernel mapping 等核心公式。
   - 可選擇顯示教學設計概念圖。

2. **Manim Animation**
   - 保留 Manim 動畫的本機渲染指令。
   - Streamlit Cloud 不即時執行 Manim，避免 Cairo、Pango、FFmpeg、OpenGL 等系統依賴造成部署失敗。
   - 網頁端提供手動載入 MP4 的選項，避免進入頁面時自動拖慢網站。
   - 目前新版動畫長度約 30 秒內，內容對應 Phase-1 Manim 概念動畫。

3. **WebGL 3D**
   - 使用 Three.js 展示 2D concentric rings。
   - 可互動觀察資料點從 2D lift 到 `z = x^2 + y^2`。
   - 顯示 separating plane、support vectors、margin rings 與 mapping surface。

4. **2D Boundary**
   - 使用 scikit-learn `SVC` 訓練 SVM。
   - 比較 linear SVM 與目前選擇 kernel 的 decision boundary。
   - 顯示 support vectors 與 margin contour。

5. **3D Kernel View**
   - 顯示 explicit 3D kernel mapping。
   - 顯示 SVM decision function surface。
   - 幫助理解「2D 中非線性、轉換後較容易分割」的直覺。

6. **Model Metrics**
   - 顯示 test accuracy、train accuracy、generalization gap、F1、support-vector ratio。
   - 顯示 confusion matrix 與模型比較圖。
   - 可用來觀察 overfitting / underfitting。

7. **Learning Notes**
   - 說明 `C`、`gamma`、support vectors 與 3D mapping 的直覺。
   - 補充 RBF kernel 並不是單純映射到一個可見 3D 軸，而是透過隱式高維相似度運算。

8. **Quiz**
   - 提供 SVM 核心概念小測驗。
   - 透過選擇題複習 support vectors、C、gamma、margin 與 kernel trick。

## 互動控制

左側 sidebar 可調整：

- `Dataset`: `circles`、`moons`、`linear`、`blobs`、`xor`
- `Samples`: 100 到 500，避免雲端運算過重
- `Noise`: 控制資料雜訊
- `Kernel`: `rbf`、`linear`、`poly`
- `C`: 控制錯誤分類懲罰強度
- `Gamma`: `scale`、`auto` 或自訂數值
- `Poly degree`: polynomial kernel 的 degree
- `Random seed`: 控制資料生成隨機性

## 互動介面強化

本版新增下列互動式教學元素：

- **Chapter Navigator**：主畫面上方的章節按鈕列，可快速切換 Concept、Manim、WebGL、2D Boundary、3D Surface、Metrics、Notes、Quiz。
- **觀察任務卡**：每個章節提供 1-2 個操作任務，引導使用者觀察 SVM 邊界、support vectors、kernel mapping 或模型指標。
- **目前參數解讀**：依據 dataset、kernel、`C`、`gamma`、`degree`、noise 自動產生簡短說明。
- **模型狀態燈**：在 2D Boundary、3D Kernel View、Model Metrics 中，用 train/test gap 判斷目前模型是否穩定、需要觀察或有 overfitting 風險。
- **手動載入 Manim MP4**：影片不自動載入，避免拖慢頁面；使用者進入 Manim Animation 後可自行勾選播放。

## 專案結構

```text
.
|-- streamlit_app.py          # Streamlit Cloud 入口檔
|-- app.py                    # 主要 Streamlit app
|-- requirements.txt          # 雲端部署用依賴，不包含 Manim
|-- requirements-dev.txt      # 本機 Manim 開發依賴
|-- runtime.txt               # Python 版本提示
|-- README.md
|-- src/
|   |-- data_generator.py     # 產生教學資料集
|   |-- kernel_transform.py   # 3D feature mapping
|   |-- plotly_visualizer.py  # Plotly 2D/3D 圖表
|   `-- svm_model.py          # SVM 訓練與評估
|-- manim_scenes/
|   `-- svm_kernel_intro.py   # Manim 教學動畫原始碼
|-- outputs/
|   `-- manim/                # 本機渲染輸出
|-- docs/
|   |-- index.html
|   `-- assets/
`-- *.png / *.gif             # 教學設計與參考素材
```

## 本機執行

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Manim 本機渲染

Manim 只建議在本機渲染，不建議在 Streamlit Cloud 即時執行。

```powershell
python -m pip install -r requirements-dev.txt
python -m manim -ql manim_scenes/svm_kernel_intro.py SVMKernelIntro --media_dir outputs/manim
```

雲端部署端的 `requirements.txt` 不包含 `manim`，這樣可以避免 `glcontext`、`pycairo`、X11、Cairo、Pango 等原生套件建置錯誤。

已預先渲染的 MP4 位於：

```text
outputs/manim/videos/svm_kernel_intro/480p15/SVMKernelIntro.mp4
```

Streamlit 的 **Manim Animation** 章節會在使用者勾選 **Load and play the Manim MP4** 後才載入影片。

## Streamlit Cloud 部署

1. 將專案 push 到 GitHub：`citriclin0422/HW8.git`
2. 前往 [https://share.streamlit.io/deploy](https://share.streamlit.io/deploy)
3. 選擇 GitHub repo
4. Main file path 設定為：

```text
streamlit_app.py
```

5. Python version 建議選 `3.12`
6. Deploy 或 Reboot app

目前正式展示網址：

[https://hw8svm618.streamlit.app/](https://hw8svm618.streamlit.app/)

## 已修正的重要問題

- 移除 Streamlit Cloud 不必要的 CORS/XSRF server override，避免前端空白。
- 移除雲端 `requirements.txt` 中的 Manim，避免原生依賴安裝失敗。
- 將雲端 MP4 改為手動載入，改善頁面流暢度。
- 重新製作 30 秒內的 Phase-1 Manim 概念動畫，並以手動載入方式接回 Streamlit。
- 新增 Chapter Navigator、觀察任務卡、參數解讀卡與模型狀態燈，強化互動教學體驗。
- 將章節導覽移到 sidebar `Chapter` 下拉選單。
- 修正 `streamlit_app.py` 的 rerun 問題：避免 `from app import *` 被 Python import cache 影響，造成主頁可見但互動後無法進入其他章節。
- README 改為繁體中文說明專案流程與部署方法。

## Troubleshooting

- 如果網站主頁出現但無法切換章節，請確認使用最新 commit，並在 Streamlit Cloud 按 **Reboot app**。
- 如果 Cloud 使用 Python 3.14，建議在 **Manage App -> Settings -> Advanced settings** 改為 Python 3.12。
- 如果部署時出現 Manim、glcontext、pycairo、X11、Cairo 相關錯誤，請確認 `requirements.txt` 沒有放 `manim`。
- 如果互動圖表較慢，請將 `Samples` 調低到 100-300。

## 備註

網站中的 `z = x^2 + y^2` 是教學用的可視化映射，用來幫助理解 Kernel Trick。真實 RBF kernel 並不是只映射到單一 3D 維度，而是透過隱式高維特徵空間與相似度計算達成非線性分類。
