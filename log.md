# HW8 SVM 3D Interactive Demo 開發與除錯紀錄

日期：2026-06-18  
Live Demo：[https://hw8svm618.streamlit.app/](https://hw8svm618.streamlit.app/)  
GitHub：[citriclin0422/HW8](https://github.com/citriclin0422/HW8.git)

## 1. 專案目標

本專案目標是建立一個 **Support Vector Machine (SVM) Kernel Trick 3D 互動教學網站**，讓使用者透過 Streamlit、Plotly、WebGL/Three.js 與 Manim 預渲染動畫理解：

- SVM 最大 margin 的概念
- Support vectors 如何決定分類邊界
- 非線性資料為什麼需要 kernel trick
- `z = x^2 + y^2` 教學映射如何幫助理解 2D 到 3D 的轉換
- 真實 RBF kernel 與教學用 3D 映射的差異
- `C`、`gamma`、`degree` 對 overfitting / underfitting 的影響

## 2. 初始部署問題

一開始 `requirements.txt` 包含 `manim`，導致 Streamlit Community Cloud 安裝失敗。

主要錯誤包含：

- `glcontext` 編譯失敗，缺少 `X11/Xlib.h`
- `pycairo` metadata 產生失敗，缺少 Cairo / pkg-config
- Manim 依賴 Cairo、Pango、FFmpeg、OpenGL / X11 等原生系統套件，不適合在 Streamlit Cloud 即時安裝與渲染

處理方式：

- 將 `requirements.txt` 精簡為雲端執行需要的套件：`numpy`、`scikit-learn`、`plotly`、`streamlit`
- 新增 `requirements-dev.txt`，把 Manim 留給本機開發與渲染
- README 補充：Manim 只在本機預渲染，不在 Streamlit Cloud 即時 render

## 3. Streamlit Cloud 空白與前端無法載入

部署後曾出現 server 已啟動，但頁面空白或前端不穩的情況。

處理方式：

- 移除 `.streamlit/config.toml` 中不必要的 server override：
  - `enableCORS = false`
  - `enableXsrfProtection = false`
  - `headless = true`
- 僅保留 Streamlit Cloud 安全的最小設定
- 建議 Streamlit Cloud 使用 Python 3.12，避免太新的 Python runtime 造成套件相容性問題

## 4. 子頁面無法進入的根因

主頁可顯示，但在本機與 Streamlit Cloud 點選章節後無法正常進入子頁面。

根因：

```python
from app import *
```

Streamlit 每次 widget 互動都會 rerun 入口檔，但 Python 會快取已 import 的 `app` module。第一次載入可以顯示主頁，第二次互動時 `app.py` 不會重新執行，因此造成頁面卡住或空白。

處理方式：

將 `streamlit_app.py` 改成每次 rerun 都執行 `app.py`：

```python
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
```

驗證：

- 本機 `streamlit_app.py` 可正常 rerun
- `Concept`、`Manim Animation`、`WebGL 3D`、`2D Boundary`、`3D Kernel View`、`Model Metrics`、`Learning Notes`、`Quiz` 均可切換

## 5. 參考同學資料後的調整

參考 `SVM 教學網站部署.pdf` 與 `classmate_log.md` 後，整理出幾個重點：

- Manim 不應在 Streamlit Cloud 即時渲染
- 真正互動應交給 sklearn + Plotly / WebGL
- 資料量應控制在 100-500
- 使用 `st.cache_data` 降低重複訓練負擔
- 網站不只要有圖表，也要有教學提示、參數解讀和小測驗

實作調整：

- 新增資料集：`linear`、`blobs`
- 保留：`circles`、`moons`、`xor`
- 新增 `poly degree` 控制
- 樣本數上限降為 500
- 加入 train/test accuracy、generalization gap、support-vector ratio
- 新增 Quiz 頁面

## 6. WebGL 3D 互動展示

為了讓使用者能即時操作 2D 到 3D 的 kernel trick 直覺，加入 WebGL / Three.js 視覺化。

功能包含：

- 2D concentric rings
- 點雲 lift 到 `z = x^2 + y^2`
- 3D separating plane
- projected nonlinear boundary
- support vectors / margin rings / mapping surface 開關
- 滑鼠旋轉與縮放

後續為了保持 Streamlit 穩定，WebGL 只在使用者進入 **WebGL 3D** 章節時載入。

## 7. Manim 動畫重製

原本 `SVMKernelIntro.mp4` 約 15 秒，概念表達不完整，且曾有文字重疊問題。

依照 `SVM_3D教學設計概念圖表.png` 第 3 框 Phase-1 內容，重新製作新版 Manim 動畫：

流程：

1. 標題開場
2. 顯示 2D concentric rings
3. 說明直線無法分開
4. 顯示 feature mapping：`phi(x, y) = (x, y, x^2 + y^2)`
5. 點雲提升到 3D
6. 顯示拋物面 `z = x^2 + y^2`
7. 顯示 3D separating plane
8. 投影回 2D 形成非線性邊界
9. 總結 Kernel Trick 與真實 RBF kernel 的差異

驗證結果：

- Manim 渲染成功
- MP4 長度：28.60 秒
- 檔案大小：約 990 KB
- 修正 3D 段落中固定文字跟著相機旋轉與底部文字重疊問題

## 8. Manim 接回 Streamlit

為了避免影片拖慢網站，沒有讓 MP4 自動載入。

實作方式：

- `Manim Animation` 章節預設只顯示說明與渲染指令
- 使用者勾選 **Load and play the Manim MP4** 後才載入影片
- MP4 仍保留在 repo：

```text
outputs/manim/videos/svm_kernel_intro/480p15/SVMKernelIntro.mp4
```

## 9. UI / UX 強化

為了讓 Streamlit 網頁更像互動式教學網站，進行多次介面調整。

新增：

- **Chapter Navigator**：主畫面上的章節按鈕列
- active chapter 使用按鈕底色顯示，不再使用 `[current]` 文字
- **觀察任務卡**：每個章節提供操作任務
- **目前參數解讀**：依據 dataset、kernel、`C`、`gamma`、`degree`、noise 自動產生提示
- **模型狀態燈**：依 train/test gap 判斷泛化穩定、需要觀察或高 overfitting 風險
- 指標卡片 CSS 美化

## 10. 參數控制位置調整

原本所有參數控制都放在 sidebar。後續討論後認為教學網站應讓使用者更直接看到「參數設定」與「解讀」的關係。

調整方式：

- sidebar 僅保留 `Chapter` 章節選擇
- `Dataset`、`Samples`、`Kernel`、`C`、`gamma`、`degree`、`Noise`、`Random seed` 移到主畫面的 **參數設定與目前參數解讀** 區塊
- 使用三欄排版，讓控制項更接近圖表與教學文字

## 11. README 更新

README 已更新為繁體中文為主，包含：

- 專案目標
- 網站流程
- 互動控制
- UI 強化說明
- 專案結構
- 本機執行方式
- Manim 本機渲染方式
- Streamlit Cloud 部署方式
- 已修正問題
- Troubleshooting
- Live Demo 連結

## 12. 目前狀態

目前版本已可正常部署於：

[https://hw8svm618.streamlit.app/](https://hw8svm618.streamlit.app/)

主要功能狀態：

- Streamlit 主頁可顯示
- 章節切換正常
- Manim Animation 可手動載入 MP4
- WebGL 3D 可互動
- 2D / 3D SVM 圖表可即時更新
- Model Metrics 顯示模型狀態與 overfitting 診斷
- Quiz 可互動作答

## 13. 後續可改善方向

- 進一步將 WebGL 從 `st.components.v1.html` 改為外部靜態 HTML 或新版 Streamlit iframe API
- 將 Quiz 分數記錄到 `st.session_state`
- 加入「重設參數」按鈕
- 加入更多教學提示，例如針對不同 kernel 顯示不同案例建議
- 可選擇是否把 MP4 放到外部 CDN，以減少 repo 大小
