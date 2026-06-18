# design.md
# SVM Kernel Trick 3D Interactive Demo Project
# For Antigravity IDE / AI Coding Agent

project:
  name: "svm-kernel-trick-3d-demo"
  title: "SVM Kernel Trick 3D 互動展示專案"
  version: "1.0.0"
  owner: "Citric Lin"
  purpose: >
    建立一個可教學、可互動、可發布到 GitHub 的 SVM Kernel Trick 3D 展示專案。
    專案包含 Manim 概念動畫、sklearn RBF SVM 真實決策面、Streamlit + Plotly 互動展示，
    最後可上傳 GitHub，並透過 GitHub Pages 建立 landing page，再連結到 Streamlit App。
  target_audience:
    - "ML 初學者"
    - "學習 SVM / Kernel Trick / RBF Kernel 的學生"
    - "希望理解 2D 非線性資料如何映射到 3D 後線性可分的人"
  core_learning_goals:
    - "理解 SVM maximum margin 的直覺"
    - "理解 kernel trick 的核心概念"
    - "理解 RBF kernel 如何產生非線性決策邊界"
    - "理解 2D feature space 與 3D transformed space 的差異"
    - "能透過互動調整 C 與 gamma 觀察決策面變化"

tech_stack:
  language:
    - "Python 3.10+"
  core_libraries:
    - "numpy"
    - "pandas"
    - "scikit-learn"
    - "plotly"
    - "streamlit"
    - "manim"
    - "matplotlib"
    - "joblib"
  optional_libraries:
    - "scipy"
    - "pillow"
  development_environment:
    - "Antigravity IDE"
  deployment:
    recommended:
      - "GitHub Repository"
      - "Streamlit Community Cloud for interactive app"
    optional:
      - "GitHub Pages for static landing page"
      - "GitHub Actions for automatic static site deployment"

repository_structure:
  root: "svm-kernel-trick-3d-demo/"
  tree: |
    svm-kernel-trick-3d-demo/
    │
    ├─ README.md
    ├─ design.md
    ├─ requirements.txt
    ├─ .gitignore
    │
    ├─ app.py
    │
    ├─ src/
    │  ├─ __init__.py
    │  ├─ data_generator.py
    │  ├─ svm_model.py
    │  ├─ kernel_transform.py
    │  ├─ plotly_visualizer.py
    │  └─ utils.py
    │
    ├─ manim_scenes/
    │  ├─ svm_kernel_intro.py
    │  └─ outputs/
    │
    ├─ notebooks/
    │  └─ 01_svm_kernel_trick_experiment.ipynb
    │
    ├─ outputs/
    │  ├─ figures/
    │  ├─ videos/
    │  ├─ models/
    │  └─ data/
    │
    ├─ docs/
    │  ├─ index.html
    │  ├─ assets/
    │  └─ README.md
    │
    └─ .github/
       └─ workflows/
          └─ pages.yml

phases:
  phase_1_manim_concept_animation:
    title: "Phase 1 - Manim 概念動畫"
    goal: >
      使用 Manim 建立教學動畫，解釋 2D 非線性資料、SVM 線性分不開、
      Kernel Trick 映射到 3D，以及 3D 中可以使用平面分開的概念。
    deliverables:
      - "manim_scenes/svm_kernel_intro.py"
      - "outputs/videos/svm_kernel_trick_intro.mp4"
      - "README.md 中加入動畫說明與執行方式"
    scenes:
      scene_1_problem_setup:
        name: "NonlinearDataProblem"
        description: "展示 2D concentric circles 資料，說明線性分類器無法用直線分開。"
        visual_elements:
          - "2D scatter plot"
          - "inner class and outer class"
          - "failed linear boundary"
      scene_2_kernel_mapping:
        name: "KernelMappingTo3D"
        description: "展示 z = x1^2 + x2^2 的概念映射，將 2D 資料拉到 3D 空間。"
        formulas:
          - "phi(x) = [x1, x2, x1^2 + x2^2]"
          - "K(x,z) = phi(x)^T phi(z)"
      scene_3_3d_linear_separation:
        name: "LinearSeparationIn3D"
        description: "展示 3D 空間中用一個平面分開資料。"
        visual_elements:
          - "3D points"
          - "separating plane"
          - "camera rotation"
      scene_4_summary:
        name: "KernelTrickSummary"
        description: "總結 Kernel Trick：不用真的顯式轉換所有高維特徵，也能計算高維內積。"
        key_message: "Kernel Trick 讓線性 SVM 能處理非線性分類問題。"
    implementation_notes:
      - "Manim 主要負責概念動畫，不需要完全重現 sklearn 真實模型。"
      - "若 3D Manim 複雜度過高，可先用簡化曲面與文字說明。"
      - "輸出 MP4 檔，放入 outputs/videos/。"
    commands:
      install: "pip install manim"
      render_low_quality: "manim -pql manim_scenes/svm_kernel_intro.py KernelMappingTo3D"
      render_high_quality: "manim -pqh manim_scenes/svm_kernel_intro.py KernelMappingTo3D"

  phase_2_sklearn_rbf_svm_decision_surface:
    title: "Phase 2 - sklearn RBF SVM 真實決策面"
    goal: >
      使用 sklearn 生成非線性資料集，訓練 Linear SVM 與 RBF SVM，
      並計算真實的 2D decision boundary 與 3D decision surface。
    deliverables:
      - "src/data_generator.py"
      - "src/svm_model.py"
      - "src/kernel_transform.py"
      - "src/plotly_visualizer.py"
      - "outputs/data/svm_demo_dataset.csv"
      - "outputs/models/rbf_svm_model.joblib"
      - "outputs/figures/2d_decision_boundary.html"
      - "outputs/figures/3d_decision_surface.html"
    dataset_design:
      default_dataset: "make_circles"
      sklearn_function: "sklearn.datasets.make_circles"
      parameters:
        n_samples: 300
        factor: 0.35
        noise: 0.08
        random_state: 42
      alternative_datasets:
        - "make_moons"
        - "custom XOR dataset"
    feature_design:
      original_features:
        - "x1"
        - "x2"
      conceptual_3d_mapping:
        - "z = x1^2 + x2^2"
      target:
        - "y: 0 or 1"
    models:
      linear_svm:
        sklearn_class: "sklearn.svm.SVC"
        params:
          kernel: "linear"
          C: 1.0
      rbf_svm:
        sklearn_class: "sklearn.svm.SVC"
        params:
          kernel: "rbf"
          C: 1.0
          gamma: "scale"
          probability: true
    evaluation:
      metrics:
        - "accuracy"
        - "precision"
        - "recall"
        - "f1"
        - "confusion_matrix"
      expected_result:
        - "Linear SVM 在 concentric circles 上表現較差"
        - "RBF SVM 能成功形成非線性決策邊界"
    visualization:
      plot_1_2d_data:
        description: "顯示原始 2D 資料分布。"
      plot_2_linear_svm_boundary:
        description: "顯示 Linear SVM 的失敗或有限效果。"
      plot_3_rbf_svm_boundary:
        description: "顯示 RBF SVM 的非線性決策邊界。"
      plot_4_3d_kernel_view:
        description: "顯示 z = x1^2 + x2^2 的 3D 映射後資料。"
      plot_5_3d_decision_surface:
        description: "使用 Plotly 顯示 RBF SVM decision_function 的 3D surface。"
    implementation_requirements:
      - "建立可重複使用的 generate_data() 函數。"
      - "建立 train_svm_models() 函數，同時回傳 linear_model 與 rbf_model。"
      - "建立 create_mesh_grid() 產生 2D grid。"
      - "使用 decision_function() 取得決策分數。"
      - "使用 Plotly go.Surface 或 go.Contour 視覺化決策面。"
      - "模型與輸出圖表需儲存到 outputs/。"
    commands:
      run_model_test: "python -m src.svm_model"

  phase_3_streamlit_plotly_interactive_app:
    title: "Phase 3 - Streamlit / Plotly 互動展示"
    goal: >
      建立 Streamlit Web App，讓使用者可以調整 dataset、C、gamma、noise、
      n_samples，並即時觀察 SVM decision boundary 與 3D kernel trick 展示。
    deliverables:
      - "app.py"
      - "requirements.txt"
      - "README.md app 執行說明"
      - "可部署至 Streamlit Community Cloud 的 repo"
    app_layout:
      sidebar_controls:
        - "dataset_type: circles / moons / xor"
        - "n_samples slider: 100 to 1000"
        - "noise slider: 0.00 to 0.30"
        - "C slider: 0.01 to 100.0, log scale if possible"
        - "gamma selector: scale / auto / custom"
        - "custom_gamma slider: 0.001 to 10.0"
        - "kernel selector: linear / rbf / poly"
      main_tabs:
        tab_1: "Concept"
        tab_2: "2D Decision Boundary"
        tab_3: "3D Kernel View"
        tab_4: "Model Metrics"
        tab_5: "Learning Notes"
    required_interactions:
      - "使用者改變 C 後，即時更新 decision boundary。"
      - "使用者改變 gamma 後，即時更新 RBF 邊界複雜度。"
      - "顯示 support vectors 數量。"
      - "顯示 accuracy、f1、confusion matrix。"
      - "Plotly 圖表可旋轉、縮放、hover。"
    plotly_outputs:
      - "2D scatter + contour decision boundary"
      - "3D scatter of mapped data"
      - "3D decision surface"
      - "support vectors highlighted"
    educational_text:
      - "C 小：margin 較寬、容錯較高、邊界較平滑。"
      - "C 大：更努力分對訓練資料，可能 overfitting。"
      - "gamma 小：每個點影響範圍大，邊界平滑。"
      - "gamma 大：每個點影響範圍小，邊界複雜。"
      - "RBF kernel 適合處理非線性分類。"
    commands:
      local_run: "streamlit run app.py"

github_workflow:
  goal: "將專案發布到 GitHub，並提供可分享連結。"
  repository_name: "svm-kernel-trick-3d-demo"
  steps:
    step_1_initialize_repo:
      commands:
        - "git init"
        - "git add ."
        - "git commit -m \"Initial commit: SVM Kernel Trick 3D demo\""
    step_2_create_github_repo:
      instruction: >
        在 GitHub 建立新 repository，名稱建議 svm-kernel-trick-3d-demo。
        建立後將 remote origin 加到本地 repo。
      commands:
        - "git remote add origin https://github.com/citriclin0422/HW8.git"
        - "git branch -M main"
        - "git push -u origin main"
    step_3_streamlit_cloud_deployment:
      recommended: true
      reason: "Streamlit/Plotly 互動 app 不適合直接部署到純 GitHub Pages，建議使用 Streamlit Community Cloud。"
      steps:
        - "前往 https://streamlit.io/cloud"
        - "連結 GitHub 帳號"
        - "選擇 repository: HW8"
        - "設定 main file path: app.py"
        - "Deploy"
      expected_url_format: "https://<app-name>.streamlit.app"
    step_4_github_pages_static_landing:
      optional: true
      reason: "GitHub Pages 適合放靜態 landing page、專案說明、影片、圖片、Streamlit App 連結。"
      docs_folder:
        path: "docs/"
        required_files:
          - "docs/index.html"
          - "docs/assets/"
      github_pages_settings:
        source: "Deploy from a branch"
        branch: "main"
        folder: "/docs"
      expected_url_format: "https://citriclin0422.github.io/HW8/"
    step_5_linking_strategy:
      recommendation:
        - "GitHub Pages 作為專案首頁"
        - "首頁放 Streamlit App 按鈕"
        - "首頁放 Manim MP4/GIF 預覽"
        - "首頁放 GitHub repo link"
        - "Streamlit App 作為主要互動展示"

files_to_generate:
  requirements_txt:
    path: "requirements.txt"
    content: |
      numpy
      pandas
      scikit-learn
      matplotlib
      plotly
      streamlit
      manim
      joblib
      scipy
  gitignore:
    path: ".gitignore"
    content: |
      __pycache__/
      *.pyc
      .venv/
      venv/
      .env
      .DS_Store
      outputs/models/*.joblib
      media/
      .streamlit/secrets.toml
  readme_md:
    path: "README.md"
    must_include:
      - "Project overview"
      - "Phase 1 Manim animation"
      - "Phase 2 sklearn RBF SVM"
      - "Phase 3 Streamlit/Plotly app"
      - "Installation"
      - "How to run locally"
      - "How to deploy"
      - "Screenshots or demo GIF placeholders"
  docs_index_html:
    path: "docs/index.html"
    purpose: "GitHub Pages static landing page"
    must_include:
      - "Project title"
      - "Short explanation of SVM Kernel Trick"
      - "Embedded or linked Manim video/GIF"
      - "Button linking to Streamlit app"
      - "Button linking to GitHub repository"
      - "Section for 3 phases"

implementation_details:
  data_generator_py:
    functions:
      - name: "generate_dataset"
        signature: "generate_dataset(dataset_type='circles', n_samples=300, noise=0.08, random_state=42)"
        returns:
          - "X: numpy array with shape (n_samples, 2)"
          - "y: numpy array"
        notes:
          - "Use make_circles, make_moons, or custom XOR."
  svm_model_py:
    functions:
      - name: "train_svm"
        signature: "train_svm(X, y, kernel='rbf', C=1.0, gamma='scale')"
        returns:
          - "trained sklearn SVC model"
      - name: "evaluate_model"
        signature: "evaluate_model(model, X_test, y_test)"
        returns:
          - "metrics dict"
      - name: "get_support_vectors"
        signature: "get_support_vectors(model)"
        returns:
          - "support vector coordinates"
  kernel_transform_py:
    functions:
      - name: "explicit_3d_mapping"
        signature: "explicit_3d_mapping(X)"
        formula: "z = x1^2 + x2^2"
        returns:
          - "X3D with columns x1, x2, z"
      - name: "rbf_similarity_note"
        formula: "K(x,z)=exp(-gamma*||x-z||^2)"
  plotly_visualizer_py:
    functions:
      - name: "plot_2d_decision_boundary"
        purpose: "Plot scatter + contour decision boundary."
      - name: "plot_3d_kernel_mapping"
        purpose: "Plot explicit 3D mapped data."
      - name: "plot_3d_decision_surface"
        purpose: "Plot decision_function surface."
      - name: "plot_confusion_matrix"
        purpose: "Use Plotly heatmap for confusion matrix."
  app_py:
    required_sections:
      - "Page config"
      - "Sidebar controls"
      - "Dataset generation"
      - "Train/test split"
      - "SVM model training"
      - "Metrics display"
      - "2D Plotly decision boundary"
      - "3D Plotly kernel mapping"
      - "Educational explanation tabs"

acceptance_criteria:
  functionality:
    - "streamlit run app.py can launch the app successfully."
    - "User can change C and gamma and see updated plots."
    - "User can switch dataset type."
    - "App displays support vectors."
    - "App displays classification metrics."
    - "Plotly 3D plot can rotate and zoom."
    - "Manim scene can render at least one MP4 animation."
  educational_quality:
    - "Clearly explains why linear SVM fails on nonlinear circles."
    - "Clearly explains why RBF SVM succeeds."
    - "Clearly shows 2D to 3D mapping intuition."
    - "Contains notes about C and gamma."
  code_quality:
    - "Functions are modular."
    - "No hard-coded absolute paths."
    - "Code runs on Windows and macOS."
    - "requirements.txt is complete."
    - "README.md is clear."
  deployment_quality:
    - "Repo can be pushed to GitHub."
    - "Streamlit Cloud deployment steps are documented."
    - "GitHub Pages docs/index.html can act as a project landing page."

agent_execution_order:
  - order: 1
    task: "Create repository structure and base files."
  - order: 2
    task: "Create requirements.txt, .gitignore, README.md."
  - order: 3
    task: "Implement src/data_generator.py."
  - order: 4
    task: "Implement src/svm_model.py."
  - order: 5
    task: "Implement src/kernel_transform.py."
  - order: 6
    task: "Implement src/plotly_visualizer.py."
  - order: 7
    task: "Implement Streamlit app.py."
  - order: 8
    task: "Test local app with streamlit run app.py."
  - order: 9
    task: "Implement Manim scene in manim_scenes/svm_kernel_intro.py."
  - order: 10
    task: "Render Manim preview video."
  - order: 11
    task: "Create docs/index.html for GitHub Pages landing page."
  - order: 12
    task: "Write deployment instructions."
  - order: 13
    task: "Commit and push to GitHub."
  - order: 14
    task: "Deploy Streamlit app and enable GitHub Pages."

antigravity_agent_prompt: |
  Please build the full project according to this design.md.

  Project goal:
  Build an educational SVM Kernel Trick 3D interactive demo using:
  - Manim for concept animation
  - sklearn for real Linear SVM and RBF SVM training
  - Streamlit + Plotly for interactive 2D/3D visualization
  - GitHub repository and optional GitHub Pages landing page

  Required phases:
  1. Phase 1: Manim concept animation
     - Show nonlinear 2D data
     - Show failed linear separation
     - Show kernel mapping idea
     - Show 3D linear separation intuition

  2. Phase 2: sklearn RBF SVM real decision surface
     - Generate make_circles dataset
     - Train Linear SVM and RBF SVM
     - Show metrics and support vectors
     - Produce 2D decision boundary and 3D surface using Plotly

  3. Phase 3: Streamlit/Plotly interactive app
     - Sidebar sliders for dataset, C, gamma, noise, n_samples
     - Update plots interactively
     - Show accuracy, precision, recall, F1, confusion matrix
     - Show support vectors
     - Include learning notes about C, gamma, RBF kernel

  Deployment:
  - Prepare GitHub repository
  - Add docs/index.html as GitHub Pages landing page
  - Explain that Streamlit interactive app should be deployed on Streamlit Community Cloud
  - Link GitHub Pages landing page to Streamlit app URL

  Please create all files, run local tests, fix errors, and provide final run/deploy instructions.
