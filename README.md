# Dementia Longitudinal Modeling

This repository contains code for a research project on **longitudinal modeling of dementia progression using clinical data**. The project is conducted as part of an interdisciplinary collaboration between researchers and medical professionals.

## Overview

Dementia is a progressive condition in which patients may exhibit different trajectories of cognitive and functional decline over time.

This project focuses on analyzing **longitudinal clinical data** to investigate disease progression patterns and develop statistical and machine learning models for modeling patient trajectories.

## Objectives

* Explore longitudinal patterns in clinical measurements.
* Analyze changes in patient conditions over time.
* Develop statistical and machine learning models for longitudinal data.
* Investigate factors associated with different progression trajectories.
* Evaluate model performance and interpretability in a clinical context.

## Methodology

The project may involve:

* Exploratory Data Analysis (EDA)
* Longitudinal data preprocessing
* Missing data analysis
* Statistical modeling
* Machine learning
* Model evaluation and interpretation

Detailed methodology will be updated as the research progresses.

## Repository Structure

```text
.
├── data/          # Local input data (ignored by Git)
├── notebooks/     # Exploratory analysis and experiments
├── src/           # Modeling and preprocessing code
├── results/       # Local experimental results (ignored by Git)
└── README.md
```

## Data Privacy

The clinical data used in this research are **not publicly available** due to patient privacy, research ethics, and data-sharing restrictions.

The current branch snapshot excludes source clinical data, patient-level results, and notebook outputs. Earlier commits may still contain previously uploaded files; ignoring or removing files from the latest snapshot does not remove them from Git history.

## Status

🚧 **Work in Progress**

This repository is actively being developed as part of an ongoing research project. Methods, experiments, and documentation may change as the research progresses.

## 本機資料

`data/`、`results/`、CSV／Excel 資料、訓練模型、虛擬環境與本機 Jupyter 設定均由 `.gitignore` 排除。
病人資料與逐病人實驗產出只保留在本機，請自行準備取得授權的輸入資料。
MERF 預期輸入位於 `data/processed/model_df_next_mmse3.csv`。

## Notebook 與 Git

Notebook 的本機執行結果可以保留；Git clean filter 會在 `git add` 時移除輸出與 execution count，避免將資料表／病人 ID 輸出提交。
不會改動本機 notebook，也不會刪除本機資料。`.gitignore` 本身無法清除 notebook 內的輸出。

首次 clone 後，先用本機 Python 執行一次：

```powershell
python scripts/setup_git.py
```

若 Python 不在 PATH，改用實際 Python 執行檔路徑，例如 `D:/anaconda3/python.exe scripts/setup_git.py`。
此設定保存在本機 `.git/config`；更換 Python 路徑時重新執行。設定後再 `git add`。

## MERF

入口：`notebooks/build_models/mixed effect ml/next_mmse_merf_nan_CU_AD_nonAD.ipynb`。
環境、完整執行與驗證指令請見 [MERF 說明](notebooks/build_models/mixed%20effect%20ml/README_merf_zh.md)。
原始 RF/SVM/XGBoost notebook 保留，MERF 共用程式為同目錄的 `merf_workflow.py`。
