# MERF（原始尺度、保留 NaN）

主檔：`next_mmse_merf_nan_CU_AD_nonAD.ipynb`。原始 RF/SVM/XGBoost notebook 與空白 `merf.ipynb` 保留。
共用實作：`merf_workflow.py`。不需 validation adapter，不改 site-packages。

在專案根目錄的 PowerShell 執行：

```powershell
D:/anaconda3/python.exe -m venv .venv-merf
./.venv-merf/Scripts/python.exe -m pip install -r 'notebooks/build_models/mixed effect ml/requirements-merf.txt'
./.venv-merf/Scripts/python.exe 'notebooks/build_models/mixed effect ml/build_notebook.py'
./.venv-merf/Scripts/python.exe 'notebooks/build_models/mixed effect ml/execute_notebook.py'
./.venv-merf/Scripts/python.exe 'notebooks/build_models/mixed effect ml/verify_artifacts.py'
```

`build_notebook.py` 會重建乾淨 notebook、專案內 kernel spec，清掉 notebook 舊輸出；一般重跑只需最後一行。
若 pip 設定禁止索引，可加 `--isolated --index-url https://pypi.org/simple`。

只跑人工檢查或不用 Jupyter 的完整實驗：

```powershell
./.venv-merf/Scripts/python.exe 'notebooks/build_models/mixed effect ml/merf_workflow.py' --synthetic-only
./.venv-merf/Scripts/python.exe 'notebooks/build_models/mixed effect ml/merf_workflow.py'
```

互動開啟：選 `.venv-merf/Scripts/python.exe` 作 notebook kernel；或在啟動 Jupyter 前設定 `$env:JUPYTER_PATH = "$PWD/.jupyter"`，選 Python (MERF project)。

固定版本在 requirements-merf.txt；實際 Python/numpy/pandas/sklearn/merf 版本與 executable 在 results/merf_nan_CU_AD_nonAD/environment.json。
完整相依版本另存 environment-freeze.txt。
若要重建這次實際環境，可用 `pip install -r results/merf_nan_CU_AD_nonAD/environment-freeze.txt`。

輸出 `results/merf_nan_CU_AD_nonAD/`：

- `report_zh.md`、`metrics.csv`、`test_predictions.csv`：中文報告、所有／共同可評估子集／校正數分層指標及逐病人預測。
- `data_audit.json`、`processing_events.csv`、`category_conflicts.json`：資料語意、缺失、逐列處理與跨組衝突。
- `synthetic_checks.json`、`execution_status.json`：人工驗證及真實實驗狀態。
- `smoke/{Group}/`：正式調參前真實資料小型驗證。
- `{Group}/split_ids.json`、`tuning.csv`、`selected_parameters.json`、`*_iterations.csv`、`group_report.json`：切分、選模、變異數及 GLL 記錄。
- `{Group}/*_sample_index.csv`、`*_excluded.csv`、`*_X_missingness.csv`：逐樣本時間稽核、缺失 target 排除、X 缺值比例。

政策：同日保留 source 第一列；category 衝突整人排除；缺失特徵保持 NaN；缺失 target 排除且不回退。
上游 blood nearest 可能取未來檢驗，晚於歷史 visit／日期不明者遮罩 NaN，不補其他檢驗。
保留原特徵類別，修正血液 gap 為過去檢驗距 visit 的 elapsed days log；第一個 visit gap 保留 NaN。
CSV 只能遮罩已配對檢驗，無法重建當時真正可用的上一筆血液量測。

MERF 校正使用 Gaussian random-intercept 假設的條件均值，只使用最終 target 前、第一筆之後的有效 transition 殘差。
tau² 與 sigma² 僅取自訓練模型；m=0 使用群體預測。20 iterations 為固定預算，並不保證收斂；查看 GLL 與報告。
基底原檔 hash 保存在 original_notebook.sha256，便於確認原始 notebook 未被改動。
