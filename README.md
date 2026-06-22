# V2B Failure Boundary

本專案研究在已有太陽光電（PV）的建築中，固定式電池、智慧型電動車充電與 Vehicle-to-Building（V2B）調度，能在什麼條件下帶來額外的經濟效益。

專案先以 Gurobi 建立 24 小時能源調度模型，透過 Monte Carlo 方法產生不同天氣、PV 裝置率、EV 數量、電池數量與電價差的情境，再以 XGBoost 分析各項條件對增量節省金額的影響。

## 研究問題

本研究不比較「有無 PV」，而是以已裝設 PV、EV 到站即充的情境作為基準：

| 情境 | 說明 |
| --- | --- |
| `S0_PV_only_unmanaged_EV` | Baseline：已有 PV、沒有固定電池，EV 到站即充且不可放電 |
| `S1_PV_battery` | PV 搭配固定式電池 |
| `S2_PV_smart_EV_charging` | PV 搭配智慧型 EV 充電，但 EV 不可放電 |
| `S3_PV_V2B` | PV 搭配智慧型 EV 充電與 V2B 放電 |
| `S4_PV_battery_V2B` | PV、固定式電池與 V2B 整合調度 |

主要目標變數為：

```text
incremental_saving = pv_only_baseline_cost - scenario_cost
```

`incremental_saving` 為正值時，代表該策略相較於 S0 baseline 額外節省成本；負值則代表該策略沒有產生額外經濟效益。

## 分析流程

```mermaid
flowchart LR
    A[建築電表資料] --> D[Monte Carlo 情境]
    B[晴天與雨天氣象資料] --> D
    D --> E[Gurobi 24 小時最佳化]
    E --> F[S0-S4 模擬資料集]
    F --> G[XGBoost 建模]
    G --> H[模型評估與特徵重要度]
    G --> I[邊際效果與交互作用分析]
    H --> J[圖表、CSV 與 Markdown 報告]
    I --> J
```

## Repository 結構

| 檔案 | 用途 |
| --- | --- |
| `data_collection.ipynb` | 建立 S0-S4 情境、執行 Gurobi 最佳化並產生機器學習資料集 |
| `optimization_case_0512.ipynb` | 單一案例的能源調度模型、規則式模擬與結果視覺化 |
| `run_v2b_xgboost_analysis.py` | 執行 XGBoost、模型評估、特徵重要度、邊際效果及交互作用分析 |
| `v2b_incremental_saving_dataset.csv` | 已產生的分析資料集 |
| `0512電表.csv` | 建築 24 小時負載資料 |
| `466920-2026-05-12.csv` | 晴天氣象與日射量資料 |
| `466920-2026-06-12.csv` | 雨天氣象與日射量資料 |

目前 repository 內的資料集包含：

- 968 組有效 `base_case_id`
- 每組包含 S0-S4 共 5 個情境
- 4,840 筆資料
- 61 個欄位

## 環境需求

建議使用 Python 3.10 以上版本。

只執行 XGBoost 分析時需要：

```bash
python -m pip install numpy pandas matplotlib scikit-learn xgboost
```

若需要 SHAP 解釋圖，可額外安裝：

```bash
python -m pip install shap
```

若要重新產生資料集，還需要 Jupyter、Gurobi Python API，以及可用的 Gurobi license：

```bash
python -m pip install jupyter gurobipy
```

Gurobi 的安裝與授權方式請參考 [Gurobi Documentation](https://docs.gurobi.com/)。

## 快速開始

### 1. Clone repository

```bash
git clone https://github.com/cryptotungy/V2B_failure_boundary.git
cd V2B_failure_boundary
```

### 2. 建立虛擬環境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn xgboost
```

Windows PowerShell 請使用：

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 執行 XGBoost 分析

Repository 已包含產生完成的資料集，因此不需要 Gurobi 即可直接執行：

```bash
python run_v2b_xgboost_analysis.py
```

預設設定：

- 輸入：`v2b_incremental_saving_dataset.csv`
- 輸出資料夾：`v2b_xgboost_outputs/`
- Random seed：`42`
- SHAP 最大抽樣數：`1000`

自訂執行方式：

```bash
python run_v2b_xgboost_analysis.py \
  --input v2b_incremental_saving_dataset.csv \
  --output-dir v2b_xgboost_outputs \
  --random-state 42 \
  --shap-sample 1000
```

查看所有參數：

```bash
python run_v2b_xgboost_analysis.py --help
```

## 重新產生資料集

安裝並啟用 Gurobi license 後，啟動 Notebook：

```bash
jupyter notebook data_collection.ipynb
```

依序執行 Notebook cells，即可重新產生：

```text
v2b_incremental_saving_dataset.csv
```

預設會嘗試建立 1,000 組 Monte Carlo base cases。每組情境會隨機抽樣：

- 天氣型態：晴天或雨天
- PV 裝置率：0–1
- EV 數量：0–10
- 固定式電池數量：0–10
- 電價差參數：0–5

每組 base case 會先計算 S0 baseline，再分別求解 S1-S4。若其中任一最佳化情境不可行，該組 base case 將不會寫入最終資料集。

> 重新產生 1,000 組資料需要執行大量最佳化模型，實際時間會依電腦效能與 Gurobi 環境而異。測試流程時可先降低 `n_base_cases`。

## XGBoost 分析內容

分析程式會：

1. 排除成本、目標值、調度結果與 case-level 比較欄位，避免資料洩漏。
2. 排除 S0，使用 S1-S4 建立整體 intervention model。
3. 分別建立 S1、S2、S3、S4 的 scenario-specific models。
4. 以 `base_case_id` 進行 group-aware train/test split 與 5-fold cross-validation，避免同一 base case 同時出現在訓練集與測試集。
5. 計算 XGBoost native importance、permutation importance 與 group permutation importance。
6. 在已安裝 `shap` 時執行 TreeSHAP 分析。
7. 分析電價差、PV 餘電、EV 數量、電池數量與停車彈性的分箱邊際效果。
8. 比較固定電池與 V2B 的互補或替代關係。

## 主要輸出

執行完成後，`v2b_xgboost_outputs/` 會包含：

| 輸出 | 內容 |
| --- | --- |
| `01_csv_overview.md` | 資料集與欄位群組說明 |
| `01_scenario_summary.csv` | 各情境的增量節省摘要 |
| `02_model_performance.csv` | Train/test 與 cross-validation 指標 |
| `03_feature_importance_native.csv` | XGBoost 原生特徵重要度 |
| `04_feature_importance_permutation.csv` | 單一特徵 permutation importance |
| `05_group_permutation_importance.csv` | PV、電價、EV、電池、負載與策略群組重要度 |
| `06_shap_importance.csv` | SHAP 特徵重要度；未安裝 SHAP 時會記錄跳過原因 |
| `07_binned_marginal_effects.csv` | 重要變數的分箱邊際效果 |
| `08_interaction_analysis.csv` | 固定電池與 V2B 的交互作用分析 |
| `V2B_XGBoost_incremental_saving_report.md` | 自動產生的繁體中文分析報告 |
| `fig_*.png` | 模型診斷、特徵重要度、情境比較與交互作用圖 |

## 重要指標

資料集內包含以下 case-level 比較指標：

| 指標 | 解讀 |
| --- | --- |
| `battery_extra_saving` | 固定式電池相較於 S0 的額外節省 |
| `smart_ev_extra_saving` | 智慧型 EV 充電相較於 S0 的額外節省 |
| `v2b_extra_over_smart_charging` | V2B 相較於僅智慧充電的額外價值 |
| `battery_v2b_interaction_saving` | 固定電池與 V2B 的互補或替代效果 |
| `S4_vs_best_single_saving` | S4 相較於最佳單一策略的額外優勢 |

## 研究限制

- 資料來自最佳化模型與 Monte Carlo 模擬，不代表真實場域的因果效果。
- 目前設定中的 EV 與固定電池退化成本皆為 0，可能高估儲能策略效益。
- 模型未納入需量電費，因此結果主要反映能源費率與時間移轉效益。
- EV 到離場時間、SOC、設備容量及電價結構均受目前假設限制。
- 實際部署前仍應納入真實停車行為、電池退化、充放電效率、需量電費與設備投資成本。
