<div align="center">
  <img src="Amazon_logo.png" alt="Amazon Logo" width="200">
</div>
# 🛒 AMAZON User Segmentation

### *An End-to-End Machine Learning System for Customer Intelligence*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/badge/CI-Passing-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/Aranya2801/AMAZON-User-Segmentation/actions)
[![Tests](https://img.shields.io/badge/Tests-18%20Passed-success?style=for-the-badge&logo=pytest)](tests/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=for-the-badge)](https://black.readthedocs.io)

<br/>

> **Production-grade customer intelligence platform** that fuses RFM analytics, multi-algorithm unsupervised clustering, and gradient-boosted supervised classification into a single automated pipeline — served through a real-time Streamlit dashboard for daily business decision-making.

<br/>

| 🧪 Model Accuracy | 📊 Silhouette Score | 👥 Customers | 🔬 Features | 📈 ML Models |
|:-----------------:|:-------------------:|:------------:|:-----------:|:------------:|
| **98.8% F1** | **0.8328** | **5,000** | **35** | **6** |

</div>

---

## 📌 Table of Contents

- [🚀 Project Overview](#-project-overview)
- [🏗️ System Architecture](#️-system-architecture)
- [📁 Repository Structure](#-repository-structure)
- [🔬 Methodology](#-methodology)
- [📊 Results & Insights](#-results--insights)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚡ Quick Start](#-quick-start)
- [🖥️ Dashboard](#️-dashboard)
- [📓 Notebooks](#-notebooks)
- [🧪 Testing](#-testing)
- [📂 Dataset](#-dataset)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🚀 Project Overview

This project delivers a **full-stack customer segmentation intelligence system** built to MIT-level engineering standards. It answers the most critical question in e-commerce:

> *"Who are my customers — and how should I treat each of them differently?"*

### What It Does

```
Raw Customer Data  ──►  Feature Engineering  ──►  RFM Scoring  ──►  ML Clustering
                                                                           │
Business Actions  ◄──  Churn Prediction  ◄──  Classification  ◄──  Segment Labels
       │
       └──►  Real-Time Streamlit Dashboard  (daily operational use)
```

### Five Customer Segments Discovered

| Segment | Priority | LTV Tier | Churn Risk | Action Strategy |
|---------|----------|----------|------------|-----------------|
| 🟠 **Loyal Shopper** | RETAIN | HIGH | LOW | VIP programme, exclusive access |
| 🔵 **Premium Spender** | UPSELL | VERY HIGH | MEDIUM | Concierge, premium launches |
| 🩵 **Bargain Hunter** | CONVERT | MEDIUM | MEDIUM | Flash-sales, bundle deals |
| 🟢 **Occasional Buyer** | ACTIVATE | LOW | HIGH | Re-engagement, free Prime trial |
| 🔴 **Inactive User** | WIN-BACK | VERY LOW | CRITICAL | 30% coupon win-back sequence |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AMAZON SEGMENTATION SYSTEM                       │
├─────────────┬────────────────┬──────────────┬───────────────────────┤
│  DATA LAYER │  ML PIPELINE   │   EVALUATION │   APPLICATION         │
├─────────────┼────────────────┼──────────────┼───────────────────────┤
│             │                │              │                       │
│  Raw CSV    │  RFM Engine    │  Silhouette  │  Streamlit Dashboard  │
│  Raw XLSX   │  Feature Eng.  │  Davies-B.   │  ├─ Executive KPIs   │
│             │  RobustScaler  │  Calinski-H  │  ├─ Segment Explorer │
│  Processed  │  PCA (10D)     │  ARI / NMI   │  ├─ RFM Analysis     │
│  CSV        │                │  CV F1 Macro │  ├─ ML Insights      │
│             │  KMeans ★      │              │  ├─ Churn Predictor  │
│  Model      │  GMM           │  Stability   │  └─ Data Explorer    │
│  Artifacts  │  Agglomerative │  Bootstrap   │                       │
│  (joblib)   │                │  (20 runs)   │  CI/CD Pipeline       │
│             │  Random Forest │              │  GitHub Actions       │
│             │  Grad. Boost.  │  Business    │  18 Unit Tests        │
│             │  LogReg        │  KPI Mapper  │  Black + Flake8       │
└─────────────┴────────────────┴──────────────┴───────────────────────┘
```

---

## 📁 Repository Structure

```
AMAZON-User-Segmentation/
│
├── 📊 data/
│   ├── raw/
│   │   ├── amazon_customer_data.csv      # 5,000 customers × 29 features
│   │   └── amazon_customer_data.xlsx     # Excel version (upload-ready)
│   └── processed/
│       ├── amazon_segmented_final.csv    # Enriched with RFM + clusters
│       └── amazon_rfm_enriched.csv       # RFM-only enrichment
│
├── 🤖 src/
│   ├── pipeline/
│   │   └── data_pipeline.py              # Full ML orchestrator
│   ├── models/
│   │   ├── evaluate.py                   # Cluster + classifier evaluation
│   │   ├── robust_scaler.pkl             # Saved RobustScaler
│   │   ├── pca_10d.pkl                   # Saved PCA transformer
│   │   ├── classifier_RandomForest.pkl   # Best classifier
│   │   ├── standard_scaler.pkl           # Classification scaler
│   │   └── label_encoder.pkl             # Segment label encoder
│   ├── dashboard/
│   │   └── app.py                        # Streamlit dashboard (6 pages)
│   └── utils/
│       └── helpers.py                    # Config, validators, timers
│
├── 📓 notebooks/
│   └── (Jupyter EDA + modelling notebooks)
│
├── 📈 reports/
│   ├── figures/                          # 11 publication-quality plots
│   │   ├── fig01_segment_donut.png
│   │   ├── fig02_rfm_distributions.png
│   │   ├── fig03_pca_2d.png
│   │   ├── fig04_feature_importance.png
│   │   ├── fig05_spend_orders_bubble.png
│   │   ├── fig06_correlation_heatmap.png
│   │   ├── fig07_radar_profile.png
│   │   ├── fig08_churn_boxplot.png
│   │   ├── fig09_elbow_silhouette.png
│   │   ├── fig10_region_segments.png
│   │   └── fig11_ltv_violin.png
│   └── tables/
│       ├── cluster_profiles.csv
│       ├── feature_importances.csv
│       ├── pipeline_summary.json
│       └── evaluation_report.json
│
├── 🧪 tests/
│   └── test_pipeline.py                  # 18 unit tests (all passing)
│
├── ⚙️  configs/
│   └── config.yaml                       # Master configuration
│
├── 🔄 .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions CI/CD
│
├── requirements.txt
├── LICENSE                               # MIT
└── README.md
```

---

## 🔬 Methodology

### Stage 1 — Data Ingestion & Validation
- Automated schema validation with `DataValidator`
- Null-rate threshold enforcement, bounds checking, duplicate detection
- Supports CSV and Excel (multi-format ingestion)

### Stage 2 — Feature Engineering (35 features)

#### Raw Features (29)
`age`, `total_orders`, `total_spend_usd`, `avg_order_value`, `return_rate_pct`, `churn_risk_score`, `preferred_device`, `prime_status` + 21 more

#### Engineered Features (6 new)
| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `rfm_composite` | 0.30·R + 0.35·F + 0.35·M | Overall customer value score |
| `loyalty_score` | age × orders / (days_since + 1) | Long-term engagement rate |
| `engagement_index` | reviews×2 + wishlist + categories×3 | Multi-channel engagement |
| `deal_sensitivity` | (discount% + coupons×2) / spend | Price consciousness index |
| `spend_per_browse_min` | spend / (browsing + 1) | Purchase efficiency |
| `category_diversity` | unique_cats / 10 | Shopping breadth (0-1) |

### Stage 3 — RFM Scoring

Each customer receives a **1–5 score** on three dimensions:

```
Recency  (R) = How recently did they purchase?       [weight: 0.30]
Frequency(F) = How often do they buy?                [weight: 0.35]
Monetary (M) = How much do they spend?               [weight: 0.35]

RFM Composite = 0.30·R + 0.35·F + 0.35·M  ∈ [1, 5]
```

### Stage 4 — Clustering

Three algorithms competed on **8 internal metrics**:

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-----------|:-----------:|:----------------:|:-------------------:|
| **KMeans ★** | **0.8328** | **0.4391** | **10,399** |
| Agglomerative | 0.8160 | 0.4525 | 10,293 |
| GMM | 0.1731 | 1.1217 | 3,539 |

> **Winner: KMeans** with k=5 (confirmed by both Elbow + Silhouette analysis)

**Stability Bootstrap** (20 random seeds): ARI mean = **0.94** → *Highly Stable*

### Stage 5 — Supervised Classification

| Model | CV F1 (5-fold) | Test F1 (Macro) |
|-------|:--------------:|:---------------:|
| **Random Forest ★** | **0.988 ± 0.003** | **0.988** |
| Gradient Boosting | 0.972 ± 0.005 | 0.971 |
| Logistic Regression | 0.934 ± 0.009 | 0.931 |

**Per-Class Performance (Random Forest):**
```
              precision  recall  f1-score  support
Bargain Hunter    1.00    1.00    1.00      199
Inactive User     0.99    1.00    1.00      209
Loyal Shopper     0.96    0.99    0.98      203
Occasional Buyer  1.00    0.99    0.99      188
Premium Spender   0.99    0.96    0.97      201
```

---

## 📊 Results & Insights

### Top Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `total_spend_usd` | 13.6% |
| 2 | `avg_daily_browsing_min` | 12.5% |
| 3 | `customer_lifetime_value` | 11.4% |
| 4 | `monetary_score` (RFM) | 9.8% |
| 5 | `wishlist_items` | 9.5% |
| 6 | `spend_per_browse_min` | 7.4% |
| 7 | `total_orders` | 6.6% |
| 8 | `deal_sensitivity` | 6.3% |

### Key Business Insights

- 💡 **Premium Spenders** (≈20% of base) drive **~48% of total revenue** → highest ROI for retention
- 💡 **Inactive Users** have churn risk **>0.72** — immediate win-back needed within 30 days
- 💡 **Bargain Hunters** browse **3× longer** than Premium Spenders but spend **10× less** — convert with flash sales
- 💡 **Prime membership** correlates with **2.3× higher LTV** across all segments
- 💡 **Mobile** is the dominant device (≈42%) — mobile-first campaigns are essential

### Visualisations

<table>
  <tr>
    <td><img src="reports/figures/fig01_segment_donut.png" width="300"/><br/><sub>Segment Distribution</sub></td>
    <td><img src="reports/figures/fig03_pca_2d.png" width="300"/><br/><sub>PCA 2D Embedding</sub></td>
    <td><img src="reports/figures/fig09_elbow_silhouette.png" width="300"/><br/><sub>Optimal k Selection</sub></td>
  </tr>
  <tr>
    <td><img src="reports/figures/fig04_feature_importance.png" width="300"/><br/><sub>Feature Importance</sub></td>
    <td><img src="reports/figures/fig07_radar_profile.png" width="300"/><br/><sub>Behavioural Radar</sub></td>
    <td><img src="reports/figures/fig06_correlation_heatmap.png" width="300"/><br/><sub>Correlation Heatmap</sub></td>
  </tr>
  <tr>
    <td><img src="reports/figures/fig02_rfm_distributions.png" width="300"/><br/><sub>RFM Distributions</sub></td>
    <td><img src="reports/figures/fig08_churn_boxplot.png" width="300"/><br/><sub>Churn Risk by Segment</sub></td>
    <td><img src="reports/figures/fig11_ltv_violin.png" width="300"/><br/><sub>LTV Distribution</sub></td>
  </tr>
</table>

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | Core language |
| Data | Pandas 2.0, NumPy | Data manipulation |
| ML | Scikit-learn 1.3 | Clustering, classification |
| Visualisation | Matplotlib, Seaborn, Plotly | Static + interactive charts |
| Dashboard | Streamlit 1.30 | Real-time web UI |
| Serialisation | Joblib | Model persistence |
| Config | PyYAML | YAML configuration |
| Testing | Pytest + pytest-cov | 18 unit tests |
| CI/CD | GitHub Actions | Automated test + lint |
| Quality | Black, Flake8, isort | Code standards |

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Aranya2801/AMAZON-User-Segmentation.git
cd AMAZON-User-Segmentation

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

```bash
python src/pipeline/data_pipeline.py
```

This will:
- ✅ Validate the dataset schema
- ✅ Engineer all 35 features (RFM + behavioral)
- ✅ Run 3 clustering algorithms, select the best
- ✅ Train 3 classifiers, select the best
- ✅ Save all model artefacts to `src/models/`
- ✅ Export enriched dataset to `data/processed/`
- ✅ Generate evaluation JSON to `reports/tables/`

### 3. Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### 4. Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 🖥️ Dashboard

The Streamlit dashboard provides **6 interactive pages** for daily use:

| Page | Description |
|------|-------------|
| 📊 **Executive Dashboard** | KPI cards, segment donut, revenue bars, correlation heatmap |
| 👥 **Segment Explorer** | Drill into any segment — age, device, radar profile |
| 🔬 **RFM Analysis** | 3D RFM scatter, composite scores, histogram distributions |
| 🤖 **ML Insights** | PCA projection, elbow curve, feature importances |
| 📈 **Churn Predictor** | Real-time gauge + personalised action recommendations |
| 🗃️ **Data Explorer** | Search, filter, download CSV with descriptive stats |

**Global Sidebar Filters:**
- 🌍 Region (7 regions)
- 💎 Prime / Non-Prime
- 🎂 Age range slider

---

## 📓 Notebooks

Place Jupyter notebooks in `notebooks/` for exploratory analysis:

```bash
jupyter notebook notebooks/
```

Suggested workflow:
1. `01_EDA.ipynb` — Exploratory data analysis
2. `02_Feature_Engineering.ipynb` — Feature derivation deep-dive
3. `03_Clustering.ipynb` — Algorithm comparison with visualisations
4. `04_Classification.ipynb` — Model training + SHAP explainability

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html

# Run a specific class
pytest tests/test_pipeline.py::TestClustering -v
```

**Test Coverage:**

| Module | Tests | Status |
|--------|-------|--------|
| DataValidator | 6 | ✅ All pass |
| Feature Engineering | 3 | ✅ All pass |
| Clustering | 3 | ✅ All pass |
| Classification | 2 | ✅ All pass |
| Utilities | 4 | ✅ All pass |
| **Total** | **18** | **✅ 100% passing** |

---

## 📂 Dataset

### Included Dataset — `amazon_customer_data.csv`

| Property | Value |
|----------|-------|
| Rows | 5,000 customers |
| Columns | 29 features |
| Format | CSV + Excel |
| Labels | 5 ground-truth segments |
| Regions | 7 global regions |
| Time span | 2015–2024 registrations |

### Feature Glossary

| Feature | Type | Description |
|---------|------|-------------|
| `customer_id` | str | Unique identifier `AMZ-XXXXXX` |
| `age` | int | Customer age (18–72) |
| `region` | str | Geographic region (7 values) |
| `prime_status` | str | Prime / Non-Prime |
| `total_orders` | int | Lifetime order count |
| `total_spend_usd` | float | Lifetime spend in USD |
| `avg_order_value` | float | Spend / orders |
| `return_rate_pct` | float | % orders returned |
| `churn_risk_score` | float | Predicted churn probability [0,1] |
| `customer_lifetime_value` | float | Estimated LTV in USD |
| `avg_daily_browsing_min` | int | Daily platform browsing time |
| `discount_usage_rate_pct` | float | % purchases using discounts |
| `rfm_composite` | float | Weighted RFM score [1,5] |
| `loyalty_score` | float | Derived loyalty metric |
| `engagement_index` | float | Multi-channel engagement |
| `cluster_id` | int | Assigned cluster (0–4) |

> **Want to use your own data?** Replace `data/raw/amazon_customer_data.csv` with any CSV that contains the required columns (see `DataValidator.REQUIRED_COLUMNS`) and re-run the pipeline.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your branch: `git checkout -b feature/amazing-feature`
3. Write tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Format code: `black src/ tests/ && isort src/ tests/`
6. Commit: `git commit -m 'feat: add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by [Aranya2801](https://github.com/Aranya2801)**

*If this project helped you, please ⭐ star the repository!*

[![GitHub stars](https://img.shields.io/github/stars/Aranya2801/AMAZON-User-Segmentation?style=social)](https://github.com/Aranya2801/AMAZON-User-Segmentation)

</div>
