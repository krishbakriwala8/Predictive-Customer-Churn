# Predictive Customer Churn & Retention Analytics (Germany-focused)

Description
- End-to-end churn prediction project using the UCI "Online Retail II" dataset filtered for customers in Germany.
- Builds cohort and behavioral features (RFM + transactional behavior), applies Recursive Feature Elimination (RFE), compares Logistic Regression, Random Forest, and XGBoost with GridSearch + 5-fold CV, exports model and aggregated data for Power BI.

Project structure
- churn_project.py            — main end-to-end script
- sql/feature_queries.sql     — SQL queries (DuckDB) that match the feature engineering in the script
- requirements.txt            — Python dependencies
- notebooks/                  — Jupyter notebooks (analysis + visualizations)
- data/raw/                   — place the dataset here (instructions below)
- outputs/                    — model artifacts, feature CSVs, and evaluation reports

Data
- Source: UCI Online Retail II (real transactional data). Download the Excel from:
  https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
- Save the file as `data/raw/online_retail_ii.xlsx` before running.

High-level pipeline
1. Ingest Excel, concat sheets, filter Country == 'Germany'
2. Clean (remove cancellations, negative quantities)
3. Build features (observation window / future window), label churn as no-purchase in future window
4. Feature selection via RFE (Logistic Regression base)
5. Train Logistic Regression, RandomForest, XGBoost with GridSearchCV (5-fold stratified CV)
6. Evaluate (AUC, precision@k, lift), save best model and Power BI CSVs

Quick start (Linux / macOS / WSL)
1. Clone repo
2. Create virtualenv:
   python -m venv venv
   source venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Place data:
   mkdir -p data/raw
   # Download online_retail_ii.xlsx from UCI and put into data/raw/
5. Run:
   python churn_project.py --data data/raw/online_retail_ii.xlsx --out outputs --future-window-days 180

Outputs
- outputs/features_for_model.csv
- outputs/powerbi_aggregates.csv
- outputs/best_model.joblib
- outputs/model_report.txt

Notes & academic rigor
- Uses reproducible seeds, stratified CV, and clear observation/future windows.
- Churn label is realistic for retail: no activity within a future window (configurable).
- Suitable for a master-level portfolio: you can highlight dataset, methods (RFE, CV), and concrete business outcomes.

If you want, I can push additional files (notebook, screenshots) or run the pipeline and add sample outputs. Provide the dataset in `data/raw/online_retail_ii.xlsx` or grant dataset upload in the repo.
