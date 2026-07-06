"""churn_project.py

End-to-end churn prediction pipeline focused on German customers from the UCI Online Retail II dataset.

Usage:
    python churn_project.py --out outputs --future-window-days 180

If no local dataset is present at data/raw/online_retail_ii.xlsx the script will attempt to download it
from the UCI repository automatically.
"""

import argparse
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.feature_selection import RFE
import joblib
import requests
from io import BytesIO

RANDOM_STATE = 42
LOGGER = logging.getLogger("churn_project")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

UCI_ONLINE_RETAIL_II_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
)


def download_dataset(dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Attempting to download dataset from UCI: %s", UCI_ONLINE_RETAIL_II_URL)
    try:
        r = requests.get(UCI_ONLINE_RETAIL_II_URL, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        LOGGER.info("Downloaded dataset to %s", dest_path)
        return dest_path
    except Exception as e:
        LOGGER.error("Failed to download dataset: %s", e)
        raise


def load_data(excel_path: str) -> pd.DataFrame:
    LOGGER.info("Loading Excel file: %s", excel_path)
    sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    LOGGER.info("Loaded %d rows across %d sheets", df.shape[0], len(sheets))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df = df.rename(columns=lambda c: str(c).strip().replace(" ", "_").replace("#", "Number"))

    # Remove cancellations: Invoice numbers that begin with 'C' or negative quantities
    if 'Invoice' in df.columns:
        df = df[~df['Invoice'].astype(str).str.startswith('C', na=False)]

    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]

    # Drop rows without CustomerID or InvoiceDate
    if 'CustomerID' in df.columns and 'InvoiceDate' in df.columns:
        df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    else:
        raise ValueError("Expected columns CustomerID and InvoiceDate in the dataset.")

    # Convert types
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    if 'UnitPrice' in df.columns:
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0.0)
    else:
        df['UnitPrice'] = 0.0

    # Filter Germany (case-insensitive)
    if 'Country' in df.columns:
        df = df[df['Country'].str.strip().str.lower() == 'germany']
    else:
        raise ValueError("Expected Country column in dataset.")

    LOGGER.info("After cleaning and filtering Germany: %d rows", df.shape[0])
    return df


def create_observation_and_label(df: pd.DataFrame, future_window_days: int = 180):
    # Observation end is snapshot_date - future_window
    snapshot_date = df['InvoiceDate'].max()
    future_window = pd.Timedelta(days=future_window_days)
    observation_end = snapshot_date - future_window
    LOGGER.info("Snapshot date: %s, observation_end: %s, future_window_days: %d",
                snapshot_date, observation_end, future_window_days)

    training_df = df[df['InvoiceDate'] <= observation_end].copy()
    future_df = df[df['InvoiceDate'] > observation_end].copy()

    customers = training_df['CustomerID'].unique()
    LOGGER.info("Customers in observation window: %d", len(customers))

    future_active = set(future_df['CustomerID'].unique())
    labels = pd.Series({
        cust: (0 if cust in future_active else 1)
        for cust in customers
    })
    labels = labels.rename('churn').astype(int)

    return training_df, labels, observation_end


def build_features(trans_df: pd.DataFrame, observation_end: pd.Timestamp) -> pd.DataFrame:
    trans_df = trans_df.copy()
    trans_df['TotalPrice'] = trans_df['Quantity'] * trans_df['UnitPrice']

    agg = trans_df.groupby('CustomerID').agg(
        first_purchase=('InvoiceDate', 'min'),
        last_purchase=('InvoiceDate', 'max'),
        num_invoices=('Invoice', pd.Series.nunique),
        total_quantity=('Quantity', 'sum'),
        monetary=('TotalPrice', 'sum'),
        distinct_items=('StockCode', pd.Series.nunique)
    ).reset_index()

    agg['recency_days'] = (observation_end - agg['last_purchase']).dt.days
    agg['tenure_days'] = (agg['last_purchase'] - agg['first_purchase']).dt.days.clip(lower=0)
    agg['avg_basket_value'] = agg['monetary'] / agg['num_invoices'].replace(0, np.nan)
    agg['avg_quantity_per_invoice'] = agg['total_quantity'] / agg['num_invoices'].replace(0, np.nan)

    # Average days between purchases per customer
    def avg_days_between(group):
        dates = group.sort_values('InvoiceDate')['InvoiceDate'].drop_duplicates()
        if len(dates) <= 1:
            return np.nan
        diffs = dates.diff().dt.days.dropna()
        return diffs.mean()

    avg_days = trans_df.groupby('CustomerID').apply(avg_days_between).rename('avg_days_between').reset_index()
    cust_agg = agg.merge(avg_days, on='CustomerID', how='left')

    cust_agg['avg_days_between'] = cust_agg['avg_days_between'].fillna(cust_agg['tenure_days'].replace(0, np.nan))
    cust_agg['frequency_per_30d'] = cust_agg['num_invoices'] / (cust_agg['tenure_days'].replace(0, 1) / 30.0)

    # Replace infs and NaNs
    cust_agg = cust_agg.replace([np.inf, -np.inf], np.nan).fillna(0)

    cust_agg = cust_agg.set_index('CustomerID')
    LOGGER.info("Built features for %d customers", cust_agg.shape[0])
    return cust_agg


def prepare_feature_label_table(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    df = features.join(labels, how='inner')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    LOGGER.info("Final dataset shape (features+label): %s", df.shape)
    return df


def feature_selection_rfe(X: pd.DataFrame, y: pd.Series, n_select: int = None):
    if n_select is None:
        n_select = max(3, int(X.shape[1] / 2))
    estimator = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='liblinear')
    rfe = RFE(estimator, n_features_to_select=n_select, step=1)
    rfe.fit(X, y)
    mask = rfe.support_
    selected_features = X.columns[mask].tolist()
    LOGGER.info("RFE selected %d features: %s", len(selected_features), selected_features)
    return selected_features, rfe


def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    LOGGER.info("AUC: %.4f, Precision: %.4f, Recall: %.4f", auc, prec, rec)
    return {'auc': auc, 'precision': prec, 'recall': rec, 'probs': probs, 'preds': preds}


def precision_at_k(y_true, y_scores, k=0.1):
    df = pd.DataFrame({'y_true': y_true.values, 'y_score': y_scores})
    cutoff = int(np.ceil(len(df) * k))
    topk = df.sort_values('y_score', ascending=False).head(cutoff)
    if cutoff == 0:
        return 0.0
    return topk['y_true'].mean()


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path('data/raw/online_retail_ii.xlsx')
    if not data_path.exists():
        download_dataset(data_path)

    df_raw = load_data(data_path)
    df = clean_data(df_raw)
    train_trans, labels, observation_end = create_observation_and_label(df, future_window_days=args.future_window_days)
    features = build_features(train_trans, observation_end)

    data = prepare_feature_label_table(features, labels)

    features_for_model = data.copy()
    features_for_model.to_csv(out_dir / "features_for_model.csv")
    LOGGER.info("Saved features_for_model.csv")

    X = data.drop(columns=['churn'])
    y = data['churn']

    selected_features, rfe_obj = feature_selection_rfe(X, y, n_select=args.n_features_to_select)
    X_sel = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    LOGGER.info("Train/test split: %d / %d", X_train.shape[0], X_test.shape[0])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'logreg': (LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='liblinear'),
                   {'C': [0.01, 0.1, 1, 10]}),
        'rf': (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
               {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}),
        'xgb': (XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
                {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]})
    }

    best_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, (est, grid) in models.items():
        LOGGER.info("Grid searching %s", name)
        # Use scaled data for all models
        grid_search = GridSearchCV(est, grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, y_train)
        best = grid_search.best_estimator_
        LOGGER.info("%s best params: %s, best score (cv auc): %.4f", name, grid_search.best_params_, grid_search.best_score_)
        eval_res = evaluate_model(best, X_test_scaled, y_test)
        eval_res['model'] = best
        eval_res['params'] = grid_search.best_params_
        best_models[name] = eval_res

    best_name = max(best_models.items(), key=lambda kv: kv[1]['auc'])[0]
    LOGGER.info("Best model on test AUC: %s (AUC: %.4f)", best_name, best_models[best_name]['auc'])
    best_model = best_models[best_name]['model']

    joblib.dump({'model': best_model, 'scaler': scaler, 'rfe': rfe_obj, 'selected_features': selected_features},
                out_dir / "best_model.joblib")
    LOGGER.info("Saved best_model.joblib")

    with open(out_dir / "model_report.txt", 'w') as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"AUC: {best_models[best_name]['auc']:.4f}\n")
        f.write(f"Precision: {best_models[best_name]['precision']:.4f}\n")
        f.write(f"Recall: {best_models[best_name]['recall']:.4f}\n")
        f.write(f"Params: {best_models[best_name]['params']}\n")
        f.write("\nAll models summary:\n")
        for k, v in best_models.items():
            f.write(f"{k}: AUC={v['auc']:.4f}, Precision={v['precision']:.4f}, Recall={v['recall']:.4f}\n")
    LOGGER.info("Saved model_report.txt")

    y_test_probs = best_models[best_name]['probs']
    p_at_10 = precision_at_k(y_test, y_test_probs, k=0.1)
    LOGGER.info("Precision@10%% (test): %.4f", p_at_10)
    with open(out_dir / "model_report.txt", 'a') as f:
        f.write(f"\nPrecision@10% (test): {p_at_10:.4f}\n")

    pipeline_scaler = scaler
    X_all_scaled = pipeline_scaler.transform(X_sel)
    all_probs = best_model.predict_proba(X_all_scaled)[:, 1]
    export_df = X_sel.copy()
    export_df['churn'] = y
    export_df['churn_prob'] = all_probs
    export_df.to_csv(out_dir / "powerbi_aggregates.csv")
    LOGGER.info("Saved powerbi_aggregates.csv (connect to Power BI Desktop)")

    LOGGER.info("Done. Outputs in %s", out_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--future-window-days", type=int, default=180, help="Future window length to label churn")
    parser.add_argument("--n-features-to-select", dest="n_features_to_select", type=int, default=None)
    args = parser.parse_args()
    main(args)
