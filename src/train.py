# File: src/train.py
# Purpose:
# - Train baseline (tabular-only) and graph-enhanced (tabular + graph) XGBoost models
# - Evaluate at default threshold (0.5)
# - Tune the classification threshold to maximize F1 (and also report best threshold by precision@min_recall)

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from xgboost import XGBClassifier

from src.config import Paths, Settings
from src.utils import ensure_dir, save_json


def metrics_from_scores(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(auc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """
    Finds the threshold in [0.01, 0.99] that maximizes F1.
    """
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)}
    return best


def best_threshold_by_precision_at_min_recall(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_recall: float = 0.98
) -> dict:
    """
    Picks the threshold that gives the highest precision subject to recall >= min_recall.
    Useful for fraud: keep recall very high, improve precision (reduce false alarms).
    """
    best = None
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if r >= min_recall:
            cand = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)}
            if best is None or cand["precision"] > best["precision"]:
                best = cand

    # If nothing satisfies the recall constraint, fall back to best F1
    if best is None:
        best = best_threshold_by_f1(y_true, proba)
        best["note"] = f"No threshold met recall >= {min_recall:.2f}. Falling back to best F1."
    else:
        best["note"] = f"Best precision with recall >= {min_recall:.2f}"
    return best


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> XGBClassifier:
    """
    Trains an XGBoost model with imbalance handling via scale_pos_weight.
    """
    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    spw = neg / pos

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=spw,
    )
    model.fit(X_train, y_train)
    return model


def prepare_xy(df: pd.DataFrame, use_graph_features: bool) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds X and y. Drops identifiers and target columns.
    Baseline: drops graph-derived columns (orig_*, dest_*)
    Graph-enhanced: keeps them.
    """
    y = df["isFraud"].astype(int)

    drop_cols = ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest", "type"]
    X = df.drop(columns=drop_cols)

    if not use_graph_features:
        graph_cols = [c for c in X.columns if c.startswith("orig_") or c.startswith("dest_")]
        X = X.drop(columns=graph_cols)

    return X, y


def evaluate_with_threshold_tuning(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    default_threshold: float
) -> dict:
    """
    Returns:
    - default metrics at default_threshold
    - best threshold by F1
    - best threshold by precision at min recall (0.98)
    """
    proba = model.predict_proba(X_test)[:, 1]

    default_metrics = metrics_from_scores(y_test.values, proba, threshold=default_threshold)
    best_f1 = best_threshold_by_f1(y_test.values, proba)
    best_prec_r98 = best_threshold_by_precision_at_min_recall(y_test.values, proba, min_recall=0.98)

    return {
        "default": default_metrics,
        "best_f1": best_f1,
        "best_precision_at_recall_0_98": best_prec_r98,
    }


def main():
    P = Paths()
    S = Settings()

    ensure_dir(P.models)
    ensure_dir(P.outputs / "metrics")
    ensure_dir(P.outputs / "predictions")

    feat_path = P.data_processed / "paysim_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("Run python -m src.build_features first to create paysim_features.csv")

    df = pd.read_csv(feat_path)

    # -------------------------
    # 1) BASELINE (tabular only)
    # -------------------------
    X_base, y = prepare_xy(df, use_graph_features=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X_base, y,
        test_size=S.test_size,
        random_state=S.random_state,
        stratify=y
    )

    baseline_model = train_xgb(X_train, y_train, S.random_state)
    baseline_eval = evaluate_with_threshold_tuning(
        baseline_model, X_test, y_test, default_threshold=S.decision_threshold
    )

    dump(baseline_model, P.models / "baseline_model.pkl")
    save_json(baseline_eval, P.outputs / "metrics" / "baseline.json")

    # -------------------------------
    # 2) GRAPH-ENHANCED (tabular+graph)
    # -------------------------------
    X_graph, y2 = prepare_xy(df, use_graph_features=True)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_graph, y2,
        test_size=S.test_size,
        random_state=S.random_state,
        stratify=y2
    )

    graph_model = train_xgb(X_train2, y_train2, S.random_state)
    graph_eval = evaluate_with_threshold_tuning(
        graph_model, X_test2, y_test2, default_threshold=S.decision_threshold
    )

    dump(graph_model, P.models / "graph_model.pkl")
    save_json(graph_eval, P.outputs / "metrics" / "graph_enhanced.json")

    # Save holdout predictions for graph model using the BEST F1 threshold (for error analysis)
    graph_proba = graph_model.predict_proba(X_test2)[:, 1]
    t_best = graph_eval["best_f1"]["threshold"]
    preds = pd.DataFrame({
        "y_true": y_test2.values,
        "y_proba": graph_proba,
        "y_pred_default_0_5": (graph_proba >= S.decision_threshold).astype(int),
        "y_pred_best_f1": (graph_proba >= t_best).astype(int),
    })
    preds.to_csv(P.outputs / "predictions" / "holdout_preds_graph.csv", index=False)

    print("Saved models and metrics (with threshold tuning).")
    print("\nBASELINE (tabular) evaluation:")
    print("  Default @0.5:", baseline_eval["default"])
    print("  Best F1:", baseline_eval["best_f1"])
    print("  Best precision with recall>=0.98:", baseline_eval["best_precision_at_recall_0_98"])

    print("\nGRAPH-ENHANCED evaluation:")
    print("  Default @0.5:", graph_eval["default"])
    print("  Best F1:", graph_eval["best_f1"])
    print("  Best precision with recall>=0.98:", graph_eval["best_precision_at_recall_0_98"])


if __name__ == "__main__": 
    main()
