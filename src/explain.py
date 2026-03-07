import pandas as pd
import shap
from joblib import load
from src.config import Paths, Settings
from src.utils import ensure_dir

def get_X(df: pd.DataFrame, use_graph_features: bool) -> pd.DataFrame:
    drop_cols = ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest", "type"]
    X = df.drop(columns=drop_cols)

    if not use_graph_features:
        graph_cols = [c for c in X.columns if c.startswith("orig_") or c.startswith("dest_")]
        X = X.drop(columns=graph_cols)

    return X

def save_shap_summary(model, X_sample: pd.DataFrame, out_png):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    P = Paths()
    S = Settings()
    ensure_dir(P.reports / "figures")

    feat_path = P.data_processed / "paysim_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("Run python -m src.build_features first.")

    df = pd.read_csv(feat_path)

    # Sample rows for SHAP
    n = min(S.shap_sample_rows, len(df))
    df_sample = df.sample(n, random_state=S.random_state).reset_index(drop=True)

    baseline_model = load(P.models / "baseline_model.pkl")
    graph_model = load(P.models / "graph_model.pkl")

    X_base = get_X(df_sample, use_graph_features=False)
    X_graph = get_X(df_sample, use_graph_features=True)

    save_shap_summary(baseline_model, X_base, P.reports / "figures" / "shap_summary_baseline.png")
    save_shap_summary(graph_model, X_graph, P.reports / "figures" / "shap_summary_graph.png")

    print("Saved SHAP plots to reports/figures/")

if __name__ == "__main__":
    main()
