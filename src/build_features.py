import pandas as pd
from src.config import Paths, Settings
from src.utils import ensure_dir
from src.build_graph import build_transaction_graph
from src.features_tabular import add_tabular_features
from src.features_graph import compute_graph_features, attach_node_features_to_transactions

def main():
    P = Paths()
    S = Settings()
    ensure_dir(P.data_processed)

    clean_path = P.data_processed / "paysim_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError("Run python -m src.make_dataset first.")

    df = pd.read_csv(clean_path)
    df_graph = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)


    # Add tabular features first
    df = add_tabular_features(df)

    # Build graph on full dataset
    G = build_transaction_graph(df_graph)

    # Compute graph features
    node_feats = compute_graph_features(
        G,
        pagerank_alpha=S.pagerank_alpha,
        betweenness_k=S.betweenness_k,
        seed=S.random_state,
    )

    # Attach node-level graph features onto each transaction
    df_feat = attach_node_features_to_transactions(df, node_feats)

    out_path = P.data_processed / "paysim_features.csv"
    df_feat.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(df_feat):,}  Cols: {df_feat.shape[1]}")

if __name__ == "__main__":
    main()

