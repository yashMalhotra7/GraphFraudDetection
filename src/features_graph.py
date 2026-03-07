import pandas as pd
import networkx as nx

def compute_graph_features(G: nx.DiGraph, pagerank_alpha: float, betweenness_k: int, seed: int) -> pd.DataFrame:
    nodes = list(G.nodes())

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # PageRank only
    pagerank = nx.pagerank(G, alpha=pagerank_alpha)

    feats = pd.DataFrame({
        "node": nodes,
        "in_degree": [in_deg.get(n, 0) for n in nodes],
        "out_degree": [out_deg.get(n, 0) for n in nodes],
        "pagerank": [pagerank.get(n, 0.0) for n in nodes],
    })
    return feats


def attach_node_features_to_transactions(txn: pd.DataFrame, node_feats: pd.DataFrame) -> pd.DataFrame:
    nf = node_feats.set_index("node")

    out = txn.copy()
    for col in ["in_degree", "out_degree", "pagerank"]:
        out[f"orig_{col}"] = out["nameOrig"].map(nf[col]).fillna(0.0)
        out[f"dest_{col}"] = out["nameDest"].map(nf[col]).fillna(0.0)
    return out
