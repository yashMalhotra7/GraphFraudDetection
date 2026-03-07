import pandas as pd
import networkx as nx
from tqdm import tqdm

def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Directed graph: nameOrig -> nameDest
    Edge attributes aggregate repeated interactions:
      - weight: number of transactions
      - total_amount: sum of amount
    """
    G = nx.DiGraph()
    it = df[["nameOrig", "nameDest", "amount"]].itertuples(index=False, name=None)

    for u, v, amt in tqdm(it, total=len(df), desc="Building graph"):
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
            G[u][v]["total_amount"] += float(amt)
        else:
            G.add_edge(u, v, weight=1, total_amount=float(amt))

    return G
