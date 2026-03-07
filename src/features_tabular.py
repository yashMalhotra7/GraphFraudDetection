import pandas as pd

def add_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Balance deltas
    out["orig_delta"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["dest_delta"] = out["newbalanceDest"] - out["oldbalanceDest"]

    # Ratios with +1 to avoid division by zero
    out["orig_balance_ratio"] = (out["newbalanceOrig"] + 1.0) / (out["oldbalanceOrg"] + 1.0)
    out["dest_balance_ratio"] = (out["newbalanceDest"] + 1.0) / (out["oldbalanceDest"] + 1.0)

    # One-hot for type (kept as separate columns)
    for t in ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]:
        out[f"type_{t.lower()}"] = (out["type"] == t).astype(int)

    return out
