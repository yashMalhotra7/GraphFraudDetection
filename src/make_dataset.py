import pandas as pd
from src.config import Paths, Settings
from src.utils import ensure_dir

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    expected = [
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[expected].copy()
    df["isFraud"] = df["isFraud"].astype(int)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(int)

    # Minimal sanity
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    return df

def main():
    P = Paths()
    S = Settings()
    ensure_dir(P.data_processed)

    raw_csv = P.data_raw / "PS_20174392719_1491204439457_log.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at: {raw_csv}\n"
            "Place the downloaded file in data/raw/ and rename it to PS_20174392719_1491204439457_log.csv"
        )

    # Load full dataset
    df = pd.read_csv(raw_csv)

    # Optional downsample for dev only
    if S.max_rows is not None and len(df) > S.max_rows:
        df = df.sample(S.max_rows, random_state=S.random_state).reset_index(drop=True)

    df = basic_clean(df)

    out_path = P.data_processed / "paysim_clean.csv"
    df.to_csv(out_path, index=False)

    fraud_rate = df["isFraud"].mean()
    print(f"Saved: {out_path}")
    print(f"Rows: {len(df):,}")
    print(f"Fraud rate (isFraud): {fraud_rate:.6f}")

if __name__ == "__main__":
    main()
