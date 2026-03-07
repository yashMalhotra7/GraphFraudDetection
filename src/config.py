from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    models: Path = root / "models"
    outputs: Path = root / "outputs"
    reports: Path = root / "reports"

@dataclass(frozen=True)
class Settings:
    random_state: int = 42
    test_size: float = 0.2

    # FULL PaySim
    max_rows: int | None = None  # set to None for full, or int for dev

    # Graph feature computation controls
    pagerank_alpha: float = 0.85

    # Betweenness centrality approximation: computes betweenness by sampling k nodes
    betweenness_k: int = 2000

    # SHAP sampling (do not run SHAP on all rows)
    shap_sample_rows: int = 5000

    # Model threshold for hard labels
    decision_threshold: float = 0.5
