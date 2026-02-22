from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Repro
    seed: int = 42

    # Paths
    data_path: Path = Path("data/allSyntheticData.RData")
    output_path: Path = Path("data/output")

    # Split
    test_size: float = 0.2

    # CVAE
    z_dim: int = 16
    hidden: int = 128
    beta: float = 0.5
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3

    # Keys inside .RData
    x_key: str = "x"
    y_key: str = "y"

    def ensure_dirs(self) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)