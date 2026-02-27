from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    seed: int = 42

    data_path: Path = Path("data/allSyntheticData.RData")
    output_path: Path = Path("data/output")

    test_size: float = 0.2

    # CVAE
    z_dim: int = 16
    hidden: int = 128
    beta: float = 0.5
    epochs: int = 500
    batch_size: int = 64
    lr: float = 1e-3
    decoder_noise: float = 0
    
    # Keys inside .RData
    x_key: str = "x"
    y_key: str = "y"
    x_transform: str = "log1p" # none | log1p
    def ensure_dirs(self) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)