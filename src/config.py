from dataclasses import dataclass 
from pathlib import Path
@dataclass
class config:
    seed: int = 42
    data_path: Path = "data/allSyntheticData.RData"
    output_path: Path = "data/output"
    test_size: float = 0.2

    #CVAE
    z_dim: int = 16
    hidden: int = 128
    beta: float = 0.5
    epochs: int = 200
    batch_size: int = 64