from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from models.cvae import CVAE
from src.config import Config
from src.transformation import make_transform


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Rebuild from saved dictionary
    cfg = Config(**ckpt["cfg"])

    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    x_dim = scaler_mean.shape[0]

    model = CVAE(
        x_dim=x_dim,
        c_dim=2,
        z_dim=cfg.z_dim,
        hidden=cfg.hidden,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = make_transform(cfg.x_transform)

    return model, cfg, scaler_mean, scaler_scale, transform


@torch.no_grad()
def sample_class(model, n, y_label, scaler_mean, scaler_scale, transform, device):
    model = model.to(device)
    model.eval()

    z = torch.randn(n, model.z_dim, device=device)
    c = F.one_hot(
        torch.full((n,), y_label, dtype=torch.long, device=device),
        num_classes=2
    ).float()

    # decode in standardized space
    x_scaled = model.decode(z, c).cpu().numpy()

    # inverse standardization -> transformed space
    x_t = x_scaled * scaler_scale + scaler_mean

    # inverse transform with logp1
    x = transform.inverse(x_t)

    return x.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_path = Path("data/output/cvae_best.pt")
    model, cfg, mean, scale, transform = load_checkpoint(ckpt_path)

    cfg.ensure_dirs()

    n = 91

    X_y0 = sample_class(model, n, 0, mean, scale, transform, device)
    X_y1 = sample_class(model, n, 1, mean, scale, transform, device)

    tag = f"{cfg.x_transform}"
    out0 = cfg.output_path / f"cvae_synth_seed{cfg.seed}_y0_{tag}.npz"
    out1 = cfg.output_path / f"cvae_synth_seed{cfg.seed}_y1_{tag}.npz"

    np.savez(out0, X=X_y0)
    np.savez(out1, X=X_y1)

    print("Saved:")
    print(" ", out0, X_y0.shape)
    print(" ", out1, X_y1.shape)


if __name__ == "__main__":
    main()