import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from models.cvae import CVAE

def load_checkpoint(checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ckpt["cfg"]
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    x_dim = scaler_mean.shape[0]

    model = CVAE(
        x_dim=x_dim,
        c_dim=2,
        z_dim=int(cfg["z_dim"]),
        hidden=int(cfg["hidden"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, cfg, scaler_mean, scaler_scale


@torch.no_grad()
def sample_from_prior(
    model: CVAE,
    n: int,
    y_label: int,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    z = torch.randn(n, model.z_dim, device=device)
    c = F.one_hot(torch.full((n,), y_label, dtype=torch.long, device=device), num_classes=2).float()

    x_hat_scaled = model.decode(z, c)  # (n, D) in standardized space
    x_hat_scaled = x_hat_scaled.cpu().numpy()

    # invert standardization
    x_hat = x_hat_scaled * scaler_scale + scaler_mean
    return x_hat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_path = Path("data/output/cvae_best.pt")
    model, cfg, mean, scale = load_checkpoint(ckpt_path)

    n = 91
    X_syn_neg = sample_from_prior(model, n=n, y_label=0, scaler_mean=mean, scaler_scale=scale, device=device)
    X_syn_pos = sample_from_prior(model, n=n, y_label=1, scaler_mean=mean, scaler_scale=scale, device=device)

    out_dir = Path(cfg.get("output_path", "data/output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / f"cvae_synth_seed{cfg.get('seed', 0)}_y0.npz", X=X_syn_neg)
    np.savez(out_dir / f"cvae_synth_seed{cfg.get('seed', 0)}_y1.npz", X=X_syn_pos)
    print(" ", out_dir / f"cvae_synth_seed{cfg.get('seed', 0)}_y0.npz", X_syn_neg.shape)
    print(" ", out_dir / f"cvae_synth_seed{cfg.get('seed', 0)}_y1.npz", X_syn_pos.shape)


if __name__ == "__main__":
    main()