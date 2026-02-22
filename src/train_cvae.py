from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from src.config import Config
from src.data import load_rdata_xy, make_loaders
from src.utils import set_seed

from models.cvae import CVAE


def elbo_loss(x, x_hat, mu, logvar, beta: float):
    # Recon: MSE sum over features, mean over batch
    recon = ((x_hat - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1.0 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)).mean()
    total = recon + beta * kl
    return total, recon, kl

@torch.no_grad()
def evaluate(model: CVAE, loader, device, beta: float) -> Dict[str, float]:
    model.eval()
    tot = rec = kl = 0.0
    n = 0
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        x_hat, mu, logvar = model(x, c)
        loss, r, k = elbo_loss(x, x_hat, mu, logvar, beta=beta)
        tot += loss.item()
        rec += r.item()
        kl += k.item()
        n += 1
    return {"loss": tot / n, "recon": rec / n, "kl": kl / n}


def main():
    cfg = Config()
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X, y = load_rdata_xy(cfg.data_path, x_key=cfg.x_key, y_key=cfg.y_key)

    train_loader, val_loader, scaler = make_loaders(
        X, y,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed
    )

    x_dim = X.shape[1]
    c_dim = 2

    model = CVAE(x_dim=x_dim, c_dim=c_dim, z_dim=cfg.z_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    best_path = cfg.output_path / "cvae_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot = rec = kl = 0.0
        n = 0

        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            opt.zero_grad(set_to_none=True)

            x_hat, mu, logvar = model(x, c)
            loss, r, k = elbo_loss(x, x_hat, mu, logvar, beta=cfg.beta)

            loss.backward()
            opt.step()

            tot += loss.item()
            rec += r.item()
            kl += k.item()
            n += 1

        train_metrics = {"loss": tot / n, "recon": rec / n, "kl": kl / n}
        val_metrics = evaluate(model, val_loader, device, beta=cfg.beta)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "cfg": cfg.__dict__,
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train loss={train_metrics['loss']:.4f} recon={train_metrics['recon']:.4f} kl={train_metrics['kl']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} recon={val_metrics['recon']:.4f} kl={val_metrics['kl']:.4f}"
            )

    print("didn't crash")


if __name__ == "__main__":
    main()