"""Append an 'unrestricted N' control section to both benchmark notebooks."""
import json
import uuid

NB_BC  = "bench_mark/01_breast_cancer_baseline.ipynb"
NB_DB  = "bench_mark/02_diabetes_baseline.ipynb"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def shared_helper_cells(load_block: str, dataset_label: str, png_name: str) -> list:
    """Cells that work for both notebooks; only the data-loading line and labels differ."""
    return [
        md(
            "# Unrestricted sample size control\n"
            "\n"
            f"The 91-sample subset was chosen to mirror HIV. To rule out sample size as the "
            f"explanation for the high CVAE distinguishability AUC, we re-run the comparison on "
            f"the **full** {dataset_label} dataset, train a fresh CVAE on it, and use a standard "
            f"80/20 stratified split (no HIV-shaped capping). If CVAE still sits well above 0.5, "
            f"sample size is not the cause."
        ),
        code(load_block + "\n"
             "\n"
             "scaler_full = StandardScaler().fit(X_full)\n"
             "X_full_s = scaler_full.transform(X_full)\n"),
        code(
            "# Train a fresh CVAE on the full dataset\n"
            "torch.manual_seed(cfg.seed)\n"
            "np.random.seed(cfg.seed)\n"
            "\n"
            "X_full_t = torch.tensor(X_full_s, dtype=torch.float32)\n"
            "y_full_t = torch.tensor(y_full,   dtype=torch.long)\n"
            "\n"
            "train_loader_full = DataLoader(\n"
            "    TensorDataset(X_full_t, y_full_t),\n"
            "    batch_size=cfg.batch_size,\n"
            "    shuffle=True,\n"
            ")\n"
            "\n"
            "model_full = CVAE(x_dim=X_full_s.shape[1], c_dim=2, z_dim=cfg.z_dim, hidden=cfg.hidden).to(device)\n"
            "opt_full   = torch.optim.Adam(model_full.parameters(), lr=cfg.lr)\n"
            "\n"
            "for epoch in range(cfg.epochs):\n"
            "    model_full.train()\n"
            "    total = 0.0\n"
            "    for xb, yb in train_loader_full:\n"
            "        xb = xb.to(device)\n"
            "        yb = yb.to(device)\n"
            "        cb = F.one_hot(yb, num_classes=2).float()\n"
            "\n"
            "        x_hat, mu, logvar = model_full(xb, cb)\n"
            "        loss = elbo_loss(xb, x_hat, mu, logvar, beta=cfg.beta)\n"
            "\n"
            "        opt_full.zero_grad()\n"
            "        loss.backward()\n"
            "        opt_full.step()\n"
            "        total += loss.item() * xb.size(0)\n"
            "\n"
            "    if (epoch + 1) % 50 == 0:\n"
            "        print(f\"Epoch {epoch+1:3d} | loss={total/len(X_full_t):.4f}\")\n"
        ),
        code(
            "def unrestricted_run_auc(X_real, y_real, X_syn, y_syn, seed):\n"
            "    \"\"\"Standard 80/20 stratified split on all real+synthetic, RF AUC.\"\"\"\n"
            "    X_all = np.vstack([X_real, X_syn])\n"
            "    s_all = np.r_[np.zeros(len(X_real), dtype=int),\n"
            "                  np.ones(len(X_syn),  dtype=int)]\n"
            "    Xtr, Xte, str_, ste = train_test_split(\n"
            "        X_all, s_all, test_size=0.2, random_state=seed, stratify=s_all\n"
            "    )\n"
            "    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)\n"
            "    rf.fit(Xtr, str_)\n"
            "    p = rf.predict_proba(Xte)[:, 1]\n"
            "    return roc_auc_score(ste, p)\n"
            "\n"
            "def cvae_sampler_full(X, y, n0, n1, seed):\n"
            "    torch.manual_seed(seed)\n"
            "    Xs0 = sample_scaled(model_full, n0, y_label=0, device=device)\n"
            "    Xs1 = sample_scaled(model_full, n1, y_label=1, device=device)\n"
            "    Xs  = np.vstack([Xs0, Xs1])\n"
            "    ys  = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])\n"
            "    return Xs, ys\n"
            "\n"
            "GENERATORS_FULL = {\n"
            "    \"bootstrap\":  lambda X, y, n0, n1, seed: sample_bootstrap(X, y, n0, n1, seed=seed),\n"
            "    \"gmm\":        lambda X, y, n0, n1, seed: sample_gmm(X, y, n0, n1, seed=seed),\n"
            "    \"columnwise\": lambda X, y, n0, n1, seed: sample_columnwise(X, y, n0, n1, seed=seed),\n"
            "    \"cvae\":       cvae_sampler_full,\n"
            "}\n"
            "\n"
            "n0_full = int((y_full == 0).sum())\n"
            "n1_full = int((y_full == 1).sum())\n"
            "print(f\"full N={len(X_full)}  n0={n0_full}  n1={n1_full}\")\n"
        ),
        code(
            "R = 50\n"
            "results_full = {}\n"
            "\n"
            "for name, gen in GENERATORS_FULL.items():\n"
            "    aucs = []\n"
            "    for r in range(R):\n"
            "        Xs, ys = gen(X_full_s, y_full, n0_full, n1_full, seed=r)\n"
            "        auc = unrestricted_run_auc(X_full_s, y_full, Xs, ys, seed=r)\n"
            "        aucs.append(auc)\n"
            "    results_full[name] = np.array(aucs)\n"
            "    print(f\"{name:10s}  mean={results_full[name].mean():.3f}  std={results_full[name].std():.3f}\")\n"
        ),
        code(
            "methods_full = list(results_full.keys())\n"
            "data_box_full = [results_full[m] for m in methods_full]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "ax.boxplot(data_box_full, labels=methods_full, showmeans=True)\n"
            "ax.axhline(0.5, linestyle=\"--\", color=\"gray\", label=\"0.5 (random)\")\n"
            "ax.set_ylim(0.4, 1.05)\n"
            "ax.set_ylabel(\"RF probe AUC\")\n"
            f"ax.set_title(\"{dataset_label}: Distribution of AUC at full sample size\")\n"
            "ax.legend(loc=\"lower right\")\n"
            "plt.tight_layout()\n"
            f"plt.savefig(\"{png_name}\", dpi=300)\n"
            "plt.show()\n"
        ),
    ]


# ---------- Breast cancer ----------
bc_load = """\
from sklearn.datasets import load_breast_cancer

data_full = load_breast_cancer()
X_full = data_full.data
y_full = data_full.target.astype(int)
print("full:", X_full.shape, "class0:", (y_full == 0).sum(), "class1:", (y_full == 1).sum())"""

bc_cells = shared_helper_cells(bc_load, "Breast Cancer", "bc_full_method_auc_distribution.png")

with open(NB_BC, encoding="utf-8") as f:
    nb = json.load(f)
nb["cells"].extend(bc_cells)
with open(NB_BC, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"BC: now {len(nb['cells'])} cells")


# ---------- Diabetes ----------
db_load = """\
from sklearn.datasets import load_diabetes

data_full = load_diabetes()
X_full = data_full.data
y_full_cont = data_full.target
y_full = (y_full_cont > np.median(y_full_cont)).astype(int)
print("full:", X_full.shape, "class0:", (y_full == 0).sum(), "class1:", (y_full == 1).sum())"""

db_cells = shared_helper_cells(db_load, "Diabetes", "diabetes_full_method_auc_distribution.png")

with open(NB_DB, encoding="utf-8") as f:
    nb = json.load(f)
nb["cells"].extend(db_cells)
with open(NB_DB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Diabetes: now {len(nb['cells'])} cells")
