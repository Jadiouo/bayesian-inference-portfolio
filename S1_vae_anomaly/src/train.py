"""
S1 · VAE 訓練迴圈（含 β-VAE 的 β 權重）。

β-VAE 的目標是 $\\mathbb{E}_q[\\log p(x|z)] - \\beta\\,\\mathrm{KL}(q\\|p)$。

- $\\beta=1$ 是標準 ELBO（唯一是 $\\log p(x)$ 合法下界的情形）
- $\\beta<1$ 偏重重建 → latent 用得多、KL 大
- $\\beta>1$ 偏重壓縮 → 容易 **posterior collapse**：KL→0、latent 被忽略、
  decoder 退化成無條件生成器

對異常偵測而言 collapse 是致命的：latent 不帶資訊時，ELBO 就只反映
「這張圖好不好重建」，完全沒有「這個 latent 位置有多不尋常」的成分。
β 掃描把這個 trade-off 量出來（見 experiments.py）。

⚠️ **只有 β=1 的模型能宣稱在算 ELBO。** β≠1 時目標函數不再是
$\\log p(x)$ 的下界，報告時不該叫它 ELBO —— 本專案在 β 掃描裡
一律稱「training objective」，只有主模型談 ELBO。
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from vae import ConvVAE, count_params, device, elbo_terms, kl_diag_gaussian, \
    recon_log_prob, set_seed


def make_loader(s, batch_size: int = 128, shuffle: bool = True, seed: int | None = None):
    x = torch.from_numpy(s.x)
    g = None
    if shuffle and seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=shuffle,
                      generator=g, num_workers=0,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate_elbo(model, s, dev, batch_size: int = 256, n_samples: int = 1) -> dict:
    """在一組影像上算平均 ELBO / 重建 / KL。"""
    model.eval()
    tot = {"elbo": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    for i in range(0, s.n, batch_size):
        xb = torch.from_numpy(s.x[i:i + batch_size]).to(dev)
        t = elbo_terms(model, xb, n_samples=n_samples)
        for k in tot:
            tot[k] += float(t[k].sum())
        n += len(xb)
    return {k: v / n for k, v in tot.items()}


def train(train_set, val_set, latent_dim: int = 16, beta: float = 1.0,
          epochs: int = 300, lr: float = 1e-3, weight_decay: float = 0.0,
          batch_size: int = 128, width: int = 32, seed: int = 0,
          patience: int = 40, verbose: bool = False, warmup_epochs: int = 0):
    """訓練一個 VAE，以 val ELBO 選最佳權重。

    `warmup_epochs > 0` 時對 KL 項做線性 annealing（0 → beta）。
    這是對抗 posterior collapse 的標準手段：訓練初期 decoder 還很爛，
    KL 項會把 q 直接壓成先驗（那是當時的最優解），之後就再也爬不出來。
    本專案的主模型**不用** warmup（`warmup_epochs=0`），
    好讓 collapse 診斷測到的是模型本身的行為，而不是被 annealing 掩蓋的。
    """
    dev = device()
    set_seed(seed)
    model = ConvVAE(latent_dim=latent_dim, width=width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = make_loader(train_set, batch_size, shuffle=True, seed=seed)

    best = {"elbo": -np.inf, "state": None, "epoch": -1}
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        beta_ep = beta if warmup_epochs == 0 else beta * min(1.0, (ep + 1) / warmup_epochs)
        run = {"recon": 0.0, "kl": 0.0}
        n = 0
        for (xb,) in loader:
            xb = xb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            mu, logvar = model.encode(xb)
            z = model.reparameterize(mu, logvar)
            recon = recon_log_prob(model.decode(z), xb)
            kl = kl_diag_gaussian(mu, logvar)
            loss = -(recon - beta_ep * kl).mean()
            loss.backward()
            opt.step()
            run["recon"] += float(recon.detach().sum())
            run["kl"] += float(kl.detach().sum())
            n += len(xb)
        sched.step()

        va = evaluate_elbo(model, val_set, dev)
        hist.append({"epoch": ep, "beta_used": beta_ep,
                     "train_recon": run["recon"] / n, "train_kl": run["kl"] / n,
                     "val_elbo": va["elbo"], "val_recon": va["recon"], "val_kl": va["kl"]})
        if va["elbo"] > best["elbo"]:
            best = {"elbo": va["elbo"], "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if verbose and ep % 25 == 0:
            print(f"  ep{ep:03d} train_recon={run['recon'] / n:8.2f} "
                  f"train_kl={run['kl'] / n:7.3f} val_elbo={va['elbo']:8.2f}")
        if ep - best["epoch"] >= patience:
            break

    model.load_state_dict(best["state"])
    model.eval()
    info = {"history": hist, "best_epoch": best["epoch"], "best_val_elbo": best["elbo"],
            "seconds": round(time.time() - t0, 1), "n_params": count_params(model),
            "epochs_run": len(hist), "beta": beta, "latent_dim": latent_dim,
            "warmup_epochs": warmup_epochs, "seed": seed}
    return model, info


def active_units(model, s, dev, threshold: float = 0.01, batch_size: int = 256) -> dict:
    """**posterior collapse 的診斷**：每個 latent 維度的平均 KL。

    塌陷的維度 KL≈0 —— encoder 對它輸出先驗、decoder 也就學會忽略它。
    `threshold` 是「算活著」的最小平均 KL（nats）；0.01 是文獻常用的量級。

    另外報 **AU (active units)** 的另一種定義：latent 的邊際變異數
    $\\mathrm{Var}_x(\\mu(x))$。兩個定義抓的東西略不同 ——
    KL 看「單筆資料把 q 推離先驗多遠」，變異數看「不同資料的 q 中心是否分散」。
    一個維度可以 KL 大但 μ 不分散（每筆都推向同一處，等於常數偏移，
    對 decoder 沒有資訊）。同時報兩者才不會被單一指標騙。
    """
    model.eval()
    kls, mus = [], []
    with torch.no_grad():
        for i in range(0, s.n, batch_size):
            xb = torch.from_numpy(s.x[i:i + batch_size]).to(dev)
            mu, logvar = model.encode(xb)
            kls.append(kl_diag_gaussian(mu, logvar, per_dim=True).cpu())
            mus.append(mu.cpu())
    kl_per_dim = torch.cat(kls).mean(0).numpy()
    mu_all = torch.cat(mus).numpy()
    mu_var = mu_all.var(axis=0)
    return {
        "kl_per_dim": kl_per_dim,
        "mu_variance_per_dim": mu_var,
        "n_active_by_kl": int((kl_per_dim > threshold).sum()),
        "n_active_by_mu_var": int((mu_var > threshold).sum()),
        "latent_dim": int(len(kl_per_dim)),
        "threshold": threshold,
        "frac_active_by_kl": float((kl_per_dim > threshold).mean()),
        "total_kl": float(kl_per_dim.sum()),
    }
