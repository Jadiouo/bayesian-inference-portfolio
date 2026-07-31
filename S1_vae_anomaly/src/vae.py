"""
S1 · 變分自編碼器 —— 主題五(變分推論 / ELBO)的完整落地。

ELBO 是什麼
-----------
我們想要 $\\log p(x)$，但它需要對 $z$ 積分，算不出來。變分推論改為
引入一個可算的 $q(z|x)$ 並最大化下界：

$$\\log p(x) = \\underbrace{\\mathbb{E}_{q}\\big[\\log p(x|z)\\big] - \\mathrm{KL}\\big(q(z|x)\\,\\|\\,p(z)\\big)}_{\\text{ELBO}}
\\;+\\; \\underbrace{\\mathrm{KL}\\big(q(z|x)\\,\\|\\,p(z|x)\\big)}_{\\ge 0,\\ \\text{the gap}}$$

三件事從這條式子直接讀出來，而且都是本專案要驗證的：

1. **ELBO 永遠是 $\\log p(x)$ 的下界**，差距正好是「近似後驗與真後驗的 KL」。
   所以 ELBO 低有兩種可能：資料真的不可能（$\\log p(x)$ 低），
   或我的近似很差（gap 大）。用 ELBO 當異常分數時，這兩者被混在一起。
2. **兩項有不同的語意**：重建項問「解碼得回來嗎」，
   KL 項問「latent 跑到先驗外了嗎」。異常可以只觸發其中一個。
3. **收緊下界會改變分數**。`iwae_bound` 用 importance weighting 讓下界更緊，
   本專案實測它對 AUROC 的影響（見 experiments.py）。

重參數化技巧
------------
ELBO 的梯度要穿過「從 $q$ 抽樣」這個隨機步驟。直接對抽樣微分是不行的，
所以把隨機性搬到與參數無關的地方：

$$z = \\mu_\\phi(x) + \\sigma_\\phi(x)\\odot\\epsilon,\\qquad \\epsilon\\sim\\mathcal N(0,I)$$

現在 $z$ 對 $\\phi$ 可微，梯度可以直接反向傳。替代方案是 REINFORCE
(score-function estimator)，它不需要可微但**變異數大得多** ——
`grad_variance_comparison` 把這個差距量出來，那是「為什麼需要重參數化」
的直接證據，而不只是引用。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2.0 * math.pi)


class ConvVAE(nn.Module):
    """28×28 灰階影像的卷積 VAE。

    decoder 輸出 Bernoulli 的 logits。MedMNIST 的像素是 [0,1] 的連續值
    （不是二元），嚴格來說應該用連續 Bernoulli 或截斷高斯；
    用 Bernoulli 交叉熵是 VAE 文獻的慣例（原始論文在 MNIST 上就這樣做），
    它等價於一個未正規化的似然。這個近似的後果記在 README 的限制章節。
    """

    def __init__(self, latent_dim: int = 16, width: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        w = width
        self.enc = nn.Sequential(
            nn.Conv2d(1, w, 4, stride=2, padding=1),      # 28 → 14
            nn.SiLU(),
            nn.Conv2d(w, w * 2, 4, stride=2, padding=1),  # 14 → 7
            nn.SiLU(),
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1),  # 7 → 4
            nn.SiLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(w * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(w * 4 * 4 * 4, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, w * 4 * 4 * 4)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (w * 4, 4, 4)),
            nn.SiLU(),
            nn.ConvTranspose2d(w * 4, w * 2, 3, stride=2, padding=1),          # 4 → 7
            nn.SiLU(),
            nn.ConvTranspose2d(w * 2, w, 4, stride=2, padding=1),              # 7 → 14
            nn.SiLU(),
            nn.ConvTranspose2d(w, 1, 4, stride=2, padding=1),                  # 14 → 28
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        # clamp logvar：避免 σ→0 造成 log q 爆炸、以及 σ 過大讓重建完全失效。
        # 這不是美化，是數值必需 —— 不 clamp 時實測會出現 NaN。
        logvar = torch.clamp(self.fc_logvar(h), min=-8.0, max=8.0)
        return mu, logvar

    def decode(self, z):
        return self.dec(self.fc_dec(z))               # logits, (N,1,28,28)

    def reparameterize(self, mu, logvar):
        """z = μ + σ⊙ε。訓練時抽樣，這是梯度能穿過隨機節點的關鍵。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


# ---------------------------------------------------------------------------
# ELBO 與它的組成
# ---------------------------------------------------------------------------


def recon_log_prob(logits, x, reduce: bool = True):
    """log p(x|z)，Bernoulli 似然。回傳**逐樣本**的和（越大越好）。"""
    # binary_cross_entropy_with_logits 回傳的是 −log p，所以取負號
    ll = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return ll.flatten(1).sum(1) if reduce else ll


def kl_diag_gaussian(mu, logvar, per_dim: bool = False):
    """KL(q(z|x) ‖ N(0,I))，對角高斯的解析式：

        KL = ½ Σ_j (μ_j² + σ_j² − 1 − log σ_j²)

    `per_dim=True` 回傳每個 latent 維度的 KL —— 那是 posterior collapse
    的直接診斷：塌陷的維度 KL≈0，表示 encoder 對它輸出先驗、
    decoder 也就學會忽略它。
    """
    k = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return k if per_dim else k.sum(1)


def kl_monte_carlo(mu, logvar, n_samples: int = 512, seed: int = 0):
    """用 Monte Carlo 估 KL —— 驗證解析式的正確性。

        KL = E_q[log q(z|x) − log p(z)]

    解析式寫錯（少個 ½、log σ² 的符號、忘記 −1）**不會拋錯**，
    只會讓訓練往一個略微不對的目標收斂，而且結果看起來仍然合理。
    所以必須有獨立的數值檢查。
    """
    g = torch.Generator(device=mu.device).manual_seed(seed)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn((n_samples, *mu.shape), generator=g, device=mu.device,
                      dtype=mu.dtype)
    z = mu.unsqueeze(0) + std.unsqueeze(0) * eps
    log_q = (-0.5 * (LOG2PI + logvar.unsqueeze(0) + eps.pow(2))).sum(-1)
    log_p = (-0.5 * (LOG2PI + z.pow(2))).sum(-1)
    return (log_q - log_p).mean(0)


def elbo_terms(model, x, n_samples: int = 1, seed: int | None = None) -> dict:
    """逐樣本的 ELBO 與它的兩個組成（不做 batch 平均）。

    回傳的 `elbo = recon − kl`。注意符號：三個量都是「越大越好」的方向，
    所以異常分數要取負（見 experiments.py）。
    """
    if seed is not None:
        torch.manual_seed(seed)
    mu, logvar = model.encode(x)
    kl = kl_diag_gaussian(mu, logvar)
    recons = []
    for _ in range(n_samples):
        z = model.reparameterize(mu, logvar)
        recons.append(recon_log_prob(model.decode(z), x))
    recon = torch.stack(recons).mean(0)
    return {"elbo": recon - kl, "recon": recon, "kl": kl,
            "kl_per_dim": kl_diag_gaussian(mu, logvar, per_dim=True),
            "mu": mu, "logvar": logvar}


def iwae_bound(model, x, K: int = 20, seed: int | None = None):
    """Importance-weighted 下界（Burda et al. 2016）：

        L_K = E[ log (1/K) Σ_k w_k ],   w_k = p(x|z_k)p(z_k) / q(z_k|x)

    性質：$L_1 = \\text{ELBO} \\le L_K \\le L_{K+1} \\le \\log p(x)$，
    且 $K\\to\\infty$ 時收斂到 $\\log p(x)$。

    所以 K 就是一個「下界鬆緊」的旋鈕，而本專案用它問一個具體問題：
    **更緊的下界會讓異常偵測更好嗎？** 這不是自明的 —— ELBO 當分數時，
    「模型近似很差」與「這筆資料真的不可能」被混在一起，
    收緊下界會移除前者，而前者對異常偵測可能反而是有用的訊號。

    實作用 `logsumexp` 而非先取 exp，否則 K 稍大就溢出。
    """
    if seed is not None:
        torch.manual_seed(seed)
    mu, logvar = model.encode(x)
    std = torch.exp(0.5 * logvar)
    logws = []
    for _ in range(K):
        eps = torch.randn_like(std)
        z = mu + std * eps
        log_p_x_given_z = recon_log_prob(model.decode(z), x)
        log_p_z = (-0.5 * (LOG2PI + z.pow(2))).sum(1)
        log_q_z = (-0.5 * (LOG2PI + logvar + eps.pow(2))).sum(1)
        logws.append(log_p_x_given_z + log_p_z - log_q_z)
    logw = torch.stack(logws)                                  # (K, N)
    return torch.logsumexp(logw, dim=0) - math.log(K)


# ---------------------------------------------------------------------------
# 重參數化 vs REINFORCE：梯度變異數
# ---------------------------------------------------------------------------


def grad_variance_comparison(model, x, n_trials: int = 60, n_mc: int = 1,
                             seed: int = 0) -> dict:
    """量測兩種梯度估計器對 encoder 參數的梯度變異數。

    **重參數化**：$z=\\mu+\\sigma\\epsilon$，梯度直接穿過 $z$。
    **REINFORCE**：$\\nabla_\\phi \\mathbb{E}_q[f] = \\mathbb{E}_q[f\\,\\nabla_\\phi\\log q]$，
    把 $z$ 當成不可微的抽樣，靠 score function 傳梯度。

    兩者都是**無偏**的，差別只在變異數 —— 而變異數決定能不能用 SGD 訓練。
    這個實驗把「為什麼需要重參數化」從一句引用變成一個數字。

    為了公平，兩邊用**完全相同的隨機數**（同一組 ε 產生同一批 z），
    所以差異純粹來自估計式，不是抽樣運氣。REINFORCE 版另外減去
    batch 內的均值當 baseline（不減的話變異數更誇張，那樣的對照不公平）。
    """
    dev = next(model.parameters()).device
    params = [p for p in model.enc.parameters() if p.requires_grad]
    params += [model.fc_mu.weight, model.fc_logvar.weight]

    g = torch.Generator(device="cpu").manual_seed(seed)
    grads = {"reparam": [], "reinforce": []}

    for t in range(n_trials):
        eps_cpu = torch.randn((n_mc, len(x), model.latent_dim), generator=g)
        eps = eps_cpu.to(dev)

        # ── 重參數化 ──────────────────────────────────────────────────
        model.zero_grad(set_to_none=True)
        mu, logvar = model.encode(x)
        std = torch.exp(0.5 * logvar)
        obj = 0.0
        for k in range(n_mc):
            z = mu + std * eps[k]
            obj = obj + recon_log_prob(model.decode(z), x).mean()
        obj = obj / n_mc - kl_diag_gaussian(mu, logvar).mean()
        obj.backward()
        grads["reparam"].append(torch.cat([p.grad.flatten() for p in params]).cpu())

        # ── REINFORCE（score function）──────────────────────────────────
        model.zero_grad(set_to_none=True)
        mu, logvar = model.encode(x)
        std = torch.exp(0.5 * logvar)
        surrogate = 0.0
        for k in range(n_mc):
            z = (mu + std * eps[k]).detach()          # 切斷路徑梯度
            f = recon_log_prob(model.decode(z), x).detach()
            f = f - f.mean()                          # baseline，降低變異數
            log_q = (-0.5 * (LOG2PI + logvar + ((z - mu) / std).pow(2))).sum(1)
            surrogate = surrogate + (f * log_q).mean()
        surrogate = surrogate / n_mc - kl_diag_gaussian(mu, logvar).mean()
        surrogate.backward()
        grads["reinforce"].append(torch.cat([p.grad.flatten() for p in params]).cpu())

    model.zero_grad(set_to_none=True)
    out = {}
    for k, v in grads.items():
        G = torch.stack(v)                             # (n_trials, n_params)
        # 逐參數的變異數，再取平均 —— 這才是「梯度有多吵」
        var = G.var(dim=0, unbiased=True)
        out[k] = {"mean_var": float(var.mean()),
                  "median_var": float(var.median()),
                  "grad_norm_mean": float(G.norm(dim=1).mean()),
                  "grad_norm_sd": float(G.norm(dim=1).std())}
    out["variance_ratio_reinforce_over_reparam"] = (
        out["reinforce"]["mean_var"] / max(out["reparam"]["mean_var"], 1e-30))
    out["n_trials"] = n_trials
    out["n_params"] = int(sum(p.numel() for p in params))
    return out


def verify_kl_formula(latent_dim: int = 16, n_cases: int = 24,
                      n_samples: int = 20000, seed: int = 0,
                      device_: torch.device | None = None) -> dict:
    """解析 KL vs Monte Carlo KL —— **對受控的 (μ, logvar) 測，不經過模型**。

    ⚠️ 為什麼不用模型的輸出來測（第一版就是那樣，然後「失敗」了）：
    **未訓練**的 encoder 輸出 μ≈0、logvar≈0，於是 KL≈0.003。
    在那個量級上相對誤差的分母趨近 0，MC 的取樣噪聲就會讓相對誤差爆到 1.1，
    看起來像公式錯了 —— 其實是測試設計錯了。

    改成直接抽合理範圍的 (μ, logvar)（KL 落在 O(1)–O(100)），
    公式的正確性才真的被檢驗到。容差按 MC 的標準誤設：
    KL 的 MC 估計標準誤 ≈ sd(log q − log p)/√n，所以這裡一併回報它，
    讓「通過」的判準有依據而不是猜一個數字。
    """
    dev = device_ or device()
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu = (torch.randn((n_cases, latent_dim), generator=g) * 1.5).to(dev)
    logvar = (torch.randn((n_cases, latent_dim), generator=g) * 0.8 - 0.5).to(dev)

    ana = kl_diag_gaussian(mu, logvar)
    mc = kl_monte_carlo(mu, logvar, n_samples=n_samples, seed=seed)

    # MC 的逐樣本標準誤（用同一組抽樣重算散佈）
    std = torch.exp(0.5 * logvar)
    gg = torch.Generator(device=dev).manual_seed(seed)
    eps = torch.randn((n_samples, *mu.shape), generator=gg, device=dev, dtype=mu.dtype)
    z = mu.unsqueeze(0) + std.unsqueeze(0) * eps
    log_q = (-0.5 * (LOG2PI + logvar.unsqueeze(0) + eps.pow(2))).sum(-1)
    log_p = (-0.5 * (LOG2PI + z.pow(2))).sum(-1)
    mc_se = (log_q - log_p).std(0) / math.sqrt(n_samples)

    d = (ana - mc).abs()
    # 通過判準：每個案例的偏差都在 4 個 MC 標準誤內
    within = d <= 4.0 * mc_se
    return {"analytic": ana.cpu().numpy(), "mc": mc.cpu().numpy(),
            "mc_se": mc_se.cpu().numpy(),
            "n_cases": n_cases, "n_mc_samples": n_samples,
            "kl_range": [float(ana.min()), float(ana.max())],
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "max_diff_in_se_units": float((d / mc_se.clamp(min=1e-12)).max()),
            "mc_se_median": float(mc_se.median()),
            "n_within_4se": int(within.sum()),
            "passed": bool(within.all())}


def verify_kl_analytic(model, x, n_samples: int = 4000, seed: int = 0) -> dict:
    """在**訓練後**的模型上再比對一次（真實的 μ/logvar 分佈）。

    與 `verify_kl_formula` 互補：那個測公式本身，這個確認在實際遇到的
    參數範圍內也成立（例如 logvar 被 clamp 到邊界時）。
    """
    with torch.no_grad():
        mu, logvar = model.encode(x)
        ana = kl_diag_gaussian(mu, logvar)
        mc = kl_monte_carlo(mu, logvar, n_samples=n_samples, seed=seed)
    d = (ana - mc).abs()
    kl_scale = max(float(ana.mean()), 1e-12)
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            # 用 batch 的平均 KL 當尺度，而不是逐樣本的 KL（後者可能趨零）
            "max_diff_over_mean_kl": float(d.max() / kl_scale),
            "analytic_mean": float(ana.mean()), "mc_mean": float(mc.mean()),
            "n_mc_samples": n_samples,
            "passed": bool(d.max() / kl_scale < 0.05)}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """與 A3 同樣的教訓：只設 seed 不足以可重現。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
