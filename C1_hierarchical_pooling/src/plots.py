"""
C1 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只負責畫，數字由 run_all 傳入。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
})

C_NP = "#d1495b"   # no pooling（紅）
C_PP = "#2e6fdb"   # partial pooling（藍）
C_CP = "#6c757d"   # complete pooling（灰）
C_HL = "#e8873a"


# ── 圖 1：三種做法（示例郡）─────────────────────────────
def three_approaches(names, n_ex, np_mean, np_lo, np_hi, pp_mean, pp_lo, pp_hi,
                     a_complete, path):
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(a_complete, color=C_CP, ls="--", lw=1.6, label="complete pooling (one value)")
    ax.errorbar(x - 0.12, np_mean, yerr=[np_mean - np_lo, np_hi - np_mean], fmt="o",
                color=C_NP, ms=6, capsize=3, lw=1.4, label="no pooling (own data only)")
    ax.errorbar(x + 0.12, pp_mean, yerr=[pp_mean - pp_lo, pp_hi - pp_mean], fmt="s",
                color=C_PP, ms=6, capsize=3, lw=1.4, label="partial pooling (hierarchical)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\nn={n}" for nm, n in zip(names, n_ex)], fontsize=8)
    ax.set_ylabel("County intercept  (log radon)")
    ax.set_title("Three ways to estimate each county — small n: no-pooling is wild, "
                 "partial pooling is tamed")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 2：收縮圖（⭐ 靈魂）+ 收縮權重 vs 樣本數 ───────────
def shrinkage(n_j, a_nopool, a_partial, a_complete, omega_emp, n_grid, omega_theory, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.7))

    jitter = np.random.default_rng(0).normal(0, 0.02, len(n_j))
    xn = n_j * np.exp(jitter)
    for i in range(len(n_j)):
        axL.plot([xn[i], xn[i]], [a_nopool[i], a_partial[i]], color="#cbd3da", lw=0.7, zorder=1)
    axL.scatter(xn, a_nopool, s=26, facecolors="none", edgecolors=C_NP, lw=1.2,
                label="no pooling", zorder=2)
    axL.scatter(xn, a_partial, s=22, color=C_PP, label="partial pooling", zorder=3)
    axL.axhline(a_complete, color=C_CP, ls="--", lw=1.5, label="complete pooling (grand mean)")
    axL.set_xscale("log")
    axL.set_xlabel("County sample size  n  (log scale)")
    axL.set_ylabel("County intercept  (log radon)")
    axL.set_title("Shrinkage: small-n counties are pulled toward the grand mean")
    axL.legend(frameon=False, fontsize=9, loc="lower right")

    axR.scatter(n_j, omega_emp, s=26, color=C_PP, alpha=0.7, label="empirical (posterior)")
    axR.plot(n_grid, omega_theory, color=C_HL, lw=2.2,
             label="theory  σ²/(σ²+n·σ_a²)")
    axR.set_xscale("log")
    axR.set_ylim(-0.05, 1.05)
    axR.set_xlabel("County sample size  n  (log scale)")
    axR.set_ylabel("Shrinkage weight toward grand mean  ω")
    axR.set_title("Pull decreases with n — and matches precision-weighting")
    axR.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 3：超參數後驗（郡間差異多大）─────────────────────
def hyperparameters(mu_a_samples, sigma_a_samples, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, s, name, col in [(axL, mu_a_samples, "μ_a  (grand mean intercept)", C_PP),
                             (axR, sigma_a_samples, "σ_a  (between-county SD)", C_HL)]:
        ax.hist(s, bins=50, color=col, alpha=0.55, density=True)
        lo, med, hi = np.percentile(s, [2.5, 50, 97.5])
        ax.axvline(med, color=col, lw=1.8)
        ax.axvline(lo, color=col, ls=":", lw=1.3)
        ax.axvline(hi, color=col, ls=":", lw=1.3)
        ax.set_title(f"{name}\nmedian {med:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]", fontsize=11)
        ax.set_yticks([])
    axR.axvline(0, color=C_CP, ls="--", lw=1.4)
    axR.text(0.01, axR.get_ylim()[1] * 0.5, " complete pooling\n assumes σ_a=0",
             color=C_CP, fontsize=8.5, va="center")
    fig.suptitle("How different are counties? — the question complete pooling can't ask",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 4：預測效能（測試集）─────────────────────────────
def predictive(model_names, rmse_all, buckets, rmse_by_bucket, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = [C_NP, C_CP, C_PP]
    axL.bar(model_names, rmse_all, color=colors, alpha=0.85)
    for i, v in enumerate(rmse_all):
        axL.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    axL.set_ylabel("Test RMSE (log radon)")
    axL.set_title("Held-out prediction error — hierarchical wins")
    axL.set_ylim(0, max(rmse_all) * 1.15)

    x = np.arange(len(buckets))
    w = 0.26
    for i, (name, col) in enumerate(zip(model_names, colors)):
        axR.bar(x + (i - 1) * w, [rmse_by_bucket[name][b] for b in buckets], w,
                color=col, alpha=0.85, label=name)
    axR.set_xticks(x)
    axR.set_xticklabels(buckets)
    axR.set_xlabel("County size bucket")
    axR.set_ylabel("Test RMSE")
    axR.set_title("Advantage is biggest for small counties")
    axR.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 5：funnel 與非中心化（Eight Schools）─────────────
def funnel(theta_c, logtau_c, div_c, theta_nc, logtau_nc, div_nc, n_div_c, n_div_nc, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, th, lt, dv, title, nd in [
        (axL, theta_c, logtau_c, div_c, "Centered", n_div_c),
        (axR, theta_nc, logtau_nc, div_nc, "Non-centered", n_div_nc)]:
        ax.scatter(th[~dv], lt[~dv], s=6, color=C_PP, alpha=0.25, edgecolor="none")
        ax.scatter(th[dv], lt[dv], s=12, color=C_NP, alpha=0.9,
                   label=f"{int(nd)} divergences")
        ax.set_xlabel("θ₁  (school A effect)")
        ax.set_title(f"{title}  parameterization")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    axL.set_ylabel("log τ  (between-school SD)")
    fig.suptitle("The funnel: centered sampling breaks in the neck; non-centered fixes it",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
