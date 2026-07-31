"""
A1 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只負責畫，數字由 run_all 傳入。
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

C_B = "#2e6fdb"   # Bayesian（藍）
C_F = "#d1495b"   # frequentist / plug-in（紅）
C_REF = "#6c757d"
C_HL = "#e8873a"
C_OK = "#2a9d8f"


# ── 圖 1：後驗係數（森林圖）──────────────────────────────
def coefficients(names, mean, lo, hi, path):
    order = np.argsort(mean)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for yi, idx in zip(y, order):
        c = C_B if mean[idx] > 0 else C_F
        ax.plot([lo[idx], hi[idx]], [yi, yi], color=c, lw=2, alpha=0.8)
        ax.plot(mean[idx], yi, "o", color=c, ms=5)
    ax.axvline(0, color=C_REF, ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("Standardized coefficient  (log-odds, 95% CI)")
    ax.set_title("Bayesian logistic regression — who drives heart-disease risk?")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 2：後驗預測 vs plug-in ─────────────────────────────
def predictive_vs_plugin(p_plug, p_pred, ex_samples, ex_plug, ex_pred, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axL.scatter(p_plug, p_pred, s=22, color=C_B, alpha=0.6, edgecolor="none")
    axL.plot([0, 1], [0, 1], color=C_REF, ls="--", lw=1.2, label="y = x")
    axL.axhline(0.5, color=C_HL, ls=":", lw=1)
    axL.set_xlabel("plug-in  sigmoid(E[α]+X·E[β])")
    axL.set_ylabel("posterior predictive  E[sigmoid(α+Xβ)]")
    axL.set_title("Predictive is pulled toward 0.5\n(Jensen: nonlinearity + uncertainty)")
    axL.legend(frameon=False, loc="upper left")

    k = ex_samples.shape[1]
    yy = np.arange(k)
    lo = np.percentile(ex_samples, 2.5, axis=0)
    hi = np.percentile(ex_samples, 97.5, axis=0)
    for j in range(k):
        axR.plot([lo[j], hi[j]], [j, j], color=C_B, lw=3, alpha=0.35)
    axR.plot(ex_pred, yy, "o", color=C_B, ms=7, label="posterior predictive (mean + 95% CI)")
    axR.plot(ex_plug, yy, "X", color=C_F, ms=9, label="plug-in point")
    axR.set_yticks(yy)
    axR.set_yticklabels([f"patient {j+1}" for j in range(k)], fontsize=9)
    axR.set_xlabel("P(disease)")
    axR.set_xlim(0, 1)
    axR.set_title("Same data, honest uncertainty\nplug-in hides the interval")
    axR.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 3：最優門檻（⭐ 決策主打）──────────────────────────
def optimal_threshold(ratios, p_stars, thr_grid, loss_curve, ratio_curve,
                      p_star_curve, loss_at_half, loss_at_star, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    axL.plot(ratios, p_stars, "o-", color=C_B, lw=2, ms=7)
    for r, ps in zip(ratios, p_stars):
        axL.annotate(f"{ps:.3f}", (r, ps), textcoords="offset points",
                     xytext=(6, 6), fontsize=9, color=C_B)
    axL.axhline(0.5, color=C_REF, ls="--", lw=1.2, label="default 0.5")
    axL.set_xscale("log")
    axL.set_xlabel("Cost ratio  C_FN / C_FP  (missing disease vs false alarm)")
    axL.set_ylabel("Optimal threshold  p*")
    axL.set_title("As missing disease gets costlier,\nthe alarm threshold plummets")
    axL.legend(frameon=False)

    axR.plot(thr_grid, loss_curve, color=C_B, lw=2)
    axR.axvline(0.5, color=C_REF, ls="--", lw=1.3,
                label=f"default 0.5  (loss {loss_at_half:.2f})")
    axR.axvline(p_star_curve, color=C_HL, lw=1.6,
                label=f"optimal p*={p_star_curve:.3f}  (loss {loss_at_star:.2f})")
    axR.set_xlabel("Decision threshold")
    axR.set_ylabel("Realized loss on test set")
    axR.set_title(f"Using 0.5 is costly  (C_FN:C_FP = {ratio_curve}:1)")
    axR.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 4：棄權選項 ───────────────────────────────────────
def reject_option(p_grid, treat_loss, notreat_loss, regions, c_fn, p_star, path):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(p_grid, treat_loss, color=C_B, lw=2, label="E[loss | treat] = (1−p)·C_FP")
    ax.plot(p_grid, notreat_loss, color=C_F, lw=2, label="E[loss | not treat] = p·C_FN")
    ytop = c_fn * 0.55
    band_colors = [C_HL, C_OK]
    for i, ((c_rej, (lo, hi), frac), col) in enumerate(zip(regions, band_colors)):
        ax.axhline(c_rej, color=col, ls=":", lw=1.3)
        ax.axvspan(lo, hi, color=col, alpha=0.11)
        ax.text(0.02, ytop * (0.95 - 0.085 * i),
                f"C_reject={c_rej:g}: refer [{lo:.2f}, {hi:.2f}]  ({frac:.0%} of patients)",
                color=col, fontsize=9.5, fontweight="bold")
    ax.axvline(p_star, color=C_REF, ls="--", lw=1, alpha=0.7)
    ax.text(p_star + 0.008, ytop * 0.03, f"p*={p_star:.3f}", color=C_REF, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ytop)
    ax.set_xlabel("P(disease)  p")
    ax.set_ylabel("Expected loss")
    ax.set_title("Add a third action — 'refer / re-test' — and a reject region appears\n"
                 "(smaller referral cost → wider reject region)")
    ax.legend(frameon=False, loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 5：校準（reliability diagram）─────────────────────
def calibration(conf_b, acc_b, ece_b, conf_f, acc_f, ece_f, p_all_b, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    axL.plot([0, 1], [0, 1], color=C_REF, ls="--", lw=1.4, label="perfect calibration")
    mb = ~np.isnan(conf_b)
    mf = ~np.isnan(conf_f)
    axL.plot(conf_b[mb], acc_b[mb], "o-", color=C_B, lw=2, ms=6,
             label=f"Bayesian   (ECE={ece_b:.3f})")
    axL.plot(conf_f[mf], acc_f[mf], "s-", color=C_F, lw=2, ms=6,
             label=f"Frequentist (ECE={ece_f:.3f})")
    axL.set_xlabel("Predicted probability (confidence)")
    axL.set_ylabel("Observed frequency (accuracy)")
    axL.set_title("Reliability diagram (5-fold OOF)")
    axL.legend(frameon=False, loc="upper left")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)

    axR.hist(p_all_b, bins=20, color=C_B, alpha=0.6)
    axR.set_xlabel("Predicted P(disease), Bayesian")
    axR.set_ylabel("count")
    axR.set_title("Where the predictions live")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 6：先驗敏感度 ─────────────────────────────────────
def prior_sensitivity(top_names, means, los, his, prior_labels, aucs, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    k = len(top_names)
    y = np.arange(k)
    colors = [C_F, C_B, C_HL]
    off = np.linspace(-0.25, 0.25, len(prior_labels))
    for j, plab in enumerate(prior_labels):
        ax.errorbar(means[j], y + off[j],
                    xerr=[means[j] - los[j], his[j] - means[j]],
                    fmt="o", color=colors[j], ms=5, lw=1.6, capsize=2,
                    label=f"{plab}  (test AUC {aucs[j]:.3f})")
    ax.axvline(0, color=C_REF, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Standardized coefficient (95% CI)")
    ax.set_title("Prior sensitivity — conclusions barely move\n(N(0,1) vs N(0,2.5) vs N(0,10))")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
