"""
D1 · 圖表。標籤用英文（matplotlib 預設字型不含 CJK；中文敘事在 README / notebook）。
所有函式只負責「畫」，數字由 run_all.py 算好傳入，保持繪圖純粹。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
})

C_A = "#d1495b"   # 對照組 A（紅）
C_B = "#2e6fdb"   # 實驗組 B（藍）
C_REF = "#6c757d" # 參考線（灰）
C_HL = "#e8873a"  # 強調（橘）


# ── 圖 1：後驗 + 提升幅度（回答老闆的問題）─────────────────────────
def posteriors_and_lift(summary, lift_samples: np.ndarray, path: str):
    aA, bA = summary.postA
    aB, bB = summary.postB
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    lo = min(summary.rateA, summary.rateB) - 4 * np.sqrt(summary.rateA * (1 - summary.rateA) / summary.nA)
    hi = max(summary.rateA, summary.rateB) + 4 * np.sqrt(summary.rateB * (1 - summary.rateB) / summary.nB)
    x = np.linspace(max(lo, 0), hi, 800)
    for (a, b), c, name, rate in [((aA, bA), C_A, "A (control)", summary.rateA),
                                  ((aB, bB), C_B, "B (variant)", summary.rateB)]:
        y = stats.beta.pdf(x, a, b)
        axL.plot(x, y, color=c, lw=2, label=f"{name}: {rate:.1%}")
        axL.fill_between(x, y, color=c, alpha=0.12)
        axL.axvline(rate, color=c, ls=":", lw=1.2, alpha=0.8)
    axL.set_xlabel("Conversion rate  θ")
    axL.set_ylabel("Posterior density")
    axL.set_title(f"Posteriors  ·  P(B > A) = {summary.prob_b_beats_a:.1%}")
    axL.legend(frameon=False, loc="upper right")
    axL.set_yticks([])

    axR.hist(lift_samples * 100, bins=120, color=C_B, alpha=0.55, density=True)
    axR.axvline(0, color=C_REF, lw=1.4, ls="--", label="no difference")
    axR.axvline(summary.lift_lo * 100, color=C_HL, lw=1.5)
    axR.axvline(summary.lift_hi * 100, color=C_HL, lw=1.5,
                label=f"95% CI [{summary.lift_lo:+.0%}, {summary.lift_hi:+.0%}]")
    axR.axvline(summary.lift_med * 100, color="#1b3a6b", lw=1.5, ls="-",
                label=f"median {summary.lift_med:+.1%}")
    axR.set_xlabel("Relative lift  θ_B / θ_A − 1  (%)")
    axR.set_ylabel("Posterior density")
    axR.set_title("How much better is B?")
    axR.legend(frameon=False, loc="upper right", fontsize=9)
    axR.set_yticks([])

    fig.suptitle("Bayesian A/B test — answering the question the boss actually asks",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 2：偷看資料的假陽性率（⭐ 主打圖）──────────────────────────
def peeking_false_positive(fpr, trajectories: np.ndarray, bayes_band: float, path: str):
    looks = fpr["looks"]
    rates = fpr["rates"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    styles = {
        "Frequentist p<0.05": dict(color=C_A, marker="o"),
        "Bayes P>0.95": dict(color=C_B, marker="s"),
        "Bayes P>0.975": dict(color="#7aa5f0", marker="^"),
        "Expected loss<ε": dict(color="#2a9d8f", marker="D"),
    }
    for label, arr in rates.items():
        st = styles.get(label, dict(color=C_REF, marker="o"))
        axL.plot(looks, arr * 100, lw=2, ms=5, label=label, **st)
    axL.axhline(5, color=C_REF, ls="--", lw=1.3, label="nominal 5%")
    axL.set_xlabel("Number of looks (peeks) over the same horizon")
    axL.set_ylabel("False-positive rate  (A/A test, %)")
    axL.set_title("Peeking inflates false positives")
    axL.legend(frameon=False, fontsize=9, loc="upper left")

    # 右：A/A 的後驗 P(B>A) 每日軌跡，強調曾越界者
    days = trajectories.shape[1]
    xday = np.arange(1, days + 1)
    crossed = (trajectories.max(1) > bayes_band) | (trajectories.min(1) < 1 - bayes_band)
    for tr in trajectories[~crossed][:60]:
        axR.plot(xday, tr, color=C_REF, alpha=0.18, lw=0.7)
    for tr in trajectories[crossed][:40]:
        axR.plot(xday, tr, color=C_HL, alpha=0.55, lw=0.9)
    axR.axhline(bayes_band, color=C_B, ls="--", lw=1.3, label=f"P>{bayes_band:g} → declare B")
    axR.axhline(1 - bayes_band, color=C_A, ls="--", lw=1.3, label=f"P<{1-bayes_band:g} → declare A")
    axR.set_ylim(0, 1)
    axR.set_xlabel("Day (each day = one peek)")
    axR.set_ylabel("Posterior P(B > A)")
    axR.set_title(f"A/A posterior wanders across thresholds\n({crossed.mean():.0%} of trials cross at least once)")
    axR.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 3：決策成本 —— 何時值得上線 ──────────────────────────────
def when_to_ship(ns, el_ship_b, el_stay_a, eps_levels, lifts, el_vs_lift, n_fixed, path: str):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axL.plot(ns, np.array(el_ship_b) * 1e4, color=C_B, lw=2, label="ship B  (regret if A better)")
    axL.plot(ns, np.array(el_stay_a) * 1e4, color=C_A, lw=2, label="stay A  (regret if B better)")
    for eps in eps_levels:
        axL.axhline(eps * 1e4, color=C_REF, ls=":", lw=1.1)
        axL.text(ns[-1], eps * 1e4, f" ε={eps*100:.3g}pp", va="center", fontsize=8, color=C_REF)
    axL.set_xlabel("Samples per arm  (rates fixed at 5.2% vs 5.8%)")
    axL.set_ylabel("Expected loss  (×10⁻⁴ conv-rate)")
    axL.set_title("Regret shrinks as data grows → ship when below tolerance ε")
    axL.legend(frameon=False, fontsize=9)
    axL.set_yscale("log")

    axR.plot(np.array(lifts) * 100, np.array(el_vs_lift) * 1e4, color=C_B, lw=2)
    for eps in eps_levels:
        axR.axhline(eps * 1e4, color=C_REF, ls=":", lw=1.1)
        axR.text(lifts[-1] * 100, eps * 1e4, f" ε={eps*100:.3g}pp", va="center", fontsize=8, color=C_REF)
    axR.set_xlabel("Observed relative lift of B  (%)")
    axR.set_ylabel("Expected loss of shipping B  (×10⁻⁴)")
    axR.set_title(f"Minimum lift worth shipping  (n={n_fixed:,}/arm)")
    axR.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 4：先驗敏感度（小樣本）────────────────────────────────────
def prior_sensitivity(labels, prob_ba, exp_loss, scenario_txt, path: str):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(labels))
    axL.bar(xs, np.array(prob_ba) * 100, color=C_B, alpha=0.85)
    axL.axhline(50, color=C_REF, ls="--", lw=1)
    for x, v in zip(xs, prob_ba):
        axL.text(x, v * 100 + 1, f"{v:.1%}", ha="center", fontsize=9)
    axL.set_xticks(xs); axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel("P(B > A)  (%)")
    axL.set_title("Posterior probability by prior")
    axL.set_ylim(0, 100)

    axR.bar(xs, np.array(exp_loss) * 1e4, color=C_HL, alpha=0.85)
    for x, v in zip(xs, exp_loss):
        axR.text(x, v * 1e4, f"{v*100:.3g}pp", ha="center", va="bottom", fontsize=9)
    axR.set_xticks(xs); axR.set_xticklabels(labels, fontsize=9)
    axR.set_ylabel("Expected loss, ship B  (×10⁻⁴)")
    axR.set_title("Expected loss by prior")

    fig.suptitle(f"Prior sensitivity — small sample ({scenario_txt})", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
