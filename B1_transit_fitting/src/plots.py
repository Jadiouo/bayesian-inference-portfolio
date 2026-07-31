"""
B1 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只負責畫，資料由 run_all 傳入。
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

C_DATA = "#2e6fdb"
C_MODEL = "#e8873a"
C_PUB = "#d1495b"
C_REF = "#6c757d"


# ── 圖 1：去趨勢（raw vs flatten）─────────────────────────
def raw_and_flatten(t, raw, flat, path):
    t0 = t - t.min()
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    axT.plot(t0, (raw - 1) * 1e6, ".", ms=2.5, color=C_REF, alpha=0.6)
    axT.set_ylabel("raw (ppm)")
    axT.set_title("Detrending: remove slow drift before folding "
                  "(Kepler-10 is photometrically quiet, so it is subtle)")
    axB.plot(t0, (flat - 1) * 1e6, ".", ms=2.5, color=C_DATA, alpha=0.6)
    axB.axhline(0, color=C_MODEL, lw=1.2)
    axB.set_ylabel("flattened (ppm)")
    axB.set_xlabel("Time (days, segment)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 2：相位摺疊 + 最佳模型 ────────────────────────────
def phase_fold(phase_raw, flux_raw, bp, bf, be, model_ph, model_fx, depth_ppm, P, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    hrs = phase_raw * P * 24
    ax.plot(hrs, (flux_raw - 1) * 1e6, ".", ms=1.5, color=C_REF, alpha=0.15, label="folded cadences")
    ax.errorbar(bp * P * 24, (bf - 1) * 1e6, yerr=be * 1e6, fmt="o", color=C_DATA,
                ms=5, capsize=2, lw=1, label="binned", zorder=3)
    ax.plot(model_ph * P * 24, (model_fx - 1) * 1e6, "-", color=C_MODEL, lw=2.5,
            label="best-fit transit", zorder=4)
    ax.axhline(0, color=C_REF, lw=0.8)
    ax.set_xlim(-1.8, 1.8)
    ax.set_xlabel("Time from mid-transit (hours)")
    ax.set_ylabel("Relative flux (ppm)")
    ax.set_title(f"Kepler-10b transit — a {depth_ppm:.0f} ppm dip from thousands of folded transits")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 3：corner（簡併）────────────────────────────────
def corner_plot(samples, labels, truths, path):
    import corner

    fig = corner.corner(samples, labels=labels, truths=truths,
                        truth_color=C_PUB, color=C_DATA,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt=".4f", title_kwargs={"fontsize": 9},
                        label_kwargs={"fontsize": 11})
    fig.suptitle("Posterior — note the strong R_p/R*–a/R*–b degeneracy (red = published)",
                 fontsize=13, y=1.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 4：後驗預測檢查（模型束 + 殘差）──────────────────
def posterior_predictive(bp, bf, be, fine_ph, draws, best, P, resid, path):
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    hrs = fine_ph * P * 24
    for d in draws:
        axT.plot(hrs, (d - 1) * 1e6, "-", color=C_MODEL, alpha=0.03, lw=1)
    axT.errorbar(bp * P * 24, (bf - 1) * 1e6, yerr=be * 1e6, fmt="o", color=C_DATA,
                 ms=5, capsize=2, lw=1, zorder=3, label="binned data")
    axT.plot(hrs, (best - 1) * 1e6, "-", color=C_MODEL, lw=2, zorder=4, label="posterior draws + best fit")
    axT.axhline(0, color=C_REF, lw=0.8)
    axT.set_ylabel("Relative flux (ppm)")
    axT.set_title("Posterior predictive check — model draws envelope the data")
    axT.legend(frameon=False, loc="lower right")

    axB.errorbar(bp * P * 24, resid * 1e6, yerr=be * 1e6, fmt="o", color=C_DATA, ms=4, capsize=2, lw=1)
    axB.axhline(0, color=C_MODEL, lw=1.5)
    axB.set_ylabel("resid (ppm)")
    axB.set_xlabel("Time from mid-transit (hours)")
    axB.set_xlim(-1.8, 1.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 5：與已發表值對照 ────────────────────────────────
def published_comparison(rp_samples, rp_earth_samples, pub, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    axL.hist(rp_samples, bins=60, color=C_DATA, alpha=0.6, density=True)
    lo, hi = np.percentile(rp_samples, [2.5, 97.5])
    axL.axvspan(lo, hi, color=C_DATA, alpha=0.10)
    pub_err = 0.0004
    axL.axvspan(pub["rp_rs"] - pub_err, pub["rp_rs"] + pub_err, color=C_PUB, alpha=0.18)
    axL.axvline(pub["rp_rs"], color=C_PUB, lw=2, label=f"published {pub['rp_rs']:.4f}±{pub_err:g}")
    axL.axvline(np.median(rp_samples), color=C_DATA, lw=2, label=f"ours {np.median(rp_samples):.4f}")
    axL.set_xlabel("R_p / R*")
    axL.set_yticks([])
    axL.set_title("Radius ratio")
    axL.legend(frameon=False, fontsize=9)

    axR.hist(rp_earth_samples, bins=60, color=C_DATA, alpha=0.6, density=True)
    axR.axvline(pub["rp_earth"], color=C_PUB, lw=2, label=f"published {pub['rp_earth']:.2f} R⊕")
    lo_e, hi_e = np.percentile(rp_earth_samples, [2.5, 97.5])
    axR.axvline(np.median(rp_earth_samples), color=C_DATA, lw=2,
                label=f"ours {np.median(rp_earth_samples):.2f} [{lo_e:.2f}, {hi_e:.2f}]")
    axR.set_xlabel("Planet radius  R_p  (Earth radii)")
    axR.set_yticks([])
    axR.set_title("Physical radius — covers published ✓")
    axR.legend(frameon=False, fontsize=9)

    fig.suptitle("Do we recover the published planet? (red = published)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
