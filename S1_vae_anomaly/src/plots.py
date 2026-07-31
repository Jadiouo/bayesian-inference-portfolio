"""
S1 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只吃 numpy / dict。

沿用 A2/A3/B2 的原則：不接觸 torch，資料由 run_all 落盤後傳入，
`replot.py` 秒級重畫而不必重新訓練。
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

C_NORM = "#2e6fdb"      # normal（in-distribution）
C_ANOM = "#d1495b"      # pneumonia（anomaly）
C_TRIVIAL = "#e8873a"   # 不看模型的 baseline
C_MODEL = "#2a9d8f"     # 模型分數
C_REF = "#6c757d"
C_CORR = "#9b5de5"      # 複雜度校正後

SCORE_LAB = {"neg_elbo": "−ELBO", "neg_recon": "−E[log p(x|z)]",
             "kl": "KL(q‖p)", "neg_iwae": "−IWAE$_{K=20}$"}
TRIV_LAB = {"pixel_entropy": "pixel entropy", "gradient_energy": "gradient energy",
            "l2_norm": "L2 norm", "mean_intensity": "mean intensity",
            "std_intensity": "std intensity", "png_bits": "PNG size (bits)"}


def _band(ax, x, lo, hi, color, alpha=0.18):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


# ── 圖 1：資料 + 兩個正確性驗證 ─────────────────────────────────────────
def data_and_verification(samples, kl_verify, kl_verify_trained, grad_var, path):
    """左：範例影像。中：解析 KL vs Monte Carlo。右：重參數化 vs REINFORCE。

    中間和右邊是**方法正確性的證據**，不是結果 —— 在談任何 ELBO 數字之前，
    要先證明 ELBO 的兩個組成都算對了，而且梯度估計器選得有理由。
    """
    fig = plt.figure(figsize=(15.5, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.30)

    # 左：範例影像
    for r, (key, lab, col) in enumerate([("test_normal", "normal (training class)", C_NORM),
                                         ("test_anomaly", "pneumonia (anomaly)", C_ANOM)]):
        ax = fig.add_subplot(gs[r, 0])
        imgs = samples[key][:8]
        grid = np.concatenate([np.concatenate(list(imgs[i * 4:(i + 1) * 4, 0]), axis=1)
                               for i in range(2)], axis=0)
        ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
        ax.set_title(lab, color=col, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    # 中：KL 驗證（受控參數）
    ax = fig.add_subplot(gs[:, 1])
    ana = np.asarray(kl_verify["analytic"]); mc = np.asarray(kl_verify["mc"])
    se = np.asarray(kl_verify["mc_se"])
    ax.errorbar(ana, mc, yerr=4 * se, fmt="o", ms=5, color=C_MODEL,
                ecolor=C_REF, elinewidth=1.0, capsize=2, label="controlled (μ, log σ²)")
    lim = [min(ana.min(), mc.min()) * 0.95, max(ana.max(), mc.max()) * 1.05]
    ax.plot(lim, lim, "--", color=C_REF, lw=1.2, label="y = x")
    ax.set_xlabel("analytic  KL = ½Σ(μ² + σ² − 1 − log σ²)")
    ax.set_ylabel("Monte-Carlo  E$_q$[log q − log p]")
    ax.set_title(f"KL formula verified\n"
                 f"max deviation {kl_verify['max_diff_in_se_units']:.2f} MC standard errors\n"
                 f"(trained model: max |Δ| / mean KL = "
                 f"{kl_verify_trained['max_diff_over_mean_kl']:.4f})", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")

    # 右：梯度變異數
    ax = fig.add_subplot(gs[:, 2])
    names = ["reparameterisation", "REINFORCE\n(+ baseline)"]
    vals = [grad_var["reparam"]["mean_var"], grad_var["reinforce"]["mean_var"]]
    bars = ax.bar(names, vals, color=[C_MODEL, C_ANOM], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("mean per-parameter gradient variance")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.4, f"{v:.2e}",
                ha="center", fontsize=9)
    ratio = grad_var["variance_ratio_reinforce_over_reparam"]
    ax.set_title(f"Why reparameterisation is needed\n"
                 f"REINFORCE variance is {ratio:,.0f}× larger\n"
                 f"(same ε, {grad_var['n_trials']} trials, "
                 f"{grad_var['n_params']:,} encoder params)", fontsize=10)
    ax.set_ylim(top=max(vals) * 12)

    fig.suptitle("Setup: the data, and two correctness checks before any result",
                 fontsize=13, y=0.975)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 2：ELBO 分佈 —— 通關標準 1 ────────────────────────────────────────
def elbo_distributions(sc_norm, sc_anom, table, path):
    """三欄：ELBO / 重建項 / KL 項的 normal-vs-anomaly 直方圖。

    通關標準說「異常樣本的 ELBO 明顯較低（直方圖可分離）」。
    這張圖同時給出它的量化（AUROC、Cohen's d、重疊係數）——
    「看起來分開」和「分得開」是兩件事，後者才可檢驗。
    """
    keys = ["elbo", "recon", "kl"]
    labs = ["ELBO = E[log p(x|z)] − KL", "reconstruction  E[log p(x|z)]", "KL(q(z|x) ‖ p(z))"]
    tkeys = ["neg_elbo", "neg_recon", "kl"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    for ax, k, lab, tk in zip(axes, keys, labs, tkeys):
        a, b = sc_norm[k], sc_anom[k]
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        bins = np.linspace(lo, hi, 48)
        ax.hist(a, bins=bins, color=C_NORM, alpha=0.62, label=f"normal (n={len(a)})",
                density=True)
        ax.hist(b, bins=bins, color=C_ANOM, alpha=0.62, label=f"pneumonia (n={len(b)})",
                density=True)
        ax.axvline(np.median(a), color=C_NORM, lw=1.6, ls="--")
        ax.axvline(np.median(b), color=C_ANOM, lw=1.6, ls="--")
        t = table[tk]
        sep = t["separation"]
        ax.set_xlabel(lab)
        ax.set_ylabel("density")
        ax.set_title(f"AUROC = {t['auroc']:.3f}  [{t['lo95']:.3f}, {t['hi95']:.3f}]\n"
                     f"Cohen's d = {sep['cohens_d']:+.2f},  overlap = "
                     f"{sep['overlap_coefficient']:.2f}", fontsize=10)
        ax.legend(fontsize=9)

    axes[2].text(0.5, 0.97, "AUROC < 0.5:\nanomalies sit CLOSER to the prior",
                 transform=axes[2].transAxes, ha="center", va="top", fontsize=9.5,
                 color=C_ANOM,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_ANOM, alpha=0.9))

    fig.suptitle("Gate 1 — is the anomaly ELBO lower, and by how much?",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 3：重建 ──────────────────────────────────────────────────────────
def reconstructions(rec, path, n: int = 8):
    """兩組各 n 張的原圖 / 重建 / 逐像素誤差，並標上該樣本的 ELBO。"""
    fig, axes = plt.subplots(6, n, figsize=(1.55 * n, 9.6))
    for blk, (key, lab, col) in enumerate([("test_normal", "normal", C_NORM),
                                           ("test_anomaly", "pneumonia", C_ANOM)]):
        d = rec[key]
        for j in range(n):
            for row, (img, cmap, vmin, vmax) in enumerate([
                    (d["x"][j, 0], "gray", 0, 1),
                    (d["xhat"][j, 0], "gray", 0, 1),
                    (np.abs(d["x"][j, 0] - d["xhat"][j, 0]), "inferno", 0, 0.5)]):
                ax = axes[blk * 3 + row, j]
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if j == 0:
                    ax.set_ylabel(["input", "recon", "|error|"][row], fontsize=9)
            axes[blk * 3, j].set_title(f"{d['elbo'][j]:.0f}", fontsize=8, color=col)

    axes[0, 0].text(-0.55, 0.5, "NORMAL", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="center", color=C_NORM, fontsize=12, weight="bold")
    axes[3, 0].text(-0.55, 0.5, "PNEUMONIA", transform=axes[3, 0].transAxes, rotation=90,
                    va="center", ha="center", color=C_ANOM, fontsize=12, weight="bold")
    fig.suptitle("Reconstructions from the posterior mean μ(x)  (title = per-sample ELBO)",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 4：trivial baseline 對照 —— 本專案最重要的一張 ───────────────────
def trivial_comparison(table, trivial, corrected, path):
    """模型分數 vs 不看模型的像素統計。

    A3 專案的教訓在這裡重演：**沒有這個對照，「VAE 偵測到病灶」
    這句話無法被證偽**。條形按 AUROC 排序，模型與 trivial 用顏色區分。
    """
    rows = []
    for k, v in table.items():
        rows.append((SCORE_LAB[k], v["auroc"], v["lo95"], v["hi95"], "model"))
    for k, v in trivial["scores"].items():
        rows.append((TRIV_LAB.get(k, k), v["auroc"], v["lo95"], v["hi95"], "trivial"))
    cc = corrected["neg_elbo"]["corrected"]
    rows.append(("−ELBO − L(x)\n(complexity-corrected)", cc["auroc"], cc["lo95"],
                 cc["hi95"], "corrected"))
    rows.sort(key=lambda r: r[1])

    cols = {"model": C_MODEL, "trivial": C_TRIVIAL, "corrected": C_CORR}
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(rows))
    for i, (lab, a, lo, hi, kind) in enumerate(rows):
        ax.barh(i, a, color=cols[kind], alpha=0.85, height=0.66)
        ax.plot([lo, hi], [i, i], color="black", lw=1.5, alpha=0.75)
        ax.text(hi + 0.010, i, f"{a:.3f}", va="center", fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.axvline(0.5, color=C_REF, ls="--", lw=1.2)
    ax.text(0.505, len(rows) - 0.4, "chance", color=C_REF, fontsize=9)
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("AUROC  (normal vs pneumonia, 95% bootstrap CI)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[k]) for k in
               ("model", "trivial", "corrected")]
    ax.legend(handles, ["VAE score", "trivial pixel statistic (no model)",
                        "VAE corrected for complexity"], loc="lower right", fontsize=9.5)
    best_t = trivial["best"]
    ax.set_title(f"The uncomfortable comparison: a gradient-energy statistic beats the VAE\n"
                 f"best trivial = {TRIV_LAB.get(best_t[0], best_t[0])} "
                 f"({best_t[1]:.3f})  vs  −ELBO ({table['neg_elbo']['auroc']:.3f})",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 5：複雜度混淆 ────────────────────────────────────────────────────
def complexity_confound(sc_norm, sc_anom, trivial, analysis, corrected, path):
    """左：ELBO vs PNG 位元數散佈。中：分層 AUROC。右：校正前後。"""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # 左：散佈
    ax = axes[0]
    bn, ba = trivial["raw_normal"]["png_bits"], trivial["raw_anomaly"]["png_bits"]
    ax.scatter(bn, sc_norm["elbo"], s=11, alpha=0.55, color=C_NORM, label="normal")
    ax.scatter(ba, sc_anom["elbo"], s=11, alpha=0.55, color=C_ANOM, label="pneumonia")
    ax.set_xlabel("PNG size of the image (bits) — a model-free complexity proxy")
    ax.set_ylabel("ELBO (nats)")
    sp = analysis["spearman_elbo_vs_bits"]
    ax.set_title(f"ELBO tracks image complexity\n"
                 f"Spearman ρ = {sp['normal_only']:.2f} (normal only), "
                 f"{sp['pooled']:.2f} (pooled)", fontsize=10)
    ax.legend(fontsize=9)

    # 中：分層 AUROC
    ax = axes[1]
    st = analysis["stratified"]
    order = ["neg_elbo", "neg_recon", "neg_iwae", "kl"]
    x = np.arange(len(order))
    raw = [analysis["raw_auroc"][k] for k in order]
    strat = [st[k]["auroc"] for k in order]
    ax.bar(x - 0.19, raw, width=0.36, color=C_MODEL, alpha=0.85, label="raw AUROC")
    ax.bar(x + 0.19, strat, width=0.36, color=C_MODEL, alpha=0.42,
           label=f"within complexity bins (k={analysis['n_bins']})")
    floor = st["png_bits_control"]["auroc"]
    ax.axhline(floor, color=C_TRIVIAL, ls="-.", lw=1.6)
    ax.text(-0.42, floor + 0.015,
            f"residual floor {floor:.3f}\n(complexity stratified against itself)",
            fontsize=8.5, color=C_TRIVIAL, ha="left", va="bottom")
    ax.axhline(0.5, color=C_REF, ls="--", lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels([SCORE_LAB[k] for k in order], fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.30, 0.85)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Control for complexity and most — not all — of\nthe signal disappears",
                 fontsize=10)

    # 右：校正前後
    ax = axes[2]
    cc = corrected["neg_elbo"]
    names = ["−ELBO\n(raw)", "−ELBO − L(x)\n(corrected)"]
    vals = [cc["raw"]["auroc"], cc["corrected"]["auroc"]]
    errs = [[vals[0] - cc["raw"]["lo95"], vals[1] - cc["corrected"]["lo95"]],
            [cc["raw"]["hi95"] - vals[0], cc["corrected"]["hi95"] - vals[1]]]
    ax.bar(names, vals, color=[C_MODEL, C_CORR], width=0.5, alpha=0.88,
           yerr=errs, capsize=5, ecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.035, f"{v:.3f}", ha="center", fontsize=11)
    d = cc["paired_delta"]
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color=C_REF, ls="--", lw=1.1)
    ax.set_title(f"Subtracting a generic compressor's bits\n"
                 f"Δ = {d['delta']:+.3f}  [{d['lo95']:+.3f}, {d['hi95']:+.3f}]  "
                 f"(paired, win rate {d['win_rate']:.0%})", fontsize=10)

    fig.suptitle("Is the ELBO detecting pathology, or just image complexity?",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 6：IWAE 與 posterior collapse ────────────────────────────────────
def bound_and_collapse(iwae, beta, kl_per_dim, seed_stab, path, beta_high: float = 16.0):
    """左：IWAE 的 K。中：β 掃描（活躍維度 vs AUROC）。右：per-dim KL。"""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # 左：IWAE
    ax = axes[0]
    K = [r["K"] for r in iwae["rows"]]
    a = [r["auroc"] for r in iwae["rows"]]
    lo = [r["lo95"] for r in iwae["rows"]]
    hi = [r["hi95"] for r in iwae["rows"]]
    ax.plot(K, a, "o-", color=C_MODEL, lw=1.9, ms=6, label="AUROC")
    _band(ax, K, lo, hi, C_MODEL)
    band = seed_stab["sd"]
    ax.fill_between(K, np.array(a).mean() - band, np.array(a).mean() + band,
                    color=C_REF, alpha=0.16, linewidth=0)
    ax.text(K[len(K) // 2], np.array(a).mean() + band * 1.25,
            f"±1 seed SD ({band:.4f})", fontsize=8.5, color=C_REF, ha="center")
    ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels(K)
    ax.set_xlabel("K (importance samples)")
    ax.set_ylabel("AUROC")
    ax2 = ax.twinx()
    gaps = [r["gap_to_elbo_normal"] for r in iwae["rows"]]
    ax2.plot(K, gaps, "s--", color=C_TRIVIAL, lw=1.5, ms=5, label="bound tightening")
    ax2.set_ylabel("L$_K$ − ELBO  (nats, normal set)", color=C_TRIVIAL)
    ax2.tick_params(axis="y", colors=C_TRIVIAL); ax2.grid(False)
    ax.set_title("A tighter bound does not help detection\n"
                 f"bound gains {gaps[-1] - gaps[0]:.2f} nats, AUROC moves "
                 f"{(max(a) - min(a)):.4f}", fontsize=10)
    ax.legend(fontsize=9, loc="lower left")

    # 中：β 掃描
    ax = axes[1]
    b = [r["beta"] for r in beta["rows"]]
    ax.plot(b, [r["auroc_elbo"] for r in beta["rows"]], "o-", color=C_MODEL,
            lw=1.9, ms=6, label="AUROC (−ELBO)")
    _band(ax, b, [r["lo95"] for r in beta["rows"]], [r["hi95"] for r in beta["rows"]],
          C_MODEL)
    ax.set_xscale("log", base=2); ax.set_xticks(b)
    ax.set_xticklabels([f"{v:g}" for v in b])
    ax.set_xlabel("β  (weight on the KL term)")
    ax.set_ylabel("AUROC", color=C_MODEL)
    ax.axvline(1.0, color=C_REF, ls="--", lw=1.1)
    ax.text(1.02, ax.get_ylim()[0] + 0.004, "β=1: the only true ELBO",
            fontsize=8.5, color=C_REF, rotation=90, va="bottom")
    ax3 = ax.twinx()
    ax3.plot(b, [r["n_active"] for r in beta["rows"]], "s--", color=C_ANOM, lw=1.6, ms=5)
    ax3.set_ylabel("active latent units (mean KL > 0.01)", color=C_ANOM)
    ax3.tick_params(axis="y", colors=C_ANOM); ax3.grid(False)
    n_hi = [r for r in beta["rows"] if r["beta"] == beta_high]
    hi_txt = (f"at β={beta_high:g}: {n_hi[0]['n_active']} active units, "
              f"AUROC {n_hi[0]['auroc_elbo']:.3f}") if n_hi else ""
    ax.set_title(f"Total posterior collapse does not hurt detection\n{hi_txt}", fontsize=10)
    ax.legend(fontsize=9, loc="lower left")

    # 右：per-dim KL
    ax = axes[2]
    for lab, v, col in [("β = 1", kl_per_dim["beta1"], C_MODEL),
                        (f"β = {beta_high:g}", kl_per_dim["beta_high"], C_ANOM)]:
        v = np.sort(np.asarray(v))[::-1]
        ax.plot(np.arange(1, len(v) + 1), np.maximum(v, 1e-6), "o-", color=col,
                lw=1.7, ms=4.5, label=lab)
    ax.axhline(0.01, color=C_REF, ls="--", lw=1.2)
    ax.text(1.2, 0.0115, "active-unit threshold (0.01 nats)", fontsize=8.5, color=C_REF)
    ax.set_yscale("log")
    ax.set_xlabel("latent dimension (sorted by KL)")
    ax.set_ylabel("mean KL contribution (nats)")
    ax.set_title("Which latent dimensions carry information", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle("Two knobs that should matter — and mostly do not",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 7：latent 空間、容量、seed ───────────────────────────────────────
def latent_capacity_seed(embed, latent, seed_stab, path):
    """左：latent PCA 投影。中：latent_dim 掃描。右：seed 散佈。"""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax = axes[0]
    pn, pa = embed["proj"]["test_normal"], embed["proj"]["test_anomaly"]
    ax.scatter(pn[:, 0], pn[:, 1], s=13, alpha=0.6, color=C_NORM, label="normal")
    ax.scatter(pa[:, 0], pa[:, 1], s=13, alpha=0.6, color=C_ANOM, label="pneumonia")
    evr = embed["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 of μ(x)  ({evr[0]:.0%} var)")
    ax.set_ylabel(f"PC2 of μ(x)  ({evr[1]:.0%} var)")
    ax.set_title("Latent means: anomalies overlap the normal cloud\n"
                 "(PCA fitted on normal samples only)", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes[1]
    d = [r["latent_dim"] for r in latent["rows"]]
    a = [r["auroc_elbo"] for r in latent["rows"]]
    ax.plot(d, a, "o-", color=C_MODEL, lw=1.9, ms=6, label="AUROC (−ELBO)")
    _band(ax, d, [r["lo95"] for r in latent["rows"]],
          [r["hi95"] for r in latent["rows"]], C_MODEL)
    m = seed_stab["mean"]
    ax.fill_between(d, m - seed_stab["sd"], m + seed_stab["sd"], color=C_REF,
                    alpha=0.16, linewidth=0)
    ax.text(d[-1], m + seed_stab["sd"] * 1.3, "±1 seed SD", fontsize=8.5,
            color=C_REF, ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks(d); ax.set_xticklabels(d)
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("AUROC", color=C_MODEL)
    ax4 = ax.twinx()
    ax4.plot(d, [r["n_active"] for r in latent["rows"]], "s--", color=C_ANOM,
             lw=1.6, ms=5)
    ax4.set_ylabel("active units", color=C_ANOM)
    ax4.tick_params(axis="y", colors=C_ANOM); ax4.grid(False)
    ax.set_title("Capacity binds only below d = 8,\nthen saturates", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[2]
    s = [r["seed"] for r in seed_stab["rows"]]
    a = [r["auroc_elbo"] for r in seed_stab["rows"]]
    ax.bar(s, a, color=C_MODEL, alpha=0.8, width=0.55)
    ax.axhline(seed_stab["mean"], color=C_ANOM, lw=1.6)
    ax.fill_between([-0.6, max(s) + 0.6], seed_stab["mean"] - seed_stab["sd"],
                    seed_stab["mean"] + seed_stab["sd"], color=C_ANOM, alpha=0.16,
                    linewidth=0)
    ax.set_xlim(-0.6, max(s) + 0.6)
    ax.set_ylim(min(a) - 0.02, max(a) + 0.015)
    ax.set_xticks(s)
    ax.set_xlabel("training seed")
    ax.set_ylabel("AUROC (−ELBO)")
    ax.set_title(f"Run-to-run spread sets the resolution floor\n"
                 f"mean {seed_stab['mean']:.4f}, SD {seed_stab['sd']:.4f}, "
                 f"range {seed_stab['range']:.4f}", fontsize=10)

    fig.suptitle("Where the signal is not: latent geometry, capacity, and noise floor",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
