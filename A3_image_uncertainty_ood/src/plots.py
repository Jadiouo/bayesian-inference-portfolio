"""
A3 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只吃 numpy / dict。

沿用 A2/B2 的原則：不接觸 torch，所有資料由 run_all 落盤後傳入，
`replot.py` 可秒級重畫而不必重新訓練模型。
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

C_IN = "#2e6fdb"
C_OOD = "#d1495b"
C_REF = "#6c757d"
C_TRIVIAL = "#e8873a"
C_OK = "#2a9d8f"
C_BAD = "#9b5de5"

MCOL = {"single": "#8d99ae", "mcdrop": "#2e6fdb", "ensemble": "#2a9d8f"}
MLAB = {"single": "Single (softmax)", "mcdrop": "MC Dropout", "ensemble": "Deep Ensemble"}
OOD_LAB = {
    "fashion": "FashionMNIST\n(different objects)",
    "derma": "DermaMNIST\n(colour, other organ)",
    "derma_gray": "DermaMNIST grayscale\n(colour cue removed)",
    "path_held": "PathMNIST held-out classes\n(same imaging distribution)",
}
OOD_SHORT = {"fashion": "Fashion", "derma": "Derma", "derma_gray": "Derma-gray",
             "path_held": "Path held-out"}


# ── 圖 1：epistemic / 信心分佈 —— 通關標準 1，誠實版 ─────────────────────
def uncertainty_histograms_multi(in_unc_by_ood, ood_unc, conf_in_by_ood, conf_ood,
                                 aurocs, path, method: str = "ensemble"):
    """上排：epistemic 直方圖（in vs 各 OOD）。下排：max-softmax 信心分佈。

    `in_unc_by_ood` / `conf_in_by_ood` 是「每個 OOD 欄對應哪個 in-distribution」
    的映射 —— 因為 PathMNIST 留出類別那一組的 in-distribution 是 PathMNIST 本身，
    不是胸片。用同一個 in-dist 對照全部 OOD 會讓最難的那組看起來假地好。

    下排是關鍵的誠實內容：如果模型對 OOD **比對 in-distribution 更自信**，
    上排的直方圖就不會右移，而 AUROC 會掉到 0.5 以下。
    只畫上排會讓讀者以為「沒右移」是隨機噪聲，看到下排才知道是系統性的反向。
    """
    keys = list(ood_unc.keys())
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(4.1 * n, 7.4))
    axes = np.atleast_2d(axes)

    for j, k in enumerate(keys):
        ax = axes[0, j]
        in_unc = in_unc_by_ood[k]
        conf_in = conf_in_by_ood[k]
        ei, eo = in_unc[method]["epistemic"], ood_unc[k][method]["epistemic"]
        hi = max(np.percentile(ei, 99.5), np.percentile(eo, 99.5))
        bins = np.linspace(0, hi, 45)
        ax.hist(ei, bins=bins, color=C_IN, alpha=0.62, density=True, label="in-distribution", lw=0)
        ax.hist(eo, bins=bins, color=C_OOD, alpha=0.62, density=True, label="OOD", lw=0)
        ax.axvline(np.median(ei), color=C_IN, lw=2.0, ls="--")
        ax.axvline(np.median(eo), color=C_OOD, lw=2.0, ls="--")
        # log y：epistemic 的分佈在 0 附近有巨大尖峰（大部分樣本毫無異議），
        # 線性軸會把尖峰畫成一根柱子、把決定 AUROC 的尾部壓成看不見的一條線。
        # 而 OOD 偵測的成敗完全發生在尾部。
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2)
        a = aurocs.get(f"{method}/epistemic", {}).get(k, float("nan"))
        shift = np.median(eo) - np.median(ei)
        ok = "✓" if a >= 0.8 else ("~" if a >= 0.65 else "✗")
        ax.set_title(f"{OOD_LAB.get(k, k)}\nAUROC = {a:.3f}  {ok}", fontsize=10)
        ax.set_xlabel("Epistemic uncertainty (nats)")
        if j == 0:
            ax.set_ylabel(f"Density — {MLAB[method]}")
            ax.legend(fontsize=9, loc="upper right")
        ax.annotate(f"median shift\n{shift:+.4f}", xy=(0.97, 0.62), xycoords="axes fraction",
                    ha="right", fontsize=8.6, color=C_REF,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_REF, alpha=0.9))

        ax2 = axes[1, j]
        ci, co = conf_in[method], conf_ood[k][method]
        bins2 = np.linspace(min(ci.min(), co.min()), 1.0, 45)
        ax2.hist(ci, bins=bins2, color=C_IN, alpha=0.62, density=True, lw=0,
                 label=f"in-dist (mean {ci.mean():.3f})")
        ax2.hist(co, bins=bins2, color=C_OOD, alpha=0.62, density=True, lw=0,
                 label=f"OOD (mean {co.mean():.3f})")
        more_conf = co.mean() > ci.mean()
        ax2.set_xlabel("Max softmax probability (confidence)")
        if j == 0:
            ax2.set_ylabel("Density")
        ax2.legend(fontsize=8.4, loc="upper left")
        if more_conf:
            ax2.set_title("⚠ MORE confident on OOD than in-dist", fontsize=9.6, color=C_OOD)
        else:
            ax2.set_title("less confident on OOD (as hoped)", fontsize=9.6, color=C_OK)

    fig.suptitle("Does OOD data actually produce higher epistemic uncertainty?",
                 fontsize=12.5, y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 2：AUROC 矩陣 + trivial baseline 對照 ───────────────────────────
def auroc_matrix(aurocs, trivial, path):
    """左：方法 × OOD 來源的 AUROC。右：最佳模型方法 vs trivial 像素 baseline。

    右圖是本專案最重要的對照：如果一行 numpy（影像 L2 norm）就打敗了
    MC Dropout，那「貝葉斯不確定性偵測到了 OOD」必須大幅打折。
    """
    ood_keys = list(next(iter(aurocs.values())).keys())
    combos = [k for k in aurocs if not np.all(np.isnan(list(aurocs[k].values())))]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.4, 5.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    mat = np.array([[aurocs[c].get(k, np.nan) for k in ood_keys] for c in combos])
    im = axL.imshow(mat, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    axL.set_xticks(range(len(ood_keys)))
    axL.set_xticklabels([OOD_SHORT.get(k, k) for k in ood_keys], fontsize=9)
    axL.set_yticks(range(len(combos)))
    axL.set_yticklabels(combos, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                axL.text(j, i, "—", ha="center", va="center", fontsize=10, color=C_REF)
                continue
            axL.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9,
                     color="black", fontweight="bold" if mat[i, j] >= 0.8 else "normal")
    axL.set_title("OOD detection AUROC: method / score × OOD source\n"
                  "(0.5 = no better than chance)")
    axL.grid(False)
    fig.colorbar(im, ax=axL, fraction=0.036, label="AUROC")

    x = np.arange(len(ood_keys))
    w = 0.26
    best_model = []
    for k in ood_keys:
        vals = [aurocs[c][k] for c in combos if not np.isnan(aurocs[c].get(k, np.nan))]
        best_model.append(max(vals) if vals else np.nan)
    triv_best = [trivial[k]["best"][1] for k in ood_keys]
    triv_name = [trivial[k]["best"][0].replace("pixel_", "") for k in ood_keys]

    axR.bar(x - w / 2, best_model, w, color=C_IN, label="best model-based score")
    axR.bar(x + w / 2, triv_best, w, color=C_TRIVIAL,
            label="best trivial pixel statistic")
    axR.axhline(0.5, color=C_REF, ls="--", lw=1.5, label="chance")
    for xi, (bm, tb, tn) in enumerate(zip(best_model, triv_best, triv_name)):
        axR.annotate(f"{bm:.3f}", (xi - w / 2, bm), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8.6)
        axR.annotate(f"{tb:.3f}\n({tn})", (xi + w / 2, tb), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8.2, color="#8a4a12")
        if tb > bm:
            axR.annotate("trivial\nwins", (xi, max(bm, tb) + 0.06), ha="center",
                         fontsize=8.4, color=C_OOD, fontweight="bold")
    axR.set_xticks(x)
    axR.set_xticklabels([OOD_SHORT.get(k, k) for k in ood_keys], fontsize=9)
    axR.set_ylabel("AUROC")
    axR.set_ylim(0.3, 1.15)
    axR.set_title("Honesty check: does the model beat\na one-line pixel statistic?")
    axR.legend(fontsize=8.8, loc="lower left")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 3：分解是否真的分離兩種不確定性 ──────────────────────────────────
def decomposition_validation(res, path):
    """左：加噪聲時兩個分量的軌跡。右：噪聲 vs 換分佈的選擇性。

    這張圖回答「aleatoric/epistemic 的分解名符其實嗎」——
    大部分作品只計算它，從不驗證。
    """
    rows = res["rows"]
    sel = res["selectivity"]
    methods = [m for m in ("mcdrop", "ensemble") if m in sel]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.6))

    for m in methods:
        nr = sorted([r for r in rows if r["method"] == m and r["perturbation"] == "noise"],
                    key=lambda r: r["level"])
        lv = [r["level"] for r in nr]
        axL.plot(lv, [r["aleatoric"] for r in nr], "o-", color=MCOL[m], lw=2.0, ms=7,
                 label=f"{MLAB[m]} — aleatoric")
        axL.plot(lv, [r["epistemic"] for r in nr], "s--", color=MCOL[m], lw=1.8, ms=6,
                 alpha=0.65, label=f"{MLAB[m]} — epistemic")
    ar = sorted([r for r in rows if r["method"] == methods[0] and r["perturbation"] == "noise"],
                key=lambda r: r["level"])
    ax2 = axL.twinx()
    ax2.plot([r["level"] for r in ar], [r["accuracy"] for r in ar], ":", color=C_REF, lw=1.8)
    ax2.set_ylabel("Accuracy (dotted grey)", color=C_REF, fontsize=9.5)
    ax2.grid(False)

    # 標示非單調反轉：σ 過大時 aleatoric 下降而準確率繼續惡化 → 自信地錯誤
    s0 = sel[methods[0]]
    if s0.get("has_reversal"):
        cut = s0["noise_level_used"]
        axL.axvspan(cut, s0["max_level_tested"], color=C_OOD, alpha=0.08, lw=0)
        axL.annotate(
            f"σ > {cut:g}: uncertainty FALLS\n"
            f"while accuracy keeps dropping\n"
            f"({s0['noise_accuracy_at_used']:.3f} → {s0['accuracy_at_max_level']:.3f})\n"
            f"→ confidently wrong: saturation\n"
            f"   pushes images off-distribution",
            xy=(0.985, 0.42), xycoords="axes fraction", ha="right", va="center",
            fontsize=8.0, color=C_OOD,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_OOD, alpha=0.9))
        axL.axvline(cut, color=C_OOD, ls="--", lw=1.2, alpha=0.7)

    axL.set_xlabel("Gaussian noise σ added to in-distribution images")
    axL.set_ylabel("Uncertainty (nats)")
    axL.set_title("Adding noise should raise aleatoric, not epistemic\n"
                  "(selectivity measured at the aleatoric peak, before reversal)")
    axL.legend(fontsize=8.6, loc="upper left")

    x = np.arange(len(methods))
    w = 0.35
    ns = [sel[m]["noise_selectivity"] for m in methods]
    ss = [sel[m]["shift_selectivity"] for m in methods]
    axR.bar(x - w / 2, ns, w, color=C_OK, label=r"noise: $\Delta$ale / $\Delta$epi")
    axR.bar(x + w / 2, ss, w, color=C_BAD, label=r"shift: $\Delta$epi / $\Delta$ale")
    axR.axhline(1.0, color=C_REF, ls="--", lw=1.8)
    axR.annotate("ratio = 1: the perturbation raises\nboth components equally\n"
                 "(no selectivity)", xy=(0.5, 1.0), xycoords=("axes fraction", "data"),
                 xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8.4,
                 color=C_REF)
    for xi, (a, b) in enumerate(zip(ns, ss)):
        axR.annotate(f"{a:.2f}", (xi - w / 2, a), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=9)
        axR.annotate(f"{b:.2f}", (xi + w / 2, b), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=9,
                     color=C_OOD if b < 1 else "black",
                     fontweight="bold" if b < 1 else "normal")
    axR.set_xticks(x)
    axR.set_xticklabels([MLAB[m] for m in methods], fontsize=9.5)
    axR.set_ylabel("Selectivity ratio  (>1 = decomposition behaves as advertised)")
    axR.set_title("Distribution shift raises ALEATORIC more than\nepistemic — the opposite of the textbook story")
    axR.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 4：risk–coverage（接 A1 的決策層）────────────────────────────────
def risk_coverage(sel, path, highlight=("ensemble/epistemic", "mcdrop/epistemic",
                                        "single/neg_max_prob")):
    """左：risk–coverage 曲線。右：放棄 20% 後的準確率提升（通關標準 3）。"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.6),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    curves = sel["curves"]
    keys = [k for k in highlight if k in curves] or list(curves)[:3]

    for k in keys:
        rc = curves[k]
        m = k.split("/")[0]
        ls = "-" if "epistemic" in k else "--"
        axL.plot(rc["coverage"], rc["accuracy"], ls, lw=2.1, color=MCOL[m],
                 label=f"{k}  (AURC {rc['aurc']:.4f})")
    m0 = keys[0].split("/")[0]
    axL.axhline(sel["accuracy"][m0], color=C_REF, ls=":", lw=1.6,
                label=f"full-coverage accuracy = {sel['accuracy'][m0]:.4f}")
    axL.axvline(0.8, color=C_OOD, lw=1.2, alpha=0.5)
    axL.set_xlabel("Coverage (fraction of cases the model keeps)")
    axL.set_ylabel("Accuracy on kept cases")
    axL.set_title("Risk–coverage: defer the most uncertain cases\nto a human reader")
    axL.legend(fontsize=8.6, loc="lower left")
    axL.invert_xaxis()

    labels, gains, base = [], [], []
    for k in keys:
        rc = curves[k]
        labels.append(k.replace("/", "\n"))
        gains.append(rc["acc_at_80"] - rc["acc_at_100"])
        base.append(rc["acc_at_100"])
    x = np.arange(len(labels))
    cols = [MCOL[k.split("/")[0]] for k in keys]
    axR.bar(x, gains, 0.55, color=cols)
    for xi, (g, b, k) in enumerate(zip(gains, base, keys)):
        rc = curves[k]
        axR.annotate(f"{rc['acc_at_100']:.3f} → {rc['acc_at_80']:.3f}\n({g:+.3f})",
                     (xi, g), textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=8.8)
    axR.axhline(0, color=C_REF, lw=1.2)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=8.6)
    axR.set_ylabel("Accuracy gain from deferring the most uncertain 20%")
    axR.set_title("Pass criterion 3: does deferring 20%\nactually help?")
    oracle = sel["oracle_aurc"].get(m0)
    if oracle is not None:
        axR.annotate(f"oracle AURC (perfect ranking) = {oracle:.4f}",
                     xy=(0.03, 0.94), xycoords="axes fraction", fontsize=8.6, color=C_REF)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 5：T 與 M 的成本效益 ────────────────────────────────────────────
def cost_benefit(res, path):
    rows = res["rows"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for ax, kind, xlab, col in (
        (axL, "mcdrop", "MC Dropout forward passes $T$ (inference cost)", MCOL["mcdrop"]),
        (axR, "ensemble", "Deep Ensemble size $M$ (training + inference cost)", MCOL["ensemble"]),
    ):
        r = sorted([x for x in rows if x["kind"] == kind], key=lambda z: z["budget"])
        b = [x["budget"] for x in r]
        ax.plot(b, [x["auroc_epistemic"] for x in r], "o-", color=col, lw=2.1, ms=7,
                label="AUROC (epistemic)")
        ax.plot(b, [x["auroc_total"] for x in r], "s--", color=col, alpha=0.6, lw=1.8, ms=6,
                label="AUROC (total entropy)")
        ax.axhline(0.5, color=C_REF, ls=":", lw=1.5, label="chance")
        ax.set_xlabel(xlab)
        ax.set_ylabel("OOD detection AUROC")
        ax.set_xscale("log")
        ax.set_xticks(b)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=9, loc="lower right")
    axL.set_title("How many MC samples are actually needed?")
    axR.set_title("How many ensemble members are worth training?")
    fig.suptitle(f"Cost–benefit of the uncertainty budget (OOD = {res['ood_set']})",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 6：校準與準確率 ─────────────────────────────────────────────────
def calibration(ece_res, acc, path):
    """可靠度圖 + ECE / 準確率對照。

    放進來是因為 OOD 偵測與校準是**不同**的事：模型可以在
    in-distribution 上校準良好卻完全偵測不到 OOD（本專案的 Fashion 組正是如此）。
    """
    methods = list(ece_res.keys())
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    axL.plot([0, 1], [0, 1], ls="--", color=C_REF, lw=1.8, label="perfect calibration")
    for m in methods:
        bins = [b for b in ece_res[m]["bins"] if b["n"] > 0]
        axL.plot([b["conf"] for b in bins], [b["acc"] for b in bins], "o-",
                 color=MCOL[m], lw=2.0, ms=6,
                 label=f"{MLAB[m]} (ECE {ece_res[m]['ece']:.4f})")
    axL.set_xlabel("Predicted confidence")
    axL.set_ylabel("Empirical accuracy")
    axL.set_title("Reliability diagram (in-distribution test set)")
    axL.legend(fontsize=9, loc="upper left")

    x = np.arange(len(methods))
    w = 0.35
    axR.bar(x - w / 2, [ece_res[m]["ece"] for m in methods], w,
            color=[MCOL[m] for m in methods], label="ECE (lower better)")
    ax2 = axR.twinx()
    ax2.plot(x, [acc[m] for m in methods], "D--", color=C_REF, ms=8, lw=1.6,
             label="accuracy")
    ax2.set_ylabel("Accuracy", color=C_REF)
    ax2.grid(False)
    for xi, m in enumerate(methods):
        axR.annotate(f"{ece_res[m]['ece']:.4f}", (xi - w / 2, ece_res[m]["ece"]),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
        ax2.annotate(f"{acc[m]:.4f}", (xi, acc[m]), textcoords="offset points",
                     xytext=(6, 4), fontsize=8.6, color=C_REF)
    axR.set_xticks(x)
    axR.set_xticklabels([MLAB[m] for m in methods], fontsize=9)
    axR.set_ylabel("Expected Calibration Error")
    axR.set_title("Calibration and accuracy are separate\nfrom OOD detection")
    axR.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 7：範例影像 —— 讓 OOD 的性質一眼可見 ────────────────────────────
def sample_grid(sets: dict, path, n_per: int = 8, seed: int = 0):
    """每個資料集抽 n_per 張影像排一列，讓讀者自己看出「這些 OOD 有多不同」。

    這張圖不是裝飾：Fashion 組的失敗與 Derma 組的顏色捷徑，
    看過影像之後就非常直觀 —— 前者的像素統計與胸片天差地遠，
    後者一眼就能看出「有顏色」。
    """
    rng = np.random.default_rng(seed)
    names = list(sets.keys())
    fig, axes = plt.subplots(len(names), n_per, figsize=(1.28 * n_per, 1.42 * len(names)))
    axes = np.atleast_2d(axes)
    for i, nm in enumerate(names):
        s = sets[nm]
        idx = rng.choice(s.n if hasattr(s, "n") else len(s["x"]), n_per, replace=False)
        x = (s.x if hasattr(s, "x") else s["x"])[idx]
        for j in range(n_per):
            img = np.transpose(x[j], (1, 2, 0)) * 0.5 + 0.5
            axes[i, j].imshow(np.clip(img, 0, 1))
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(nm, fontsize=8)
        axes[i, 0].axis("on")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for sp in axes[i, 0].spines.values():
            sp.set_visible(False)
        axes[i, 0].set_ylabel(OOD_SHORT.get(nm, nm), fontsize=8.5, rotation=0,
                              ha="right", va="center", labelpad=8)
    fig.suptitle("In-distribution vs the three OOD sources", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
