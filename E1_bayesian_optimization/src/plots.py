"""
E1 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只吃 numpy / dict。

沿用 A2/A3/B2 的原則：不接觸 GP 或 BO 物件，所有資料由 run_all 落盤後傳入。
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

C_RAND = "#6c757d"
C_EI = "#2e6fdb"
C_UCB = "#2a9d8f"
C_TS = "#e8873a"
C_REF = "#22223b"
C_BAD = "#d1495b"
C_OK = "#2a9d8f"

MCOL = {"random": C_RAND, "bo/ei": C_EI, "bo/ucb": C_UCB, "bo/thompson": C_TS}
MLAB = {"random": "Random search", "bo/ei": "BO — Expected Improvement",
        "bo/ucb": "BO — UCB (κ=2)", "bo/thompson": "BO — Thompson sampling"}
KCOL = {"RBF": "#9b5de5", "Matern52": "#2a9d8f"}


def _band(ax, x, mean, sd, color, label=None, alpha=0.18, log_floor=None):
    """mean ± 1 sd 的誤差帶。

    `log_floor`：在 log y 軸上使用時必須傳。原因是 `mean − sd` 常常是負值
    （BO 收斂後 mean 接近 0 而 sd 仍有量級），而 log 軸畫不出非正值 ——
    matplotlib 會把整條下緣塌到軸底，產生一片巨大陰影把所有曲線遮住。
    截斷下緣是視覺處理，不改變 mean 線本身（那才是報告的數字）。
    """
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    lo = mean - sd
    if log_floor is not None:
        lo = np.maximum(lo, log_floor)
    ax.plot(x, mean, color=color, lw=2.1, label=label)
    ax.fill_between(x, lo, mean + sd, color=color, alpha=alpha, lw=0)


def _log_floor_for(series_list, frac: float = 0.35) -> float:
    """為 log 軸挑一個誤差帶下限：取所有 mean 曲線最小正值的一個比例。"""
    mins = []
    for m in series_list:
        m = np.asarray(m, dtype=float)
        pos = m[m > 0]
        if len(pos):
            mins.append(pos.min())
    return (min(mins) * frac) if mins else 1e-3


# ── 圖 1：手刻 GP 的驗證 + 後驗示意 ─────────────────────────────────────
def gp_verification(demo, checks, grad_checks, path):
    """左：1D GP 後驗（資料 + μ ± 2σ + 樣本）。右：與 sklearn / 數值梯度的誤差。

    左圖是「高斯共軛在函數空間」的視覺化：觀測點處不確定性被壓縮到接近 0，
    遠離觀測處回到先驗寬度。右圖證明手刻的實作是對的，不是看起來對。
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    xs, mu, sd = demo["xs"], demo["mu"], demo["sd"]
    axL.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=C_EI, alpha=0.18, lw=0,
                     label="posterior μ ± 2σ")
    for i, s in enumerate(demo["samples"]):
        axL.plot(xs, s, color=C_EI, lw=0.9, alpha=0.45,
                 label="posterior samples" if i == 0 else None)
    axL.plot(xs, mu, color=C_EI, lw=2.3, label="posterior mean μ(x)")
    axL.plot(xs, demo["truth"], color=C_REF, ls="--", lw=1.8, label="true function")
    axL.plot(demo["X"], demo["y"], "o", color=C_BAD, ms=9, zorder=5,
             label="observations")
    axL.set_xlabel("x")
    axL.set_ylabel("f(x)")
    axL.set_title("Hand-coded GP posterior: Gaussian conjugacy\nin function space "
                  f"({demo['kernel']}, ℓ={demo['lengthscale']:.2f})")
    axL.legend(fontsize=8.8, loc="upper left", ncol=2)

    labels, vals, cols = [], [], []
    for c in checks:
        labels.append(f"{c['kernel']}\nvs sklearn\n(mean)")
        vals.append(max(c["max_rel_diff_mean"], 1e-17))
        cols.append(C_OK if c["passed"] else C_BAD)
        labels.append(f"{c['kernel']}\nvs sklearn\n(std)")
        vals.append(max(c["max_rel_diff_std"], 1e-17))
        cols.append(C_OK if c["passed"] else C_BAD)
    for g in grad_checks:
        labels.append(f"{g['kernel']}\ngradient\nvs finite diff")
        vals.append(max(g["max_rel_err"], 1e-17))
        cols.append(C_OK if g["passed"] else C_BAD)

    x = np.arange(len(labels))
    axR.bar(x, vals, 0.6, color=cols)
    axR.axhline(1e-6, color=C_BAD, ls="--", lw=1.6, label="tolerance 1e−6")
    axR.set_yscale("log")
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=7.6)
    axR.set_ylabel("Max relative error (log scale)")
    axR.set_title("Correctness checks: posterior formulas\nand analytic gradients")
    axR.legend(fontsize=9)
    for xi, v in zip(x, vals):
        axR.annotate(f"{v:.1e}", (xi, v), textcoords="offset points", xytext=(0, 4),
                     ha="center", fontsize=7.8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 2：收斂曲線（通關標準 1）─────────────────────────────────────────
def convergence(benchmarks, path):
    """每個目標函數一個 panel：BO 三種 acquisition vs 隨機搜尋，30 seeds 的 mean±sd。"""
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.2))
    axes = np.atleast_1d(axes)
    for ax, bm in zip(axes, benchmarks):
        t = np.arange(1, bm["n_total"] + 1)
        use_log = bm["objective"] == "Branin"
        methods = [m for m in ("random", "bo/ei", "bo/ucb", "bo/thompson")
                   if m in bm["methods"]]
        floor = (_log_floor_for([bm["methods"][m]["mean"] for m in methods])
                 if use_log else None)
        if use_log and bm["f_min"] is not None:
            floor = min(floor, bm["f_min"] * 0.9)
        for m in methods:
            a = bm["methods"][m]
            _band(ax, t, a["mean"], a["sd"], MCOL[m], MLAB[m], log_floor=floor)
        if bm["f_min"] is not None:
            ax.axhline(bm["f_min"], color=C_REF, ls=":", lw=1.8,
                       label=f"global min = {bm['f_min']:.4f}")
        ax.axvline(bm["n_init"], color=C_REF, lw=1.0, alpha=0.35)
        ax.set_xlabel("Number of function evaluations")
        ax.set_title(f"{bm['objective']}  ({bm['dim']}D)\n"
                     f"{bm['methods']['random']['n_seeds']} seeds, mean ± 1 sd")
        if use_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor)
        ax.annotate("initial\ndesign", xy=(bm["n_init"], 0.97), xycoords=("data", "axes fraction"),
                    fontsize=7.6, color=C_REF, ha="left", va="top",
                    xytext=(3, 0), textcoords="offset points")
        ax.legend(fontsize=8.4, loc="upper right")
    axes[0].set_ylabel("Best value found so far (lower is better)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 3：κ 敏感度 + 卡局部最優的軌跡（通關標準 2）─────────────────────
def kappa_effect(sweep, traj, path):
    """上排：2D 取樣軌跡（κ=0 / 2 / 20）。下左：收斂曲線。下右：探索行為量化。"""
    ks = list(traj["trajectories"].keys())
    fig = plt.figure(figsize=(5.0 * len(ks), 9.6))
    gs = fig.add_gridspec(2, len(ks), height_ratios=[1.05, 1])

    g = traj["grid_x"]
    Z = np.asarray(traj["Z"])
    for j, kname in enumerate(ks):
        ax = fig.add_subplot(gs[0, j])
        cs = ax.contourf(g, g, np.log10(Z - Z.min() + 1e-3), levels=28, cmap="viridis_r")
        tr = traj["trajectories"][kname]
        X = np.asarray(tr["X"])
        n_init = traj["n_init"]
        ax.plot(X[:n_init, 0], X[:n_init, 1], "s", color="white", ms=7, mec="black",
                mew=1.0, label="initial design")
        ax.plot(X[n_init:, 0], X[n_init:, 1], "-o", color=C_BAD, ms=5, lw=0.9,
                alpha=0.9, label="BO trajectory")
        if traj["x_min"] is not None:
            xm = np.asarray(traj["x_min"])
            ax.plot(xm[0], xm[1], "*", color="#ffd166", ms=20, mec="black", mew=0.8,
                    label="global optimum")
        ax.set_title(f"{kname}   best = {tr['best_value']:.4f}\n"
                     f"mean distance to best point = {tr['explore_radius']:.3f}",
                     fontsize=10)
        ax.set_xlabel("$x_1$ (normalised)")
        if j == 0:
            ax.set_ylabel("$x_2$ (normalised)")
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    axL = fig.add_subplot(gs[1, 0])
    t = np.arange(1, sweep["n_total"] + 1)
    cmap = plt.get_cmap("coolwarm")
    names = list(sweep["runs"].keys())
    floor = _log_floor_for([sweep["runs"][k]["mean"] for k in names] +
                           [sweep["random"]["mean"]])
    for i, kname in enumerate(names):
        a = sweep["runs"][kname]
        col = cmap(i / max(len(names) - 1, 1))
        _band(axL, t, a["mean"], a["sd"], col, kname, alpha=0.12, log_floor=floor)
    ar = sweep["random"]
    _band(axL, t, ar["mean"], ar["sd"], C_RAND, "Random search", alpha=0.12,
          log_floor=floor)
    axL.set_yscale("log")
    axL.set_ylim(bottom=floor)
    axL.set_xlabel("Number of function evaluations")
    axL.set_ylabel("Best value so far")
    axL.set_title(f"κ sweep on {sweep['objective']} "
                  f"({sweep['random']['n_seeds']} seeds)")
    axL.legend(fontsize=8.4, ncol=2)

    axR = fig.add_subplot(gs[1, 1:])
    kv = [float(k.split("=")[1]) for k in names]
    finals = [sweep["runs"][k]["final_mean"] for k in names]
    fsd = [sweep["runs"][k]["final_sd"] for k in names]
    radii = [sweep["runs"][k]["explore_radius_mean"] for k in names]
    ax2 = axR.twinx()
    axR.errorbar(range(len(kv)), finals, yerr=fsd, fmt="o-", color=C_EI, ms=9,
                 lw=2.0, capsize=5, label="final best value (left axis)")
    axR.axhline(sweep["random"]["final_mean"], color=C_RAND, ls="--", lw=1.8,
                label="random search level")
    ax2.plot(range(len(kv)), radii, "s--", color=C_BAD, ms=8, lw=1.8,
             label="exploration radius (right axis)")
    ax2.set_ylabel("Mean distance from best point", color=C_BAD)
    ax2.grid(False)
    axR.set_xticks(range(len(kv)))
    axR.set_xticklabels([f"κ={k:g}" for k in kv])
    axR.set_ylabel("Final best value (lower better)")
    axR.set_title("Exploitation-only (κ=0) gets stuck;\nκ=20 degenerates toward random")
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=8.6, loc="upper center")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 4：維度詛咒（通關標準 3）─────────────────────────────────────────
def dimension_curse(dc, path):
    """左：各維度 BO vs random 的最終值。右：配對優勢 ± 95% 區間，找臨界維度。"""
    dims = dc["dims"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.4))

    bo_m = [dc["per_dim"][str(d)]["bo"]["final_mean"] for d in dims]
    bo_s = [dc["per_dim"][str(d)]["bo"]["final_sd"] for d in dims]
    rd_m = [dc["per_dim"][str(d)]["random"]["final_mean"] for d in dims]
    rd_s = [dc["per_dim"][str(d)]["random"]["final_sd"] for d in dims]
    x = np.arange(len(dims))
    axL.errorbar(x - 0.06, bo_m, yerr=bo_s, fmt="o-", color=C_EI, ms=9, lw=2.0,
                 capsize=5, label="BO (EI)")
    axL.errorbar(x + 0.06, rd_m, yerr=rd_s, fmt="s--", color=C_RAND, ms=8, lw=2.0,
                 capsize=5, label="Random search")
    axL.set_xticks(x)
    axL.set_xticklabels([f"{d}D" for d in dims])
    axL.set_xlabel("Ackley dimension")
    axL.set_ylabel("Final best value (lower is better)")
    axL.set_title(f"Fixed budget of {dc['n_total']} evaluations\n"
                  f"({dc['per_dim'][str(dims[0])]['bo']['n_seeds']} seeds, mean ± 1 sd)")
    axL.legend(fontsize=9.5)

    gain = [dc["per_dim"][str(d)]["paired_gain_mean"] for d in dims]
    se = [dc["per_dim"][str(d)]["paired_gain_se"] for d in dims]
    wins = [dc["per_dim"][str(d)]["bo_wins_frac"] for d in dims]
    cols = [C_OK if g - 1.96 * s > 0 else C_BAD for g, s in zip(gain, se)]
    axR.bar(x, gain, 0.55, yerr=[1.96 * s for s in se], capsize=6, color=cols)
    axR.axhline(0, color=C_REF, lw=1.6)
    for xi, (g, s, w) in enumerate(zip(gain, se, wins)):
        axR.annotate(f"{g:+.2f}\nwins {100 * w:.0f}%", (xi, g),
                     textcoords="offset points",
                     xytext=(0, 8 if g >= 0 else -22), ha="center", fontsize=8.6)
    axR.set_xticks(x)
    axR.set_xticklabels([f"{d}D" for d in dims])
    axR.set_xlabel("Ackley dimension")
    axR.set_ylabel("Paired advantage of BO over random\n(same seed, >0 = BO better)")
    crit = dc.get("critical_dim")
    axR.set_title("Paired comparison with 95% intervals\n" +
                  (f"critical dimension ≈ {crit}D (interval covers 0)"
                   if crit else "BO wins at every dimension tested"))
    if crit is not None and crit in dims:
        axR.axvline(dims.index(crit), color=C_BAD, ls="--", lw=1.6, alpha=0.7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 5：維度詛咒的機制 ───────────────────────────────────────────────
def curse_mechanism(conc, shortfall, path):
    """左：核值如何隨維度集中到 0。右：後驗解釋力與 EI 崩潰，但內層優化沒失敗。"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.4))

    rows = conc["rows"]
    dims = [r["dim"] for r in rows]
    x = np.arange(len(dims))
    axL.plot(x, [r["frac_above_0.1"] for r in rows], "o-", color=C_EI, ms=9, lw=2.1,
             label="fraction of pairs with k > 0.1  (“effective neighbours”)")
    axL.plot(x, [r["frac_above_0.01"] for r in rows], "s--", color=C_UCB, ms=8, lw=1.8,
             label="fraction with k > 0.01")
    ax2 = axL.twinx()
    ax2.plot(x, [r["dist_rel_sd"] for r in rows], "^:", color=C_BAD, ms=8, lw=1.8,
             label="relative sd of pairwise distance")
    ax2.set_ylabel("Relative sd of distances (distance concentration)", color=C_BAD,
                   fontsize=9.5)
    ax2.grid(False)
    axL.set_xticks(x)
    axL.set_xticklabels([f"{d}D" for d in dims])
    axL.set_xlabel("Dimension")
    axL.set_ylabel("Fraction of point pairs that “see” each other")
    axL.set_title(f"Mechanism 1: kernel values collapse\n({conc['kernel']}, ℓ={conc['lengthscale']}, "
                  f"{conc['n_points']} uniform points)")
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, fontsize=8.2, loc="upper right")
    for xi, r in zip(x, rows):
        axL.annotate(f"k̃={r['kernel_median']:.1e}", (xi, r["frac_above_0.1"]),
                     textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.6)

    sd = shortfall["per_dim"]
    dims2 = [e["dim"] for e in sd]
    x2 = np.arange(len(dims2))
    axR.plot(x2, [e["mu_spread_over_yrange_mean"] for e in sd], "o-", color=C_EI,
             ms=9, lw=2.1, label="posterior mean spread ÷ observed y range")
    axR.plot(x2, [e["best_ei_over_yrange_mean"] for e in sd], "s-", color=C_UCB,
             ms=8, lw=2.0, label="best EI ÷ observed y range")
    axR.plot(x2, [abs(e["shortfall_at_smallest_mean"]) /
                  max(e["y_range_observed_mean"], 1e-12) for e in sd],
             "^--", color=C_RAND, ms=8, lw=1.8,
             label="inner-optimiser shortfall ÷ y range  (≈0 ⇒ NOT the problem)")
    axR.set_yscale("log")
    axR.set_xticks(x2)
    axR.set_xticklabels([f"{d}D" for d in dims2])
    axR.set_xlabel("Ackley dimension")
    axR.set_ylabel("Normalised magnitude (log scale)")
    axR.set_title("Mechanism 2: the GP degenerates to its prior —\n"
                  "the inner optimiser is NOT what fails")
    axR.legend(fontsize=8.2, loc="lower left")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 6：核函數就是先驗（通關標準 4）───────────────────────────────────
def kernel_comparison(kc, path):
    """左：收斂曲線對照。右：學到的 lengthscale 與配對差異。"""
    objs = list(kc["per_objective"].keys())
    fig, axes = plt.subplots(1, len(objs) + 1, figsize=(5.2 * (len(objs) + 1), 5.2))
    axes = np.atleast_1d(axes)

    for ax, oname in zip(axes[:-1], objs):
        e = kc["per_objective"][oname]
        t = np.arange(1, kc["n_total"] + 1)
        use_log = oname == "Branin"
        floor = (_log_floor_for([e["kernels"][k]["mean"] for k in kc["kernels"]])
                 if use_log else None)
        if use_log and e["f_min"] is not None:
            floor = min(floor, e["f_min"] * 0.9)
        for kname in kc["kernels"]:
            a = e["kernels"][kname]
            _band(ax, t, a["mean"], a["sd"], KCOL[kname],
                  f"{kname}  (ℓ={a['final_lengthscale_mean']:.3f})", log_floor=floor)
        if e["f_min"] is not None:
            ax.axhline(e["f_min"], color=C_REF, ls=":", lw=1.8, label="global min")
        ax.set_xlabel("Number of function evaluations")
        ax.set_title(f"{oname}  ({e['dim']}D)\nbetter: {e['better']}")
        if use_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor)
        ax.legend(fontsize=8.6)
    axes[0].set_ylabel("Best value so far")

    axR = axes[-1]
    x = np.arange(len(objs))
    w = 0.35
    for i, kname in enumerate(kc["kernels"]):
        ls = [kc["per_objective"][o]["kernels"][kname]["final_lengthscale_mean"]
              for o in objs]
        lsd = [kc["per_objective"][o]["kernels"][kname]["final_lengthscale_sd"]
               for o in objs]
        axR.bar(x + (i - 0.5) * w, ls, w, yerr=lsd, capsize=5, color=KCOL[kname],
                label=kname)
    axR.set_xticks(x)
    axR.set_xticklabels(objs, fontsize=9.5)
    axR.set_ylabel("Learned lengthscale ℓ (normalised domain)")
    axR.set_title("The kernel IS the prior:\nsmoothness assumption changes what is learned")
    axR.legend(fontsize=9)
    for xi, o in enumerate(objs):
        e = kc["per_objective"][o]
        axR.annotate(f"paired Δ = {e['paired_diff_mean']:+.4f}\n± {1.96 * e['paired_diff_se']:.4f}",
                     (xi, 0.02), xycoords=("data", "axes fraction"), ha="center",
                     fontsize=8.2, color=C_REF)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 7：真實應用 —— 混凝土配方 ───────────────────────────────────────
def concrete(ca, path):
    """左：強度收斂曲線（BO vs random）。右：省下的實驗次數與最佳配方。"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    t = np.arange(1, ca["n_total"] + 1)
    for key, col, lab in (("bo", C_EI, "BO (Expected Improvement)"),
                          ("random", C_RAND, "Random search")):
        a = ca[key]
        # 內部是最小化 (−strength)，畫圖轉回強度
        mean = -np.asarray(a["mean"])
        sd = np.asarray(a["sd"])
        _band(axL, t, mean, sd, col, f"{lab}  (final {mean[-1]:.2f} MPa)")
    axL.axhline(ca["surrogate"]["best_in_data"], color=C_REF, ls=":", lw=1.8,
                label=f"best in dataset = {ca['surrogate']['best_in_data']:.1f} MPa")
    n_bo = ca["evals_bo_to_match_random_final"]
    if n_bo:
        axL.axvline(n_bo, color=C_OK, ls="--", lw=1.8)
        axL.annotate(f"BO reaches random's final level\nafter {n_bo} evaluations\n"
                     f"→ saves {ca['evals_saved']} of {ca['n_total']} "
                     f"({ca['saved_pct']:.0f}%)",
                     xy=(n_bo, -float(ca["random"]["mean"][-1])),
                     xytext=(10, -46), textcoords="offset points", fontsize=9,
                     color=C_OK,
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_OK, alpha=0.92))
    axL.set_xlabel("Number of experiments (surrogate queries)")
    axL.set_ylabel("Best compressive strength found (MPa)")
    axL.set_title(f"Concrete mix optimisation ({ca['bo']['n_seeds']} seeds, mean ± 1 sd)\n"
                  f"surrogate CV R² = {ca['surrogate']['surrogate_cv_r2_mean']:.3f}")
    axL.legend(fontsize=8.8, loc="lower right")

    recipe = ca["best_recipe"]
    names = list(recipe.keys())
    vals = np.array([recipe[k] for k in names], dtype=float)
    lo = np.asarray(ca["surrogate"]["domain_lo"], dtype=float)
    hi = np.asarray(ca["surrogate"]["domain_hi"], dtype=float)
    frac = (vals - lo) / np.maximum(hi - lo, 1e-12)
    y = np.arange(len(names))[::-1]
    axR.barh(y, frac, 0.6, color=C_EI, alpha=0.85)
    axR.set_yticks(y)
    axR.set_yticklabels(names, fontsize=9)
    axR.set_xlim(0, 1.28)
    axR.set_xlabel("Position within the searched range  (0 = 1st pct, 1 = 99th pct)")
    for yi, (nm, v, f) in zip(y, zip(names, vals, frac)):
        axR.annotate(f"{v:.1f}", (f, yi), textcoords="offset points", xytext=(5, -3),
                     fontsize=8.6)
    axR.set_title(f"Best recipe found: {ca['best_strength_found']:.1f} MPa\n"
                  f"(paired gain over random {ca['paired_gain_mean_mpa']:+.2f} MPa, "
                  f"BO wins {100 * ca['bo_wins_frac']:.0f}%)")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
