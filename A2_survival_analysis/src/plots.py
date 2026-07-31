"""
A2 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只負責畫，資料由 run_all 傳入。

沿用 B2 的設計原則：推論結果與出圖資料落盤分離，所以這裡的每個函式
都只吃 numpy 陣列與 dict，完全不接觸 PyMC —— 調圖不必重跑推論。
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

C_EVENT = "#d1495b"
C_CENS = "#2a9d8f"
C_BAYES = "#2e6fdb"
C_COX = "#e8873a"
C_KM = "#22223b"
C_REF = "#6c757d"
C_BAD1 = "#d1495b"
C_BAD2 = "#9b5de5"

MCOL = {"weibull": "#2e6fdb", "lognormal": "#2a9d8f", "exponential": "#e8873a"}
MLABEL = {"weibull": "Weibull", "lognormal": "Log-Normal", "exponential": "Exponential"}


def _band(ax, x, draws, color, label=None, alpha_hi=0.16, alpha_lo=0.30):
    """畫後驗帶：95% 與 50% 兩層 + 中位數線。"""
    lo95, lo50, med, hi50, hi95 = np.percentile(draws, [2.5, 25, 50, 75, 97.5], axis=0)
    ax.fill_between(x, lo95, hi95, color=color, alpha=alpha_hi, lw=0)
    ax.fill_between(x, lo50, hi50, color=color, alpha=alpha_lo, lw=0)
    ax.plot(x, med, color=color, lw=2.0, label=label)
    return med


# ── 圖 1：刪失似然的直覺 ────────────────────────────────────────────────
def censoring_likelihood(t, event, demo, path, n_show=28, seed=3):
    """左：病人時間軸（事件 vs 右刪失）。右：同一個觀察時間的兩種似然貢獻。

    demo: dict(t_grid, pdf, t_obs, pdf_at_t, sf_at_t) —— 一個 Weibull 的示範曲線。
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(t), n_show, replace=False))
    order = np.argsort(t[idx])
    idx = idx[order]
    for row, i in enumerate(idx):
        is_ev = event[i] == 1
        col = C_EVENT if is_ev else C_CENS
        axL.plot([0, t[i]], [row, row], color=col, lw=1.6, alpha=0.75, solid_capstyle="butt")
        if is_ev:
            axL.plot(t[i], row, "o", color=col, ms=6.5, zorder=3)
        else:
            axL.annotate("", xy=(t[i] + 0.42, row), xytext=(t[i], row),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
    axL.plot([], [], "o", color=C_EVENT, ms=6.5, label=r"event at $t_i$  →  contributes $f(t_i)$")
    axL.plot([], [], ">", color=C_CENS, ms=6.5,
             label=r"right-censored  →  contributes $S(t_i)$")
    axL.set_xlabel("Time since diagnosis (years)")
    axL.set_ylabel(f"Patients (random sample of {n_show})")
    axL.set_yticks([])
    axL.set_title("Censoring is not missing data:\nan arrow says \"survived at least this long\"")
    axL.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    tg, pdf = demo["t_grid"], demo["pdf"]
    t0 = demo["t_obs"]
    axR.plot(tg, pdf, color=C_REF, lw=2.0, label=r"Weibull density $f(t)$")
    m = tg >= t0
    axR.fill_between(tg[m], 0, pdf[m], color=C_CENS, alpha=0.30, lw=0,
                     label=rf"censored: $S({t0:g})={demo['sf_at_t']:.3f}$  (shaded area)")
    axR.vlines(t0, 0, demo["pdf_at_t"], color=C_EVENT, lw=2.4,
               label=rf"event: $f({t0:g})={demo['pdf_at_t']:.3f}$  (height)")
    axR.plot(t0, demo["pdf_at_t"], "o", color=C_EVENT, ms=7, zorder=3)
    axR.set_xlabel("Time (years)")
    axR.set_ylabel("Density")
    axR.set_title(f"Same observation at t={t0:g}, two likelihood terms\n"
                  "treating a censored case as an event puts all\nmass on one point instead of the whole tail")
    axR.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    axR.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 2：三種刪失處理的後果（通關標準 1）────────────────────────────────
def censoring_handling(km, curves, medians, path):
    """左：族群存活曲線 vs KM。右：中位存活時間的偏差。

    km: dict(timeline, survival, lo, hi)
    curves: {"correct"/"as_event"/"drop": (t_grid, draws)}
    medians: dict of {name: dict(median, lo, hi)} + {"km": float}
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    axL.step(km["timeline"], km["survival"], where="post", color=C_KM, lw=2.4,
             label="Kaplan–Meier (correct censoring)", zorder=5)
    axL.fill_between(km["timeline"], km["lo"], km["hi"], step="post",
                     color=C_KM, alpha=0.12, lw=0)

    style = {
        "correct": (C_BAYES, "Bayes, censoring handled: $S(t_i)$"),
        "as_event": (C_BAD1, "WRONG: censored treated as events"),
        "drop": (C_BAD2, "WRONG: censored rows dropped"),
    }
    for name, (tg, draws) in curves.items():
        col, lab = style[name]
        _band(axL, tg, draws, col, label=lab)

    axL.set_xlabel("Time since diagnosis (years)")
    axL.set_ylabel("Survival probability $S(t)$")
    axL.set_title("Both ways of mishandling censoring\nbias survival downward")
    axL.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    axL.set_ylim(0, 1.02)

    names = ["correct", "as_event", "drop"]
    labels = ["Censoring\nhandled", "Censored\n= events", "Censored\ndropped"]
    cols = [C_BAYES, C_BAD1, C_BAD2]
    ref = medians["km"]
    for i, (nm, lab, col) in enumerate(zip(names, labels, cols)):
        m = medians[nm]
        axR.errorbar(i, m["median"], yerr=[[m["median"] - m["lo"]], [m["hi"] - m["median"]]],
                     fmt="o", color=col, ms=10, capsize=6, lw=2.0)
        bias = 100 * (m["median"] - ref) / ref
        axR.annotate(f"{m['median']:.2f} yr\n({bias:+.0f}%)", (i, m["median"]),
                     textcoords="offset points", xytext=(20, -4), fontsize=10, color=col)
    axR.axhline(ref, color=C_KM, ls="--", lw=1.8, label=f"Kaplan–Meier median = {ref:.2f} yr")
    axR.set_xticks(range(3))
    axR.set_xticklabels(labels, fontsize=9.5)
    axR.set_xlim(-0.5, 2.9)
    axR.set_ylabel("Median survival (years)")
    axR.set_title("Median survival time,\nposterior median with 95% CrI")
    axR.legend(loc="lower left", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 3：個體存活曲線 + 不確定性帶（通關標準 2）────────────────────────
def individual_survival(patients, path):
    """patients: list of dict(name, desc, t_grid, draws, cox, med_post, med_cox)"""
    n = len(patients)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, patients):
        med = _band(ax, p["t_grid"], p["draws"], C_BAYES,
                    label="Bayesian posterior (50% / 95% band)")
        ax.plot(p["t_grid"], p["cox"], color=C_COX, lw=2.2, ls="--",
                label="Cox: a single line, no band")
        ax.axhline(0.5, color=C_REF, ls=":", lw=1.2)
        ax.annotate(f"median survival\n{p['med_post'][0]:.1f} yr "
                    f"[{p['med_post'][1]:.1f}, {p['med_post'][2]:.1f}]",
                    xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9.5,
                    color=C_BAYES,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_BAYES, alpha=0.9))
        ax.set_xlabel("Time since diagnosis (years)")
        ax.set_title(f"{p['name']}\n{p['desc']}", fontsize=10.5)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    axes[0].set_ylabel("Survival probability $S(t)$")
    fig.suptitle("Individual survival curves: the Bayesian answer is a distribution over curves",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 4：模型比較 WAIC / LOO（通關標準 3）──────────────────────────────
def model_comparison(ic, cv, path):
    """左：elpd_waic ± se（相對最佳）。右：p_waic vs 實際參數數 + OOF 指標。

    ic: DataFrame（evaluate.ic_table 的輸出，index=model）
    cv: {model: dict(c_index, c_boot_se, ibs)}
    """
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16.5, 5.2),
                                        gridspec_kw={"width_ratios": [1.15, 1, 1]})
    models_ = list(ic.index)
    best = ic["elpd_waic"].max()
    y = np.arange(len(models_))[::-1]

    for yi, m in zip(y, models_):
        d = ic.loc[m, "elpd_waic"] - best
        se = ic.loc[m, "se_waic"]
        axL.errorbar(d, yi, xerr=se, fmt="o", color=MCOL[m], ms=11, capsize=6, lw=2.2)
        axL.annotate(f"{d:+.1f}", (d, yi), textcoords="offset points", xytext=(0, 14),
                     ha="center", fontsize=10, color=MCOL[m])
    axL.axvline(0, color=C_REF, ls="--", lw=1.5)
    axL.set_yticks(y)
    axL.set_yticklabels([MLABEL[m] for m in models_])
    axL.set_xlabel(r"$\Delta$ elpd$_\mathrm{WAIC}$ (relative to best; higher is better)")
    axL.set_title("WAIC ranks the baseline distributions\n(error bars: standard error)")

    w = 0.36
    x = np.arange(len(models_))
    axM.bar(x - w / 2, ic["p_waic"], w, color=[MCOL[m] for m in models_],
            label=r"$p_\mathrm{WAIC}$ (effective params)")
    axM.bar(x + w / 2, ic["n_params_stated"], w, color=C_REF, alpha=0.55,
            label="parameters actually written")
    for xi, m in zip(x, models_):
        axM.annotate(f"{ic.loc[m, 'p_waic']:.1f}", (xi - w / 2, ic.loc[m, "p_waic"]),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9.5)
    axM.set_xticks(x)
    axM.set_xticklabels([MLABEL[m] for m in models_], fontsize=9.5)
    axM.set_ylabel("Number of parameters")
    axM.set_title("The WAIC penalty is an *effective*\nparameter count, not a written one")
    axM.legend(fontsize=9.5, loc="upper left")

    xs = np.arange(len(models_))
    ci = np.array([cv[m]["c_index"] for m in models_])
    se = np.array([cv[m]["c_boot_se"] for m in models_])
    axR.errorbar(xs, ci, yerr=1.96 * se, fmt="s", ms=10, capsize=6, lw=2.0,
                 color=C_BAYES, label="5-fold OOF C-index (±95%)")
    if "cox" in cv:
        axR.axhline(cv["cox"]["c_index"], color=C_COX, ls="--", lw=1.8,
                    label=f"Cox C-index = {cv['cox']['c_index']:.3f}")
    axR.set_xticks(xs)
    axR.set_xticklabels([MLABEL[m] for m in models_], fontsize=9.5)
    axR.set_ylabel("Concordance index")
    axR.set_title("...but out-of-fold discrimination\ncannot tell them apart")
    axR.legend(fontsize=9.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 5：風險函數形狀 —— WAIC 排序的機制 ───────────────────────────────
def hazard_shapes(t_grid, pop_hazards, ind_hazards, nonparam, peaks, path):
    """左：三模型的**族群層級** hazard vs 非參數 Nelson–Aalen。
    右：Log-Normal 的個體 vs 族群 hazard —— frailty 效應，以及殘餘失配。

    左圖必須用族群層級 hazard 才是公平比較：個體 h(t|x̄) 與族群 h_pop(t) 在有
    異質性時本來就不同（高風險者先離開風險集合），拿前者去對 Nelson–Aalen
    會誤判模型與資料不合。
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.6))
    na_t, na_h = nonparam[0], nonparam[1]
    na_info = nonparam[2] if len(nonparam) > 2 else None

    def _plot_na(ax, label=True):
        """畫 Nelson–Aalen，並把 at-risk 不足的尾段畫成淡線 —— 那一段是噪聲。"""
        if na_info is None:
            ax.plot(na_t, na_h, color=C_KM, lw=2.0, ls=":",
                    label="Nelson–Aalen (nonparametric)" if label else None)
            return
        rel = na_info["reliable"]
        ax.plot(na_t[rel], na_h[rel], color=C_KM, lw=2.0, ls=":",
                label=(f"Nelson–Aalen, at-risk $\\geq$ {na_info['min_at_risk']}"
                       if label else None))
        if (~rel).any():
            ax.plot(na_t[~rel], na_h[~rel], color=C_KM, lw=1.4, ls=":", alpha=0.28,
                    label=("noise-dominated tail (at-risk < "
                           f"{na_info['min_at_risk']})" if label else None))
            ax.axvspan(na_info["t_cut"], na_t.max(), color=C_REF, alpha=0.07, lw=0)

    for m, draws in pop_hazards.items():
        _band(axL, t_grid, draws, MCOL[m], label=MLABEL[m], alpha_hi=0.12, alpha_lo=0.22)
    _plot_na(axL)
    axL.axvline(peaks["na_peak"], color=C_KM, lw=1.2, ls="--", alpha=0.6)
    axL.annotate(f"nonparametric peak\n{peaks['na_peak']:.2f} yr",
                 xy=(peaks["na_peak"], axL.get_ylim()[1] * 0.40), fontsize=9,
                 color=C_KM, ha="left", xytext=(6, 0), textcoords="offset points")
    axL.set_xlabel("Time since diagnosis (years)")
    axL.set_ylabel(r"Population hazard $h_{\mathrm{pop}}(t)$ (events per year)")
    axL.set_title("Why WAIC prefers Log-Normal: only it can\nlet the hazard peak and then fall")
    axL.legend(fontsize=8.6, loc="lower right")
    axL.set_ylim(bottom=0)

    _band(axR, t_grid, ind_hazards["lognormal"], C_BAD2,
          label=r"individual hazard $h(t|\bar{x})$", alpha_hi=0.12, alpha_lo=0.22)
    _band(axR, t_grid, pop_hazards["lognormal"], MCOL["lognormal"],
          label=r"population hazard $h_{\mathrm{pop}}(t)$", alpha_hi=0.14, alpha_lo=0.26)
    _plot_na(axR)
    pk = peaks["lognormal_pop"]
    axR.axvline(pk["median"], color=MCOL["lognormal"], lw=1.6, ls="--")
    axR.axvspan(pk["lo"], pk["hi"], color=MCOL["lognormal"], alpha=0.10, lw=0)
    axR.axvline(peaks["na_peak"], color=C_KM, lw=1.6, ls="--")
    axR.set_xlabel("Time since diagnosis (years)")
    axR.set_ylabel("Hazard (events per year)")
    axR.set_title("Frailty effect, and an honest gap:\npopulation peak precedes the individual one")
    axR.legend(fontsize=9.2, loc="upper right")
    axR.set_ylim(bottom=0)
    note = (f"individual peak   {peaks['lognormal_ind_peak']:.2f} yr\n"
            f"population peak  {pk['median']:.2f} yr  [{pk['lo']:.2f}, {pk['hi']:.2f}]\n"
            f"nonparametric     {peaks['na_peak']:.2f} yr  ← still outside the CrI")
    if na_info is not None:
        note += f"\nNelson–Aalen trusted only up to {na_info['t_cut']:.1f} yr"
    axR.annotate(note, xy=(0.035, 0.055), xycoords="axes fraction", fontsize=8.8,
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_REF, alpha=0.93))

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 6：貝葉斯後驗 vs Cox 信賴區間 ────────────────────────────────────
def bayes_vs_cox(labels, ph_draws, cox_tbl, path):
    """森林圖：β_PH（由 AFT 換算）的後驗 vs Cox 的 log HR ± 95% CI。

    labels: 人類可讀特徵名列表（與 ph_draws 的欄位順序一致）
    ph_draws: (S, p) 後驗抽樣；cox_tbl: DataFrame(log_hr, lo95, hi95)，index 與欄位同序
    """
    p = len(labels)
    fig, ax = plt.subplots(figsize=(9.5, 0.72 * p + 2.4))
    y = np.arange(p)[::-1]
    off = 0.17

    lo95, lo50, med, hi50, hi95 = np.percentile(ph_draws, [2.5, 25, 50, 75, 97.5], axis=0)
    for i, yi in enumerate(y):
        ax.plot([lo95[i], hi95[i]], [yi + off] * 2, color=C_BAYES, lw=1.6, alpha=0.85)
        ax.plot([lo50[i], hi50[i]], [yi + off] * 2, color=C_BAYES, lw=4.2, alpha=0.85)
        ax.plot(med[i], yi + off, "o", color=C_BAYES, ms=7.5, zorder=3)

    cl, cu = cox_tbl["lo95"].to_numpy(), cox_tbl["hi95"].to_numpy()
    cm = cox_tbl["log_hr"].to_numpy()
    for i, yi in enumerate(y):
        ax.plot([cl[i], cu[i]], [yi - off] * 2, color=C_COX, lw=1.6, alpha=0.85)
        ax.plot(cm[i], yi - off, "s", color=C_COX, ms=7, zorder=3)

    ax.plot([], [], "o-", color=C_BAYES, label=r"Bayesian AFT $\to$ PH: $\beta_{PH}=-k\beta_{AFT}$"
                                               "\n(95% / 50% credible interval)")
    ax.plot([], [], "s-", color=C_COX, label="Cox partial likelihood\n(95% confidence interval)")
    ax.axvline(0, color=C_REF, ls="--", lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"log hazard ratio  (>0 = shorter survival)")
    ax.set_title("Two frameworks, nearly identical numbers —\nthe difference is what the interval *means*")
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 7：刪失的資訊代價 —— 精度由事件數決定 ────────────────────────────
ARM_STYLE = {
    "truncate": (C_CENS, "o", "A: shorten follow-up\n     n=686 kept, censored rows retained"),
    "subsample": (C_BAYES, "s", "B: random subsample\n     censoring rate fixed at 56%"),
    "truncate_drop": (C_BAD1, "^", "C: shorten follow-up, then drop censored\n     n = event count, no censored rows"),
}


def information_cost(summary, gap, path, key_label="pnodes"):
    """左：後驗 sd vs 事件數（三臂 + 理論 ∝ events^-1/2）。右：同一份資料對樣本數作圖。

    這張圖的結論**不是**「精度只由事件數決定」—— 實測否證了那個假設。
    三臂在事件數對齊後仍分離，分離量就是刪失列攜帶的迴歸資訊。
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6))

    for arm, (col, mk, lab) in ARM_STYLE.items():
        if arm not in summary:
            continue
        pts = summary[arm]["points"]
        ev = np.array([p["n_events"] for p in pts])
        sd = np.array([p["sd_mean"] for p in pts])
        err = np.array([p["sd_sd"] for p in pts])
        axL.errorbar(ev, sd, yerr=err, fmt=mk + "-", color=col, ms=8, lw=1.8, capsize=4,
                     label=f"{lab}\n     log-log slope = {summary[arm]['loglog_slope']:+.2f}")

    ev_all = np.array([p["n_events"] for p in summary["subsample"]["points"]])
    sd_all = np.array([p["sd_mean"] for p in summary["subsample"]["points"]])
    ref = sd_all[-1] * (ev_all / ev_all[-1]) ** -0.5
    axL.plot(ev_all, ref, color=C_KM, ls=":", lw=2.0,
             label=r"theory $\propto$ events$^{-1/2}$ (slope $-0.50$)")
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel("Number of observed events")
    axL.set_ylabel(rf"Posterior SD of $\beta_{{\mathrm{{{key_label}}}}}$")
    axL.set_title("Matched on events, the arms do NOT coincide:\ncensored rows still carry regression information")
    axL.legend(fontsize=8.2, loc="upper right")

    # 右：bias vs variance —— 光看區間寬度會挑到最錯的模型
    base = gap["baseline_mean"]
    for arm, (col, mk, lab) in ARM_STYLE.items():
        if arm not in summary:
            continue
        pts = summary[arm]["points"]
        bias = np.array([100.0 * abs(p["post_mean"] - base) / abs(base) for p in pts])
        sd = np.array([p["sd_mean"] for p in pts])
        ev = np.array([p["n_events"] for p in pts])
        axR.plot(bias, sd, mk + "-", color=col, ms=7, lw=1.4, alpha=0.85,
                 label=lab.split("\n")[0])
        axR.scatter(bias, sd, s=18 + 120 * (ev / ev.max()), color=col, alpha=0.35, lw=0)

    axR.set_xlabel(rf"Bias of $\beta_{{\mathrm{{{key_label}}}}}$ vs full-data posterior (%)")
    axR.set_ylabel(rf"Posterior SD of $\beta_{{\mathrm{{{key_label}}}}}$")
    axR.set_title("Precision is not correctness:\narm C is the tightest AND the most biased")
    note = f"mean |bias| / mean SD, matched on events:\n"
    for arm in ("truncate", "subsample", "truncate_drop"):
        if f"mean_abs_bias_pct_{arm}" in gap:
            tag = {"truncate": "A", "subsample": "B", "truncate_drop": "C"}[arm]
            note += (f"  {tag}:  {gap[f'mean_abs_bias_pct_{arm}']:5.1f}%   "
                     f"{gap[f'mean_sd_{arm}']:.4f}\n")
    if "n_horizons_C_narrowest_and_most_biased" in gap:
        note += (f"C narrowest & most biased in "
                 f"{gap['n_horizons_C_narrowest_and_most_biased']}/{gap['n_horizons']} settings")
    axR.annotate(note, xy=(0.34, 0.55), xycoords="axes fraction", fontsize=8.6,
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_REF, alpha=0.93))
    axR.legend(fontsize=9, loc="upper right")
    axR.annotate("marker size ∝ events", xy=(0.02, 0.055), xycoords="axes fraction",
                 fontsize=8, color=C_REF, style="italic")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
