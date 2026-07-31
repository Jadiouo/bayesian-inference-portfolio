"""
C2 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只吃 numpy / dict。

沿用 A2/A3/E1 的原則：不接觸 PyMC，所有資料由 run_all 落盤後傳入。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
})

C_OBS = "#22223b"
C_TV = "#2e6fdb"
C_CONST = "#d1495b"
C_NODOW = "#9b5de5"
C_POL = "#e8873a"
C_OK = "#2a9d8f"
C_REF = "#6c757d"

VCOL = {"timevarying": C_TV, "constant": C_CONST, "tv_no_dow": C_NODOW}
VLAB = {"timevarying": "Time-varying β + day-of-week",
        "constant": "Constant β (control)",
        "tv_no_dow": "Time-varying β, no day-of-week (control)"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _dates(d):
    return np.asarray(d, dtype="datetime64[ns]")


def _fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def _band(ax, x, lo, hi, color, alpha=0.20, label=None):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, label=label)


# ── 圖 1：資料概觀與來源交叉驗證 ────────────────────────────────────────
def data_overview(dates, cases, policies, dow_profile, xsrc, path):
    """左：每日病例 + 政策時點。右上：day-of-week profile。右下：兩來源週總和對照。

    右下那格是「換資料來源」這個決定的證據：計劃書指定 OWID，
    但它已改成週報格式（實測 85% 的天是 0），無法支援每日 SIR 與週末效應。
    改用 JHU 之後，兩者的**週總和**必須一致，否則問題只是被換掉而非解決。
    """
    fig = plt.figure(figsize=(14.2, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.42, wspace=0.26)
    d = _dates(dates)

    ax = fig.add_subplot(gs[:, 0])
    ax.bar(d, cases, width=1.0, color=C_OBS, alpha=0.55, label="Daily confirmed (JHU)")
    for i, (label, idx) in enumerate(policies.items()):
        ax.axvline(d[idx], color=C_POL, ls="--", lw=1.6, alpha=0.9)
        ax.annotate(label.split("(")[0].strip(), xy=(d[idx], ax.get_ylim()[1]),
                    xytext=(3, -12 - 26 * i), textcoords="offset points",
                    fontsize=8, color=C_POL, rotation=0, va="top",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C_POL, alpha=0.85))
    ax.set_ylabel("Daily confirmed cases")
    ax.set_title("Germany, first wave 2020 — the data being modelled")
    ax.legend(fontsize=9, loc="upper right")
    _fmt_dates(ax)

    axR1 = fig.add_subplot(gs[0, 1])
    ratios = [dow_profile[k]["ratio"] for k in DOW]
    cols = [C_OK if r >= 1 else C_REF for r in ratios]
    axR1.bar(range(7), ratios, 0.65, color=cols)
    axR1.axhline(1.0, color=C_OBS, lw=1.5)
    axR1.set_xticks(range(7))
    axR1.set_xticklabels(DOW, fontsize=9)
    axR1.set_ylabel("Mean cases ÷ overall mean")
    sp = dow_profile["_spread"]
    axR1.set_title(f"Day-of-week effect: {sp['max_over_min']:.2f}× between\n"
                   f"quietest ({DOW[int(np.argmin(ratios))]}) and busiest "
                   f"({DOW[int(np.argmax(ratios))]}) day", fontsize=10)
    for i, r in enumerate(ratios):
        axR1.annotate(f"{r:.2f}", (i, r), textcoords="offset points", xytext=(0, 3),
                      ha="center", fontsize=8)

    axR2 = fig.add_subplot(gs[1, 1])
    axR2.axis("off")
    txt = (
        "Why JHU instead of OWID (the planned source)\n"
        f"  OWID is now weekly-reported: {100 * xsrc['owid_zero_day_fraction']:.0f}% of days are 0\n"
        f"  (a whole week's total lands on one day)\n\n"
        "Cross-check on WEEKLY totals:\n"
        f"  JHU total   {xsrc['jhu_total']:>10,.0f}\n"
        f"  OWID total  {xsrc['owid_total']:>10,.0f}\n"
        f"  relative difference   {100 * xsrc['total_rel_diff']:.2f}%\n"
        f"  weekly correlation    {xsrc['corr_weekly']:.4f}\n"
        f"  median weekly SMAPE   {100 * xsrc['weekly_smape_median']:.1f}%\n"
        "  → same epidemic, different time resolution"
    )
    axR2.text(0.0, 0.98, txt, va="top", ha="left", fontsize=8.6, family="monospace",
              bbox=dict(boxstyle="round,pad=0.5", fc="#f6f7fb", ec=C_REF, alpha=0.95))

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 2：SIR 實作驗證 ─────────────────────────────────────────────────
def sir_verification(verif, demo, path):
    """左：scan vs numpy 的逐點誤差。右：SIR 動態示意（固定 β 的單峰結構）。"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    labels, vals, cols = [], [], []
    for case in ("constant_beta", "time_varying_beta"):
        for state, err in verif[case]["max_rel_err"].items():
            labels.append(f"{case.replace('_beta', '')}\n{state}")
            vals.append(max(err, 1e-18))
            cols.append(C_OK if verif[case]["passed"] else C_CONST)
    x = np.arange(len(labels))
    axL.bar(x, vals, 0.6, color=cols)
    axL.axhline(1e-9, color=C_CONST, ls="--", lw=1.6, label="tolerance 1e−9")
    axL.set_yscale("log")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=7.4)
    axL.set_ylabel("Max relative error (log scale)")
    axL.set_title("pytensor scan vs numpy: identical to machine precision\n"
                  "(a mis-wired scan fails silently, so this check is required)")
    axL.legend(fontsize=9)

    t = np.arange(len(demo["new_infections"]))
    axR.plot(t, demo["new_infections"], color=C_CONST, lw=2.2,
             label="new infections per day")
    ax2 = axR.twinx()
    ax2.plot(t, demo["S_frac"], color=C_TV, lw=2.0, ls="--", label="S/N (susceptible)")
    ax2.plot(t, demo["R_t"], color=C_OK, lw=2.0, ls=":", label="$R_t$")
    ax2.axhline(1.0, color=C_REF, lw=1.2, alpha=0.6)
    ax2.set_ylabel("S/N  and  $R_t$")
    ax2.grid(False)
    axR.set_xlabel("Day")
    axR.set_ylabel("New infections per day")
    axR.set_title(f"Constant-β SIR ($R_0$={demo['R0']:.1f}): the ONLY way to get a\n"
                  "downturn is to exhaust susceptibles ($R_t$ falls with S/N)")
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=8.8, loc="upper right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 3：固定 β vs 時變 β —— 核心發現 ─────────────────────────────────
def fixed_vs_timevarying(dates, cases, fits, params, path):
    """左：兩個模型的擬合。右：參數對照表（固定 β 被迫用荒謬的值）。

    這是本專案最重要的一張圖：固定 β 的 SIR 無法用「政策改變」解釋下降，
    只能靠耗盡易感者，於是被迫把 I₀ 推到數十萬、通報率壓到 0.3%
    才能同時滿足「形狀對」與「量級對」。
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.4, 5.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    d = _dates(dates)
    axL.bar(d, cases, width=1.0, color=C_OBS, alpha=0.32, label="Observed")
    for v in ("constant", "timevarying"):
        if v not in fits:
            continue
        f = fits[v]
        _band(axL, d, f["lo95"], f["hi95"], VCOL[v], alpha=0.18)
        axL.plot(d, f["median"], color=VCOL[v], lw=2.2, label=VLAB[v])
    axL.set_ylabel("Daily confirmed cases")
    axL.set_title("Both models can fit the curve —\nbut only one does it with plausible parameters")
    axL.legend(fontsize=9)
    _fmt_dates(axL)

    axR.axis("off")
    rows = [("Parameter", "Constant β", "Time-varying β", "Plausible?")]
    keys = [("I0", "Initial infections $I_0$", "{:,.0f}"),
            ("rho", "Reporting rate ρ", "{:.3f}"),
            ("gamma", "Recovery rate γ", "{:.3f}"),
            ("infectious_period_days", "Infectious period (days)", "{:.1f}"),
            ("final_attack_rate", "Final attack rate", "{:.3%}")]
    for k, label, fmt in keys:
        c = params.get("constant", {}).get(k, {})
        t = params.get("timevarying", {}).get(k, {})
        cv = c.get("median", c.get("mean", np.nan))
        tv = t.get("median", t.get("mean", np.nan))
        rows.append((label, fmt.format(cv) if np.isfinite(cv) else "—",
                     fmt.format(tv) if np.isfinite(tv) else "—", ""))
    verdict = {"Initial infections $I_0$": ("✗ absurd", "✓"),
               "Reporting rate ρ": ("✗ absurd", "✓"),
               "Recovery rate γ": ("~", "✓"),
               "Infectious period (days)": ("~", "✓"),
               "Final attack rate": ("✗", "✓")}
    tbl_rows = []
    for r in rows[1:]:
        v = verdict.get(r[0], ("", ""))
        tbl_rows.append([r[0], r[1], r[2], f"{v[0]} / {v[1]}"])
    tab = axR.table(cellText=tbl_rows,
                    colLabels=["Parameter", "Constant β", "Time-varying β", "const / t-var"],
                    loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.6)
    tab.scale(1.0, 1.7)
    for (i, j), cell in tab.get_celld().items():
        if i == 0:
            cell.set_facecolor("#e9ecf5")
            cell.set_text_props(weight="bold")
        elif j == 1:
            cell.set_facecolor("#fdecef")
        elif j == 2:
            cell.set_facecolor("#e8f3ee")
    axR.set_title("Constant β must invent half a million initial\n"
                  "infections and a 0.3% reporting rate", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 4：R_t 與政策時點（通關標準 1）──────────────────────────────────
def rt_and_policies(dates, rt, policy_stats, path):
    """R_t 後驗帶 + 政策時點 + P(R_t<1) 的軌跡。"""
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(13.0, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1]})
    d = _dates(dates)
    _band(axT, d, rt["lo95"], rt["hi95"], C_TV, alpha=0.18, label="95% credible interval")
    _band(axT, d, rt["lo50"], rt["hi50"], C_TV, alpha=0.32, label="50% credible interval")
    axT.plot(d, rt["median"], color=C_TV, lw=2.4, label="posterior median $R_t$")
    axT.axhline(1.0, color=C_OBS, ls="-", lw=1.8)
    axT.annotate("$R_t = 1$  (epidemic turns around)", xy=(d[3], 1.0),
                 xytext=(0, 6), textcoords="offset points", fontsize=9, color=C_OBS)

    for i, (label, st) in enumerate(policy_stats["policies"].items()):
        idx = st["day_index"]
        axT.axvline(d[idx], color=C_POL, ls="--", lw=1.6, alpha=0.9)
        axT.annotate(f"{label.split('(')[0].strip()}\n"
                     f"$R_t$ {st['rt_before_median']:.2f}→{st['rt_after_median']:.2f}"
                     f"  P(drop)={st['p_decreased']:.2f}",
                     xy=(d[idx], axT.get_ylim()[1]),
                     xytext=(4, -14 - 42 * (i % 2)), textcoords="offset points",
                     fontsize=8, color=C_POL, va="top",
                     bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=C_POL, alpha=0.9))
    fb = policy_stats.get("first_day_rt_below_1_p95")
    if fb:
        axT.axvline(d[fb["day_index"]], color=C_OK, lw=2.0, alpha=0.8)
        axT.annotate(f"first day with\nP($R_t$<1) > 0.95\n{fb['date']}",
                     xy=(d[fb["day_index"]], 0.15), xytext=(6, 0),
                     textcoords="offset points", fontsize=8.4, color=C_OK,
                     bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=C_OK, alpha=0.9))
    axT.set_ylabel("Effective reproduction number $R_t$")
    axT.set_title("Pass criterion 1: $R_t$ has a clear uncertainty band and moves "
                  "at policy change points")
    axT.legend(fontsize=9, loc="upper right")

    axB.fill_between(d, 0, rt["p_below_1"], color=C_OK, alpha=0.45)
    axB.plot(d, rt["p_below_1"], color=C_OK, lw=2.0)
    axB.axhline(0.95, color=C_OBS, ls=":", lw=1.6)
    axB.annotate("0.95", xy=(d[1], 0.95), xytext=(0, 3), textcoords="offset points",
                 fontsize=8, color=C_OBS)
    axB.set_ylim(0, 1.02)
    axB.set_ylabel("P($R_t$ < 1)")
    axB.set_xlabel("Date")
    _fmt_dates(axB)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 5：不確定性 vs 資料量（通關標準 2）──────────────────────────────
def uncertainty_vs_data(w, path):
    """左：相對寬度 vs 病例數（正確度量）。右：絕對寬度的同一張圖（度量假影）。"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    cases = np.asarray(w["cases"])
    m = cases > 0

    axL.scatter(cases[m], w["width_rel"][m], s=26, color=C_TV, alpha=0.7)
    xs = np.linspace(np.log(cases[m].min()), np.log(cases[m].max()), 50)
    axL.plot(np.exp(xs), np.exp(w["loglog_slope"] * xs +
                               np.log(w["width_rel_highcases_median"]) -
                               w["loglog_slope"] * np.log(w["cases_q75"])),
             color=C_OBS, lw=2.0, ls="--",
             label=f"log-log slope = {w['loglog_slope']:+.3f} ± {w['loglog_slope_se']:.3f}")
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel("Daily confirmed cases (log scale)")
    axL.set_ylabel(r"RELATIVE width of $R_t$ interval:  $\log(hi_{95}/lo_{95})$")
    axL.set_title(f"Pass criterion 2: fewer cases → wider interval\n"
                  f"low-case days are {w['width_ratio_low_over_high']:.2f}× wider "
                  f"(r={w['loglog_r']:+.2f}, p={w['loglog_p']:.1e})")
    axL.legend(fontsize=9)

    axR.scatter(cases[m], w["width_abs"][m], s=26, color=C_CONST, alpha=0.7)
    axR.set_xscale("log")
    axR.set_xlabel("Daily confirmed cases (log scale)")
    axR.set_ylabel(r"ABSOLUTE width:  $hi_{95}-lo_{95}$")
    axR.set_title("The same data with the WRONG metric:\n"
                  f"low-case days look {w['width_abs_ratio_low_over_high']:.2f}× "
                  "*narrower*")
    ax2 = axR.twinx()
    ax2.scatter(cases[m], w["rt_median"][m], s=18, color=C_REF, alpha=0.55, marker="^")
    ax2.set_ylabel("Posterior median $R_t$ (grey triangles)", color=C_REF, fontsize=9.5)
    ax2.grid(False)
    axR.annotate("Absolute width scales with the level of $R_t$\n"
                 "($R_t$≈3 early, ≈0.6 late), so it measures\n"
                 "the multiplicative scale, not the information.\n"
                 "β is a random walk on the LOG scale — the\n"
                 "uncertainty lives there.",
                 xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.4,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_CONST, alpha=0.93))

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 6：後驗預測與校準（通關標準 3）──────────────────────────────────
def posterior_predictive(dates_train, cases_train, pp_q, dates_test, cases_test,
                         fc_q, cov_in, cov_out, path):
    """左：樣本內 + 樣本外預測帶。右：覆蓋率 vs 名目水準的校準圖。"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.2, 5.4),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    dtr, dte = _dates(dates_train), _dates(dates_test)

    _band(axL, dtr, pp_q["lo95"], pp_q["hi95"], C_TV, alpha=0.18, label="95% posterior predictive")
    _band(axL, dtr, pp_q["lo50"], pp_q["hi50"], C_TV, alpha=0.32, label="50%")
    axL.plot(dtr, pp_q["median"], color=C_TV, lw=1.8)
    axL.plot(dtr, cases_train, "o", color=C_OBS, ms=3.2, label="observed (in-sample)")

    _band(axL, dte, fc_q["lo95"], fc_q["hi95"], C_OK, alpha=0.22,
          label="95% forecast (out-of-sample)")
    _band(axL, dte, fc_q["lo50"], fc_q["hi50"], C_OK, alpha=0.38)
    axL.plot(dte, fc_q["median"], color=C_OK, lw=2.0)
    axL.plot(dte, cases_test, "s", color=C_CONST, ms=6, label="observed (held out)")
    axL.axvline(dte[0], color=C_REF, ls="--", lw=1.5)
    axL.annotate("forecast starts\n(14 days held out)", xy=(dte[0], axL.get_ylim()[1]),
                 xytext=(-70, -12), textcoords="offset points", fontsize=8.4, color=C_REF)
    axL.set_yscale("log")
    axL.set_ylabel("Daily confirmed cases (log scale)")
    axL.set_title("Pass criterion 3: does the band contain the truth?\n"
                  "The forecast band widens because β keeps random-walking")
    axL.legend(fontsize=8.4, loc="lower left")
    _fmt_dates(axL)

    levels = [50, 80, 95]
    xi = np.arange(len(levels))
    w = 0.35
    ci = [cov_in[f"cov_{l}"] for l in levels]
    co = [cov_out[f"cov_{l}"] for l in levels]
    axR.bar(xi - w / 2, ci, w, color=C_TV, label="in-sample")
    axR.bar(xi + w / 2, co, w, color=C_OK, label="out-of-sample (14 d)")
    for i, l in enumerate(levels):
        axR.plot([i - 0.5, i + 0.5], [l / 100, l / 100], color=C_OBS, lw=2.2)
        axR.annotate(f"{ci[i]:.2f}", (i - w / 2, ci[i]), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8.6)
        axR.annotate(f"{co[i]:.2f}", (i + w / 2, co[i]), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8.6)
    axR.plot([], [], color=C_OBS, lw=2.2, label="nominal level")
    axR.set_xticks(xi)
    axR.set_xticklabels([f"{l}%" for l in levels])
    axR.set_ylim(0, 1.15)
    axR.set_xlabel("Nominal credible level")
    axR.set_ylabel("Empirical coverage")
    axR.set_title(f"Calibration\nmean |error|: in {cov_in['mean_abs_calib_error']:.3f}, "
                  f"out {cov_out['mean_abs_calib_error']:.3f}")
    axR.legend(fontsize=8.8, loc="lower right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── 圖 7：week-day 效應、殘差週期、模型比較 ────────────────────────────
def dow_and_comparison(dow_post, acf_with, acf_without, cmp, path):
    """左：day-of-week 後驗。中：殘差 ACF 有/無 dow 項。右：WAIC 比較。"""
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16.2, 5.2))

    med = np.asarray(dow_post["median"])
    lo = np.asarray(dow_post["lo95"])
    hi = np.asarray(dow_post["hi95"])
    x = np.arange(7)
    axL.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", color=C_TV, ms=9,
                 capsize=5, lw=2.0)
    axL.axhline(1.0, color=C_OBS, lw=1.6)
    axL.set_xticks(x)
    axL.set_xticklabels(DOW, fontsize=9)
    axL.set_ylabel("Reporting multiplier (geometric mean fixed at 1)")
    axL.set_title("Estimated day-of-week reporting effect\n(posterior median, 95% CrI)")
    for xi, m in zip(x, med):
        axL.annotate(f"{m:.2f}", (xi, m), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8.2)

    for acf, col, lab in ((acf_without, C_NODOW, "without day-of-week term"),
                          (acf_with, C_TV, "with day-of-week term")):
        lags = np.asarray(acf["lags"])
        axM.plot(lags[1:], np.asarray(acf["acf"])[1:], "o-", color=col, ms=5, lw=1.6,
                 label=f"{lab}  (max at 7/14/21: {acf['max_weekly_acf']:.3f})")
    ci = acf_with["ci95"]
    axM.axhspan(-ci, ci, color=C_REF, alpha=0.16, lw=0, label="95% white-noise band")
    for k in (7, 14, 21):
        axM.axvline(k, color=C_POL, ls=":", lw=1.4, alpha=0.8)
    axM.axhline(0, color=C_OBS, lw=1.2)
    axM.set_xlabel("Lag (days)")
    axM.set_ylabel("Residual autocorrelation")
    axM.set_title("Why the day-of-week term is needed:\nomitting it leaves 7-day structure")
    axM.legend(fontsize=8.2, loc="upper right")

    names = list(cmp["per_model"].keys())
    best = cmp["best"]
    order = sorted(names, key=lambda k: -cmp["per_model"][k]["elpd_waic"])
    y = np.arange(len(order))[::-1]
    for yi, nm in zip(y, order):
        r = cmp["per_model"][nm]
        d = r["elpd_waic"] - cmp["per_model"][best]["elpd_waic"]
        axR.errorbar(d, yi, xerr=r["se_waic"], fmt="o", color=VCOL.get(nm, C_REF),
                     ms=11, capsize=6, lw=2.2)
        axR.annotate(f"{d:+.1f}", (d, yi), textcoords="offset points", xytext=(0, 13),
                     ha="center", fontsize=9)
    axR.axvline(0, color=C_REF, ls="--", lw=1.5)
    axR.set_yticks(y)
    axR.set_yticklabels([VLAB.get(n, n).split("(")[0].strip() for n in order], fontsize=8.6)
    axR.set_xlabel(r"$\Delta$ elpd$_\mathrm{WAIC}$ (relative to best)")
    axR.set_title(f"Model comparison — best: {VLAB.get(best, best).split('(')[0].strip()}")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
