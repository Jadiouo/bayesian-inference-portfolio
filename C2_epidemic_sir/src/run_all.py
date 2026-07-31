"""
C2 · 一鍵重現：`python src/run_all.py`

流程：資料（含來源交叉驗證）→ SIR 實作驗證 → 三個模型版本推論
      → 三個通關標準的量化 → 週末效應必要性 → 模型比較 → 落盤 → 出圖

⚠️ **定位**：這是**技術練習**（練狀態空間推論與時變參數的先驗設計），
不是公衛結論。真實流行病學牽涉通報延遲、檢驗量能、行為改變等大量領域知識。
見 README 的限制章節。

沿用 A2/A3/E1 的設計原則：推論結果與出圖資料落盤分離。
執行時間參考：約 35–40 分鐘（時變模型的 pytensor scan 梯度是主要成本）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import evaluate as E
import model as M
import plots as P
import sir as SIR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "data", "C_epi"))
FIG_DIR = os.path.join(ROOT, "figures")
PLOTDATA = os.path.join(DATA_DIR, "c2_plotdata.npz")
RESULTS = os.path.join(FIG_DIR, "results.json")

COUNTRY = "Germany"
START, END = "2020-03-01", "2020-06-30"
N_FORECAST = 14

# 主模型跑完整鏈；兩個對照組只需展示「參數荒謬」與「殘差有週期」，鏈可短一些
SAMPLE_CFG = {
    "timevarying": dict(draws=1000, tune=1500, chains=4, target_accept=0.95),
    "constant": dict(draws=1000, tune=1500, chains=4, target_accept=0.9),
    "tv_no_dow": dict(draws=750, tune=1000, chains=4, target_accept=0.95),
}


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _jsonify(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return float(o)


def _quantiles(samples) -> dict:
    s = np.asarray(samples)
    return {"median": np.median(s, axis=0),
            "lo95": np.percentile(s, 2.5, axis=0), "hi95": np.percentile(s, 97.5, axis=0),
            "lo50": np.percentile(s, 25, axis=0), "hi50": np.percentile(s, 75, axis=0)}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    res: dict = {"config": {"country": COUNTRY, "start": START, "end": END,
                            "n_forecast": N_FORECAST, "sampling": SAMPLE_CFG}}
    arrays: dict = {}

    # ── 1 · 資料 ─────────────────────────────────────────────────────────
    ep = D.load(DATA_DIR, COUNTRY, START, END)
    tr, te = D.train_test_split(ep, N_FORECAST)
    _log(f"data: {ep.summary()}")
    _log(f"  train {tr.n_days} days, held out {te.n_days} days "
         f"({te.dates[0].date()}–{te.dates[-1].date()})")

    xsrc = D.cross_validate_sources(DATA_DIR, COUNTRY, START, END)
    dowp = D.day_of_week_profile(tr)
    res["data"] = {"n_days": ep.n_days, "n_train": tr.n_days, "n_test": te.n_days,
                   "population": ep.population, "total_cases": float(ep.cases.sum()),
                   "negatives_clipped": ep.n_negative_clipped,
                   "policy_indices": tr.policy_indices(),
                   "dates_start": str(ep.dates[0].date()), "dates_end": str(ep.dates[-1].date())}
    res["source_cross_validation"] = xsrc
    res["dow_profile"] = dowp
    _log(f"  source cross-check: JHU {xsrc['jhu_total']:.0f} vs OWID {xsrc['owid_total']:.0f} "
         f"(rel diff {100 * xsrc['total_rel_diff']:.2f}%, weekly corr {xsrc['corr_weekly']:.4f}, "
         f"OWID zero-days {100 * xsrc['owid_zero_day_fraction']:.0f}%)")
    _log(f"  day-of-week spread: {dowp['_spread']['max_over_min']:.3f}× "
         f"(min {dowp['_spread']['min_ratio']:.3f}, max {dowp['_spread']['max_ratio']:.3f})")

    arrays["dates_train"] = tr.dates.to_numpy()
    arrays["dates_test"] = te.dates.to_numpy()
    arrays["cases_train"] = tr.cases
    arrays["cases_test"] = te.cases
    arrays["dates_all"] = ep.dates.to_numpy()
    arrays["cases_all"] = ep.cases

    # ── 2 · SIR 實作驗證 ─────────────────────────────────────────────────
    verif = SIR.verify_scan_matches_numpy(n_days=tr.n_days)
    res["sir_verification"] = verif
    for k, v in verif.items():
        if k == "passed":
            continue
        _log(f"SIR scan vs numpy [{k:18s}] passed={v['passed']} "
             f"worst_rel_err={v['worst']:.2e}")
    assert verif["passed"], "pytensor scan 與 numpy 的 SIR 不一致"

    N = ep.population
    demo_sim = SIR.simulate_numpy(0.28, 0.1, N - 200, 200, N, 220)
    demo = {"new_infections": demo_sim["new_infections"],
            "S_frac": demo_sim["S"] / N,
            "R_t": SIR.reproduction_number(np.repeat(0.28, 220), 0.1, demo_sim["S"], N),
            "R0": 2.8}
    for k in ("new_infections", "S_frac", "R_t"):
        arrays[f"demo_{k}"] = demo[k]

    # ── 3 · 三個模型版本 ─────────────────────────────────────────────────
    idatas, conv, params, fits = {}, {}, {}, {}
    for version in ("constant", "timevarying", "tv_no_dow"):
        cfg = SAMPLE_CFG[version]
        _log(f"sampling [{version}] draws={cfg['draws']} tune={cfg['tune']} "
             f"chains={cfg['chains']}...")
        t1 = time.time()
        idata = M.sample(M.build(tr, version), **cfg)
        idatas[version] = idata
        conv[version] = M.check_convergence(idata, label=version, raise_on_fail=False)
        params[version] = M.posterior_summary(idata, tr)
        _log(f"  [{version}] {time.time() - t1:.0f}s div={conv[version]['divergent']} "
             f"r_hat={conv[version]['max_rhat_scalars']:.3f} "
             f"ESS={conv[version]['min_ess_scalars']:.0f} passed={conv[version]['passed']}")
        if not conv[version]["passed"]:
            _log(f"  ⚠️ [{version}] convergence issues: {conv[version]['failures']}")
        p = params[version]
        _log(f"  [{version}] I0={p['I0']['median']:,.0f} rho={p['rho']['median']:.4f} "
             f"gamma={p['gamma']['median']:.4f} "
             f"infectious_period={p['infectious_period_days']['median']:.2f}d "
             f"attack_rate={p['final_attack_rate']['median']:.4%}")

        pp = E.posterior_predictive(idata, tr)
        fits[version] = _quantiles(pp)
        for k, v in fits[version].items():
            arrays[f"pp_{version}_{k}"] = v

    res["convergence"] = conv
    res["params"] = params

    # ── 4 · 通關標準 1：R_t 與政策時點 ───────────────────────────────────
    main_id = idatas["timevarying"]
    rt = E.rt_trace(main_id)
    pol = E.rt_at_policy_changes(main_id, tr)
    res["rt_policies"] = pol
    for k in ("median", "lo95", "hi95", "lo50", "hi50", "p_below_1"):
        arrays[f"rt_{k}"] = rt[k]
    _log(f"R_t: start={pol['rt_start_median']:.2f} end={pol['rt_end_median']:.2f}")
    for label, st in pol["policies"].items():
        _log(f"  {st['date']} {label[:42]:42s} "
             f"R_t {st['rt_before_median']:.2f}→{st['rt_after_median']:.2f} "
             f"drop={st['drop_median']:+.2f} P(drop)={st['p_decreased']:.3f}")
    _log(f"  first day P(R_t<1)>0.95: {pol['first_day_rt_below_1_p95']}")

    # ── 5 · 通關標準 2：不確定性 vs 資料量 ───────────────────────────────
    w = E.rt_width_vs_cases(main_id, tr)
    res["rt_width_vs_cases"] = {k: v for k, v in w.items()
                                if k not in ("width_rel", "width_abs", "cases", "rt_median")}
    for k in ("width_rel", "width_abs", "rt_median"):
        arrays[f"w_{k}"] = w[k]
    _log(f"R_t interval width vs cases: log-log slope={w['loglog_slope']:+.4f}"
         f"±{w['loglog_slope_se']:.4f} r={w['loglog_r']:+.3f} p={w['loglog_p']:.2e}")
    _log(f"  relative width low/high cases = {w['width_ratio_low_over_high']:.3f}× "
         f"(absolute-width metric would give {w['width_abs_ratio_low_over_high']:.3f}×)")

    # ── 6 · 通關標準 3：後驗預測校準 ─────────────────────────────────────
    pp_main = E.posterior_predictive(main_id, tr)
    cov_in = E.coverage(tr.cases, pp_main)
    fc = E.forecast(main_id, tr, N_FORECAST, dow_future=te.dow)
    cov_out = E.coverage(te.cases, fc["cases"])
    fc_q = _quantiles(fc["cases"])
    for k, v in fc_q.items():
        arrays[f"fc_{k}"] = v
    arrays["fc_rt_median"] = np.median(fc["R_t"], axis=0)
    res["coverage_in_sample"] = cov_in
    res["coverage_out_of_sample"] = cov_out
    _log(f"coverage in-sample : " + ", ".join(
        f"{l}%={cov_in[f'cov_{l}']:.3f}" for l in (50, 80, 95)) +
        f"  mean|err|={cov_in['mean_abs_calib_error']:.4f}")
    _log(f"coverage out-sample: " + ", ".join(
        f"{l}%={cov_out[f'cov_{l}']:.3f}" for l in (50, 80, 95)) +
        f"  mean|err|={cov_out['mean_abs_calib_error']:.4f}")

    # ── 7 · 週末效應的必要性 ─────────────────────────────────────────────
    acf_with = E.residual_periodicity(main_id, tr)
    acf_without = E.residual_periodicity(idatas["tv_no_dow"], tr)
    res["residual_periodicity"] = {
        "with_dow": {k: v for k, v in acf_with.items() if k not in ("acf", "lags", "resid")},
        "without_dow": {k: v for k, v in acf_without.items()
                        if k not in ("acf", "lags", "resid")},
    }
    arrays["acf_with"] = acf_with["acf"]
    arrays["acf_without"] = acf_without["acf"]
    arrays["acf_lags"] = acf_with["lags"]
    _log(f"residual ACF at weekly lags — with dow: "
         f"{ {k: round(v, 4) for k, v in acf_with['acf_at_weekly_lags'].items()} } "
         f"(max {acf_with['max_weekly_acf']:.4f}, {acf_with['n_weekly_lags_significant']} sig)")
    _log(f"                            without dow: "
         f"{ {k: round(v, 4) for k, v in acf_without['acf_at_weekly_lags'].items()} } "
         f"(max {acf_without['max_weekly_acf']:.4f}, "
         f"{acf_without['n_weekly_lags_significant']} sig, ci95={acf_without['ci95']:.4f})")

    # ── 8 · 模型比較 ─────────────────────────────────────────────────────
    cmp = E.compare_models(idatas)
    res["model_comparison"] = cmp
    _log(f"WAIC comparison — best: {cmp['best']}")
    for nm, r in cmp["per_model"].items():
        _log(f"  {nm:13s} elpd_waic={r['elpd_waic']:9.1f}±{r['se_waic']:5.1f} "
             f"p_waic={r['p_waic']:6.1f} max_pareto_k={r['max_pareto_k']:.2f}")

    dow_post = params["timevarying"].get("dow_effect")
    res["dow_posterior"] = dow_post

    # ── 落盤 ─────────────────────────────────────────────────────────────
    np.savez_compressed(PLOTDATA, **arrays)
    res["_meta"] = {"runtime_sec": round(time.time() - t0, 1),
                    "plotdata": os.path.relpath(PLOTDATA, ROOT)}
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=_jsonify)
    _log(f"saved {PLOTDATA} ({os.path.getsize(PLOTDATA) / 1e6:.2f} MB) and {RESULTS}")

    make_figures(arrays, res, verif, demo, fits, params, rt, pol, w, cov_in, cov_out,
                 fc_q, acf_with, acf_without, cmp, dow_post, tr, te)
    _log(f"done in {res['_meta']['runtime_sec']:.0f}s")


def make_figures(arrays, res, verif, demo, fits, params, rt, pol, w, cov_in, cov_out,
                 fc_q, acf_with, acf_without, cmp, dow_post, tr, te):
    P.data_overview(arrays["dates_all"], arrays["cases_all"], tr.policy_indices(),
                    res["dow_profile"], res["source_cross_validation"],
                    os.path.join(FIG_DIR, "01_data_overview.png"))
    P.sir_verification(verif, demo, os.path.join(FIG_DIR, "02_sir_verification.png"))
    P.fixed_vs_timevarying(arrays["dates_train"], arrays["cases_train"], fits, params,
                           os.path.join(FIG_DIR, "03_fixed_vs_timevarying.png"))
    P.rt_and_policies(arrays["dates_train"], rt, pol,
                      os.path.join(FIG_DIR, "04_rt_and_policies.png"))
    P.uncertainty_vs_data(w, os.path.join(FIG_DIR, "05_uncertainty_vs_data.png"))
    P.posterior_predictive(arrays["dates_train"], arrays["cases_train"],
                           fits["timevarying"], arrays["dates_test"],
                           arrays["cases_test"], fc_q, cov_in, cov_out,
                           os.path.join(FIG_DIR, "06_posterior_predictive.png"))
    P.dow_and_comparison(dow_post, acf_with, acf_without, cmp,
                         os.path.join(FIG_DIR, "07_dow_and_comparison.png"))
    _log("figures written")


if __name__ == "__main__":
    main()
