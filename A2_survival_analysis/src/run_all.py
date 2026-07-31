"""
A2 · 一鍵重現：`python src/run_all.py`

流程：資料 → 手刻似然驗證 → 三個基準分佈推論 → 刪失處理對照 → 頻率派對照
      → 模型比較（WAIC/LOO + 5-fold OOF）→ 風險函數形狀 → 資訊代價實驗
      → 先驗敏感度 → 落盤 → 出圖

沿用 B2 的設計原則：**推論結果與出圖資料落盤分離**。
所有繪圖需要的陣列寫進 data/A_medical/a2_plotdata.npz，純量結果寫進
figures/results.json，之後調整任何圖表都只要跑 `python src/replot.py`，
不必重跑推論（本專案全程約 4 分鐘，其中資訊代價實驗佔一半）。

執行時間參考（RTX 5070 Ti 機器、CPU 取樣）：約 4 分鐘。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import censoring as cn
import data as D
import evaluate as ev
import frequentist as fq
import information as inf
import models as M
import plots as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "data", "A_medical"))
FIG_DIR = os.path.join(ROOT, "figures")
PLOTDATA = os.path.join(DATA_DIR, "a2_plotdata.npz")
RESULTS = os.path.join(FIG_DIR, "results.json")

DRAWS, TUNE, CHAINS = 2000, 2000, 4
T_MAX = 7.0
T_GRID = np.linspace(0.02, T_MAX, 160)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    t_start = time.time()
    res: dict = {}
    arrays: dict = {}

    # ── 1. 資料 ──────────────────────────────────────────────────────────
    s, scaler = D.load(DATA_DIR)
    _log(f"data: {s.summary()}")
    res["data"] = {
        "n": s.n, "n_events": s.n_events, "n_censored": s.n_censored,
        "censoring_rate": s.censoring_rate,
        "followup_min": float(s.t.min()), "followup_max": float(s.t.max()),
        "median_followup_censored": float(np.median(s.t[s.event == 0])),
        "median_time_events": float(np.median(s.t[s.event == 1])),
        "features": s.features,
    }
    arrays["t"] = s.t
    arrays["event"] = s.event

    # ── 2. 手刻刪失似然 vs pm.Censored（靈魂）──────────────────────────
    rng = np.random.default_rng(0)
    beta_probe = rng.normal(0, 0.3, s.X.shape[1])
    lam_probe = np.exp(1.6 + s.X @ beta_probe)
    checks = [
        cn.verify_equivalence(s.t, s.event, "weibull", (1.4, lam_probe)),
        cn.verify_equivalence(s.t, s.event, "lognormal", (0.9, lam_probe)),
        cn.verify_equivalence(s.t, s.event, "exponential", (lam_probe,)),
    ]
    res["likelihood_check"] = checks
    for c in checks:
        _log(f"likelihood {c['kind']:12s} passed={c['passed']} max_abs_err={c['max_abs_err']:.2e}")
    assert all(c["passed"] for c in checks), "手刻似然與 pm.Censored 不一致"

    # ── 3. 三個基準分佈（全資料）─────────────────────────────────────────
    idatas, conv = {}, {}
    for kind in M.KINDS:
        idatas[kind] = M.sample(M.build(s, kind), draws=DRAWS, tune=TUNE, chains=CHAINS)
        conv[kind] = M.check_convergence(idatas[kind], label=kind)
        _log(f"{kind:12s} div={conv[kind]['divergent']} r_hat={conv[kind]['max_rhat']:.3f} "
             f"ess={conv[kind]['min_ess_bulk']:.0f}")
    res["convergence"] = conv

    import arviz as az

    sm = az.summary(idatas["weibull"], var_names=["beta0", "beta", "k"])
    res["weibull_posterior"] = {
        str(i): {"mean": float(r["mean"]), "sd": float(r["sd"]),
                 "hdi_3%": float(r["hdi_3%"]), "hdi_97%": float(r["hdi_97%"]),
                 "r_hat": float(r["r_hat"])}
        for i, r in sm.iterrows()
    }
    k_draws = M.posterior_flat(idatas["weibull"], "k")
    res["weibull_shape_k"] = {
        "median": float(np.median(k_draws)),
        "lo95": float(np.percentile(k_draws, 2.5)),
        "hi95": float(np.percentile(k_draws, 97.5)),
        "p_k_gt_1": float((k_draws > 1).mean()),
    }
    _log(f"Weibull k = {res['weibull_shape_k']['median']:.3f} "
         f"[{res['weibull_shape_k']['lo95']:.3f}, {res['weibull_shape_k']['hi95']:.3f}], "
         f"P(k>1)={res['weibull_shape_k']['p_k_gt_1']:.4f}")

    # ── 4. 刪失處理三方對照（通關標準 1）────────────────────────────────
    km, km_info = fq.fit_km(s)
    arrays["km_timeline"] = km_info["timeline"]
    arrays["km_survival"] = km_info["survival"]
    arrays["km_lo"] = km_info["lo"]
    arrays["km_hi"] = km_info["hi"]

    variants = {
        "correct": s,
        "as_event": D.censored_as_event(s),
        "drop": D.drop_censored(s),
    }
    curves, medians = {}, {"km": km_info["median_survival"]}
    for name, sv in variants.items():
        censoring = M.CORRECT if name == "correct" else M.AS_EVENT
        idata = (idatas["weibull"] if name == "correct"
                 else M.sample(M.build(sv, "weibull", censoring=censoring),
                               draws=DRAWS, tune=TUNE, chains=CHAINS))
        M.check_convergence(idata, label=f"censoring:{name}")
        draws = M.population_survival(idata, s.X, T_GRID, "weibull", max_draws=600)
        curves[name] = (T_GRID, draws)
        arrays[f"pop_{name}"] = draws
        med = M.median_survival(idata, np.zeros(s.X.shape[1]), "weibull")
        medians[name] = {
            "median": float(np.median(med)),
            "lo": float(np.percentile(med, 2.5)),
            "hi": float(np.percentile(med, 97.5)),
            "n_used": sv.n, "events_used": sv.n_events,
        }
        bias = 100 * (medians[name]["median"] - km_info["median_survival"]) / km_info["median_survival"]
        medians[name]["bias_vs_km_pct"] = float(bias)
        _log(f"censoring[{name:9s}] median={medians[name]['median']:.2f} yr ({bias:+.1f}% vs KM)")
    res["censoring_handling"] = medians
    arrays["t_grid"] = T_GRID

    # ── 5. 頻率派對照 ────────────────────────────────────────────────────
    cph, cox_tbl = fq.fit_cox(s)
    aft, aft_tbl = fq.fit_weibull_aft(s)
    ph_test = fq.ph_assumption_test(cph, s)
    res["cox"] = json.loads(cox_tbl.to_json(orient="index"))
    res["weibull_aft_mle"] = json.loads(aft_tbl.to_json(orient="index"))
    res["ph_test"] = json.loads(ph_test.to_json(orient="index"))
    res["km_median_survival"] = km_info["median_survival"]
    res["weibull_aft_mle_rho"] = float(np.exp(aft.params_.loc[("rho_", "Intercept")]))
    _log(f"Cox fitted; PH test min p = {ph_test['p'].min():.3f}; "
         f"MLE rho={res['weibull_aft_mle_rho']:.3f} vs Bayes k={res['weibull_shape_k']['median']:.3f}")

    # AFT → PH 換算，與 Cox 並排
    ph_draws = M.aft_to_ph(idatas["weibull"], "weibull")
    arrays["ph_draws"] = ph_draws
    cox_ordered = cox_tbl.loc[s.features]
    arrays["cox_log_hr"] = cox_ordered["log_hr"].to_numpy()
    arrays["cox_lo95"] = cox_ordered["lo95"].to_numpy()
    arrays["cox_hi95"] = cox_ordered["hi95"].to_numpy()

    ph_med = np.median(ph_draws, axis=0)
    res["aft_to_ph_vs_cox"] = {
        f: {"bayes_ph_median": float(ph_med[i]),
            "bayes_lo95": float(np.percentile(ph_draws[:, i], 2.5)),
            "bayes_hi95": float(np.percentile(ph_draws[:, i], 97.5)),
            "cox_log_hr": float(cox_ordered["log_hr"].iloc[i]),
            "cox_lo95": float(cox_ordered["lo95"].iloc[i]),
            "cox_hi95": float(cox_ordered["hi95"].iloc[i]),
            "abs_diff": float(abs(ph_med[i] - cox_ordered["log_hr"].iloc[i])),
            "ci_width_ratio": float(
                (np.percentile(ph_draws[:, i], 97.5) - np.percentile(ph_draws[:, i], 2.5))
                / (cox_ordered["hi95"].iloc[i] - cox_ordered["lo95"].iloc[i])),
            }
        for i, f in enumerate(s.features)
    }
    diffs = [v["abs_diff"] for v in res["aft_to_ph_vs_cox"].values()]
    res["aft_to_ph_max_abs_diff"] = float(np.max(diffs))
    res["aft_to_ph_mean_abs_diff"] = float(np.mean(diffs))
    _log(f"AFT→PH vs Cox: mean |diff| = {res['aft_to_ph_mean_abs_diff']:.4f}, "
         f"max = {res['aft_to_ph_max_abs_diff']:.4f}")

    # ── 6. 個體存活曲線（通關標準 2）────────────────────────────────────
    risk = ev.bayes_risk_score(idatas["weibull"], s.X, "weibull")
    lo_i = int(np.argsort(risk)[int(0.05 * s.n)])
    hi_i = int(np.argsort(risk)[int(0.95 * s.n)])
    patients = []
    for tag, i in (("Low-risk patient", lo_i), ("High-risk patient", hi_i)):
        draws = M.survival_curves(idatas["weibull"], s.X[i], T_GRID, "weibull", max_draws=2000)
        cox_curve = fq.cox_survival_at(cph, s.X[i], s.features, T_GRID)
        med = M.median_survival(idatas["weibull"], s.X[i], "weibull")
        raw = s.df_raw.iloc[i]
        desc = (f"{int(np.expm1(raw['pnodes']))} positive nodes, "
                f"progesterone {int(np.expm1(raw['progrec']))} fmol, "
                f"hormone therapy: {'yes' if raw['horTh'] > 0.5 else 'no'}")
        patients.append({
            "name": tag, "desc": desc, "t_grid": T_GRID, "draws": draws, "cox": cox_curve,
            "med_post": (float(np.median(med)), float(np.percentile(med, 2.5)),
                         float(np.percentile(med, 97.5))),
        })
        arrays[f"patient_{'lo' if i == lo_i else 'hi'}_draws"] = draws
        arrays[f"patient_{'lo' if i == lo_i else 'hi'}_cox"] = cox_curve
        band_w = float(np.percentile(draws[:, 40], 97.5) - np.percentile(draws[:, 40], 2.5))
        res.setdefault("patients", {})[tag] = {
            "index": i, "desc": desc,
            "median_survival": patients[-1]["med_post"],
            "band_width_at_t": {"t": float(T_GRID[40]), "width": band_w},
        }
        _log(f"{tag}: median {patients[-1]['med_post'][0]:.2f} yr, "
             f"95% band width at t={T_GRID[40]:.1f}yr = {band_w:.3f}")
    res["patient_meta"] = {"low_index": lo_i, "high_index": hi_i,
                           "desc_low": patients[0]["desc"], "desc_high": patients[1]["desc"]}

    # ── 7. 模型比較：WAIC/LOO + 5-fold OOF（通關標準 3）─────────────────
    ic = ev.ic_table(idatas)
    cmp_df = ev.compare(idatas)
    res["ic_table"] = json.loads(ic.to_json(orient="index"))
    res["az_compare"] = json.loads(cmp_df.to_json(orient="index"))
    best, second = ic.index[0], ic.index[1]
    elpd_diff = float(ic.loc[best, "elpd_waic"] - ic.loc[second, "elpd_waic"])
    se_diff = float(cmp_df.loc[second, "dse"])
    res["waic_winner"] = {
        "best": best, "second": second, "elpd_diff": elpd_diff, "se_diff": se_diff,
        "sigma": elpd_diff / se_diff if se_diff > 0 else float("nan"),
    }
    _log(f"WAIC winner: {best} (Δelpd={elpd_diff:.2f}, dse={se_diff:.2f} → "
         f"{res['waic_winner']['sigma']:.1f}σ); max Pareto k={ic['max_pareto_k'].max():.2f}")

    cv = {}
    for kind in M.KINDS:
        r = ev.cross_val(s, kind, draws=1200, tune=1200)
        boot = ev.c_index_bootstrap(s.t, s.event, r["oof_risk"], n_boot=500)
        cv[kind] = {"c_index": r["c_index"], "c_boot_se": boot["se"],
                    "c_lo95": boot["lo95"], "c_hi95": boot["hi95"],
                    "ibs": r["ibs"], "comparable_pairs": r["comparable_pairs"],
                    "total_pairs": r["total_pairs"]}
        arrays[f"oof_risk_{kind}"] = r["oof_risk"]
        arrays[f"brier_{kind}"] = r["brier"]
        arrays["brier_times"] = r["brier_times"]
        _log(f"OOF {kind:12s} C-index={r['c_index']:.4f}±{boot['se']:.4f} IBS={r['ibs']:.4f}")

    # Cox 的 OOF C-index
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof_cox = np.zeros(s.n)
    for tr, te in skf.split(s.X, s.event):
        s_tr = D.Survival(X=s.X[tr], t=s.t[tr], event=s.event[tr], features=s.features)
        s_te = D.Survival(X=s.X[te], t=s.t[te], event=s.event[te], features=s.features)
        c, _ = fq.fit_cox(s_tr)
        oof_cox[te] = c.predict_partial_hazard(fq.to_frame(s_te)).to_numpy()
    ci_cox, _ = ev.c_index(s.t, s.event, oof_cox)
    boot_cox = ev.c_index_bootstrap(s.t, s.event, oof_cox, n_boot=500)
    cv["cox"] = {"c_index": ci_cox, "c_boot_se": boot_cox["se"],
                 "c_lo95": boot_cox["lo95"], "c_hi95": boot_cox["hi95"]}
    arrays["oof_risk_cox"] = oof_cox
    _log(f"OOF cox          C-index={ci_cox:.4f}±{boot_cox['se']:.4f}")

    # 配對 bootstrap：WAIC 的贏家在判別力上真的贏嗎？
    pair = ev.paired_c_index_diff(s.t, s.event, arrays[f"oof_risk_{best}"],
                                  arrays[f"oof_risk_{second}"], n_boot=500)
    res["paired_c_diff_best_vs_second"] = {
        "best": best, "second": second, **{k: v for k, v in pair.items()}}
    pair_cox = ev.paired_c_index_diff(s.t, s.event, arrays[f"oof_risk_{best}"], -oof_cox, n_boot=500)
    res["paired_c_diff_best_vs_cox"] = pair_cox
    res["cv"] = cv
    _log(f"paired ΔC-index {best} vs {second}: {pair['mean_diff']:+.4f} "
         f"[{pair['lo95']:+.4f}, {pair['hi95']:+.4f}], P(better)={pair['p_a_better']:.2f}")

    # ── 8. 風險函數形狀 —— WAIC 排序的機制 ──────────────────────────────
    x_ref = np.zeros(s.X.shape[1])
    ind_hazards, pop_hazards = {}, {}
    for kind in M.KINDS:
        hi_ = ev.hazard_curve(idatas[kind], x_ref, T_GRID, kind, max_draws=600)
        hp_ = ev.population_hazard_curve(idatas[kind], s.X, T_GRID, kind, max_draws=400)
        ind_hazards[kind], pop_hazards[kind] = hi_, hp_
        arrays[f"hazard_ind_{kind}"] = hi_
        arrays[f"hazard_pop_{kind}"] = hp_
    na_t, na_h, na_info = ev.nelson_aalen_hazard(s.t, s.event)
    arrays["na_t"], arrays["na_h"] = na_t, na_h
    arrays["na_at_risk"] = na_info["at_risk"]
    arrays["na_reliable"] = na_info["reliable"]
    na_peak = float(na_t[np.nanargmax(na_h)])

    # NA 峰值對核寬度的敏感度 —— 確認 1.4 年不是平滑參數的產物
    na_peak_bw = {}
    for bw in (0.3, 0.45, 0.6, 0.9):
        tt, hh, _ = ev.nelson_aalen_hazard(s.t, s.event, bandwidth=bw)
        na_peak_bw[str(bw)] = float(tt[np.nanargmax(hh)])

    peaks = {"na_peak": na_peak, "na_peak_by_bandwidth": na_peak_bw,
             "na_min_at_risk": na_info["min_at_risk"], "na_t_cut": na_info["t_cut"]}
    # NA 尾端的「第二個峰」是不是噪聲？記錄晚期的事件稀少程度
    late = {}
    for lo, hi in ((4.0, 5.0), (5.0, 6.0), (6.0, 7.3)):
        m = (s.t >= lo) & (s.t < hi)
        late[f"{lo:g}-{hi:g}yr"] = {"n_obs": int(m.sum()), "n_events": int(s.event[m].sum()),
                                    "at_risk_at_lo": int((s.t >= lo).sum())}
    res["late_followup_sparsity"] = late
    for kind in M.KINDS:
        pk_draws = T_GRID[np.argmax(pop_hazards[kind], axis=1)]
        peaks[f"{kind}_pop"] = {
            "median": float(np.median(pk_draws)),
            "lo": float(np.percentile(pk_draws, 2.5)),
            "hi": float(np.percentile(pk_draws, 97.5)),
        }
        peaks[f"{kind}_ind_peak"] = float(T_GRID[np.argmax(np.median(ind_hazards[kind], axis=0))])
        arrays[f"peak_pop_draws_{kind}"] = pk_draws
    lp = peaks["lognormal_pop"]
    peaks["lognormal_pop_covers_na"] = bool(lp["lo"] <= na_peak <= lp["hi"])
    res["hazard_peaks"] = peaks
    _log(f"hazard peaks: log-normal individual {peaks['lognormal_ind_peak']:.2f} yr, "
         f"population {lp['median']:.2f} yr [{lp['lo']:.2f}, {lp['hi']:.2f}]; "
         f"nonparametric {na_peak:.2f} yr (bw-stable: {na_peak_bw}); "
         f"CrI covers nonparametric: {peaks['lognormal_pop_covers_na']}")

    # ── 9. 資訊代價實驗 ──────────────────────────────────────────────────
    _log("information-cost experiment (dual arm)...")
    raw = inf.run_information_experiment(s, seeds=(0, 1, 2), draws=1200, tune=1200)
    summary = inf.summarize_arms(raw, "sd_pnodes")
    # 偏差的基準是全資料後驗均值 —— 「正確答案」的最佳代理
    baseline_pnodes = float(
        M.posterior_flat(idatas["weibull"], "beta")[:, s.features.index("pnodes")].mean())
    gap = inf.arm_gap(summary, baseline_mean=baseline_pnodes)
    res["information"] = {
        "summary": {a: {"loglog_slope": summary[a]["loglog_slope"],
                        "points": summary[a]["points"]} for a in summary},
        "gap": {k: v for k, v in gap.items()},
        "baseline_pnodes": baseline_pnodes,
        "total_retries": int(sum(r["retries"] for arm in raw.values() for r in arm)),
    }
    _log(f"info-cost slopes: " + ", ".join(
        f"{a} {summary[a]['loglog_slope']:+.3f}" for a in summary))
    _log(f"info-cost bias/sd (mean over settings): " + ", ".join(
        f"{a}: |bias|={gap[f'mean_abs_bias_pct_{a}']:.1f}% sd={gap[f'mean_sd_{a}']:.4f}"
        for a in summary))
    _log(f"C narrowest AND most biased in "
         f"{gap.get('n_horizons_C_narrowest_and_most_biased')}/{gap.get('n_horizons')} settings; "
         f"retries: {res['information']['total_retries']}")

    # ── 10. 先驗敏感度 ───────────────────────────────────────────────────
    sens = {}
    for beta_sd in (0.25, 0.5, 1.0, 2.0):
        idata = M.sample(M.build(s, "weibull", beta_sd=beta_sd),
                         draws=1500, tune=1500, chains=2)
        M.check_convergence(idata, label=f"prior sd={beta_sd}")
        b = M.posterior_flat(idata, "beta")[:, s.features.index("pnodes")]
        med = M.median_survival(idata, np.zeros(s.X.shape[1]), "weibull")
        sens[str(beta_sd)] = {
            "pnodes_mean": float(b.mean()), "pnodes_sd": float(b.std(ddof=1)),
            "median_survival": float(np.median(med)),
        }
        _log(f"prior beta_sd={beta_sd}: pnodes={b.mean():+.4f}±{b.std(ddof=1):.4f}, "
             f"median={np.median(med):.2f} yr")
    base = sens["0.5"]["pnodes_mean"]
    res["prior_sensitivity"] = {
        "runs": sens,
        "max_rel_shift_pct": float(max(abs(v["pnodes_mean"] - base) / abs(base) * 100
                                       for v in sens.values())),
    }

    # ── 11. 落盤 ─────────────────────────────────────────────────────────
    np.savez_compressed(PLOTDATA, **arrays)
    res["_meta"] = {
        "runtime_sec": round(time.time() - t_start, 1),
        "draws": DRAWS, "tune": TUNE, "chains": CHAINS,
        "plotdata": os.path.relpath(PLOTDATA, ROOT),
    }
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=float)
    _log(f"saved {PLOTDATA} ({os.path.getsize(PLOTDATA) / 1e6:.1f} MB) and {RESULTS}")

    # ── 12. 出圖 ─────────────────────────────────────────────────────────
    make_figures(arrays, res, s, cox_ordered, ic, cv, curves, medians, km_info,
                 patients, pop_hazards, ind_hazards, peaks, summary, gap)
    _log(f"done in {res['_meta']['runtime_sec']:.0f}s")


def make_figures(arrays, res, s, cox_ordered, ic, cv, curves, medians, km_info,
                 patients, pop_hazards, ind_hazards, peaks, summary, gap):
    """出圖。replot.py 會用落盤資料重建同樣的呼叫。"""
    k_med = res["weibull_shape_k"]["median"]
    lam_ref = float(np.exp(res["weibull_posterior"]["beta0"]["mean"]))
    tg = np.linspace(0.02, 7.0, 300)
    t_obs = 2.5
    demo = {
        "t_grid": tg,
        "pdf": np.exp(cn.weibull_logpdf(tg, k_med, lam_ref)),
        "t_obs": t_obs,
        "pdf_at_t": float(np.exp(cn.weibull_logpdf(t_obs, k_med, lam_ref))),
        "sf_at_t": float(np.exp(cn.weibull_logsf(t_obs, k_med, lam_ref))),
    }
    P.censoring_likelihood(arrays["t"], arrays["event"], demo,
                           os.path.join(FIG_DIR, "01_censoring_likelihood.png"))
    P.censoring_handling(km_info, curves, medians,
                         os.path.join(FIG_DIR, "02_censoring_handling.png"))
    P.individual_survival(patients, os.path.join(FIG_DIR, "03_individual_survival.png"))
    P.model_comparison(ic, cv, os.path.join(FIG_DIR, "04_model_comparison.png"))
    P.hazard_shapes(arrays["t_grid"], pop_hazards, ind_hazards,
                    (arrays["na_t"], arrays["na_h"],
                     {"at_risk": arrays["na_at_risk"], "reliable": arrays["na_reliable"],
                      "min_at_risk": res["hazard_peaks"]["na_min_at_risk"],
                      "t_cut": res["hazard_peaks"]["na_t_cut"]}), peaks,
                    os.path.join(FIG_DIR, "05_hazard_shapes.png"))
    P.bayes_vs_cox([D.LABELS[f] for f in s.features], arrays["ph_draws"], cox_ordered,
                   os.path.join(FIG_DIR, "06_bayes_vs_cox.png"))
    P.information_cost(summary, gap, os.path.join(FIG_DIR, "07_information_cost.png"))
    _log("figures written")


if __name__ == "__main__":
    main()
