"""
A2 · 從落盤資料重畫全部圖表：`python src/replot.py`

沿用 B2 的設計原則：推論結果與出圖資料分離。這支程式只讀
`data/A_medical/a2_plotdata.npz` + `figures/results.json`，
不跑任何 MCMC，秒級完成 —— 調整配色、標籤、版面都不必重跑 4 分鐘的推論。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import censoring as cn
import data as D
import models as M
import plots as P
import run_all as R


def main():
    z = np.load(R.PLOTDATA)
    res = json.load(open(R.RESULTS))
    features = res["data"]["features"]
    T_GRID = z["t_grid"]

    km_info = {"timeline": z["km_timeline"], "survival": z["km_survival"],
               "lo": z["km_lo"], "hi": z["km_hi"],
               "median_survival": res["km_median_survival"]}
    curves = {name: (T_GRID, z[f"pop_{name}"]) for name in ("correct", "as_event", "drop")}
    medians = res["censoring_handling"]

    patients = [
        {"name": "Low-risk patient", "desc": res["patient_meta"]["desc_low"],
         "t_grid": T_GRID, "draws": z["patient_lo_draws"], "cox": z["patient_lo_cox"],
         "med_post": tuple(res["patients"]["Low-risk patient"]["median_survival"])},
        {"name": "High-risk patient", "desc": res["patient_meta"]["desc_high"],
         "t_grid": T_GRID, "draws": z["patient_hi_draws"], "cox": z["patient_hi_cox"],
         "med_post": tuple(res["patients"]["High-risk patient"]["median_survival"])},
    ]

    ic = pd.DataFrame(res["ic_table"]).T
    ic = ic.loc[ic["elpd_waic"].sort_values(ascending=False).index]
    cv = res["cv"]
    ind_hazards = {k: z[f"hazard_ind_{k}"] for k in M.KINDS}
    pop_hazards = {k: z[f"hazard_pop_{k}"] for k in M.KINDS}
    peaks = res["hazard_peaks"]
    cox_tbl = pd.DataFrame({"log_hr": z["cox_log_hr"], "lo95": z["cox_lo95"],
                            "hi95": z["cox_hi95"]}, index=features)

    summary = {a: v for a, v in res["information"]["summary"].items()}
    gap = res["information"]["gap"]

    k_med = res["weibull_shape_k"]["median"]
    lam_ref = float(np.exp(res["weibull_posterior"]["beta0"]["mean"]))
    tg = np.linspace(0.02, 7.0, 300)
    t_obs = 2.5
    demo = {"t_grid": tg, "pdf": np.exp(cn.weibull_logpdf(tg, k_med, lam_ref)),
            "t_obs": t_obs,
            "pdf_at_t": float(np.exp(cn.weibull_logpdf(t_obs, k_med, lam_ref))),
            "sf_at_t": float(np.exp(cn.weibull_logsf(t_obs, k_med, lam_ref)))}

    F = R.FIG_DIR
    P.censoring_likelihood(z["t"], z["event"], demo, os.path.join(F, "01_censoring_likelihood.png"))
    P.censoring_handling(km_info, curves, medians, os.path.join(F, "02_censoring_handling.png"))
    P.individual_survival(patients, os.path.join(F, "03_individual_survival.png"))
    P.model_comparison(ic, cv, os.path.join(F, "04_model_comparison.png"))
    na_pack = (z["na_t"], z["na_h"],
               {"at_risk": z["na_at_risk"], "reliable": z["na_reliable"],
                "min_at_risk": peaks["na_min_at_risk"], "t_cut": peaks["na_t_cut"]})
    P.hazard_shapes(T_GRID, pop_hazards, ind_hazards, na_pack, peaks,
                    os.path.join(F, "05_hazard_shapes.png"))
    P.bayes_vs_cox([D.LABELS[f] for f in features], z["ph_draws"], cox_tbl,
                   os.path.join(F, "06_bayes_vs_cox.png"))
    P.information_cost(summary, gap, os.path.join(F, "07_information_cost.png"))
    print(f"replotted 7 figures into {F}")


if __name__ == "__main__":
    main()
