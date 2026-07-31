"""
C2 · 從落盤資料重畫全部圖表：`python src/replot.py`

只讀 `data/C_epi/c2_plotdata.npz` + `figures/results.json`，
不跑任何 MCMC —— 秒級完成。調配色、標籤、版面都不必重跑 40 分鐘的推論。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plots as P
import run_all as R

QK = ("median", "lo95", "hi95", "lo50", "hi50")


def main():
    z = np.load(R.PLOTDATA, allow_pickle=False)
    res = json.load(open(R.RESULTS))
    F = R.FIG_DIR

    policies = {k: int(v) for k, v in res["data"]["policy_indices"].items()}

    # 圖 1
    P.data_overview(z["dates_all"], z["cases_all"], policies, res["dow_profile"],
                    res["source_cross_validation"],
                    os.path.join(F, "01_data_overview.png"))

    # 圖 2
    demo = {"new_infections": z["demo_new_infections"], "S_frac": z["demo_S_frac"],
            "R_t": z["demo_R_t"], "R0": 2.8}
    P.sir_verification(res["sir_verification"], demo,
                       os.path.join(F, "02_sir_verification.png"))

    # 圖 3
    fits = {v: {k: z[f"pp_{v}_{k}"] for k in QK}
            for v in ("constant", "timevarying", "tv_no_dow")
            if f"pp_{v}_median" in z}
    P.fixed_vs_timevarying(z["dates_train"], z["cases_train"], fits, res["params"],
                           os.path.join(F, "03_fixed_vs_timevarying.png"))

    # 圖 4
    rt = {k: z[f"rt_{k}"] for k in (*QK, "p_below_1")}
    P.rt_and_policies(z["dates_train"], rt, res["rt_policies"],
                      os.path.join(F, "04_rt_and_policies.png"))

    # 圖 5
    w = {**res["rt_width_vs_cases"], "width_rel": z["w_width_rel"],
         "width_abs": z["w_width_abs"], "rt_median": z["w_rt_median"],
         "cases": z["cases_train"]}
    P.uncertainty_vs_data(w, os.path.join(F, "05_uncertainty_vs_data.png"))

    # 圖 6
    fc_q = {k: z[f"fc_{k}"] for k in QK}
    P.posterior_predictive(z["dates_train"], z["cases_train"], fits["timevarying"],
                           z["dates_test"], z["cases_test"], fc_q,
                           res["coverage_in_sample"], res["coverage_out_of_sample"],
                           os.path.join(F, "06_posterior_predictive.png"))

    # 圖 7
    acf_with = {**res["residual_periodicity"]["with_dow"],
                "acf": z["acf_with"], "lags": z["acf_lags"]}
    acf_without = {**res["residual_periodicity"]["without_dow"],
                   "acf": z["acf_without"], "lags": z["acf_lags"]}
    P.dow_and_comparison(res["dow_posterior"], acf_with, acf_without,
                         res["model_comparison"],
                         os.path.join(F, "07_dow_and_comparison.png"))

    print(f"replotted 7 figures into {F}")


if __name__ == "__main__":
    main()
