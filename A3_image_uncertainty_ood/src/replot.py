"""
A3 · 從落盤資料重畫全部圖表：`python src/replot.py`

只讀 `data/A_medical/a3_plotdata.npz` + `figures/results.json`，
不訓練任何模型、不跑任何前向傳播 —— 秒級完成。
調配色、標籤、版面都不必重跑完整管線。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiments as X
import plots as P
import run_all as R

UNC_KEYS = ("total", "aleatoric", "epistemic", "max_prob")


class _Bag:
    """讓 plots.sample_grid 能吃落盤的樣本陣列（它只用 .x / .n）。"""

    def __init__(self, x):
        self.x = x
        self.n = len(x)


def main():
    z = np.load(R.PLOTDATA)
    res = json.load(open(R.RESULTS))

    t1_ood = ["fashion", "derma", "derma_gray"]
    t2_ood = ["path_held"]

    def unc(prefix, who, method, oname=None):
        base = f"{prefix}_in_{method}" if oname is None else f"{prefix}_ood_{oname}_{method}"
        return {k: z[f"{base}_{k}"] for k in UNC_KEYS}

    in_unc_t1 = {m: unc("t1", "in", m) for m in X.METHODS}
    in_unc_t2 = {m: unc("t2", "in", m) for m in X.METHODS}
    ood_unc = {o: {m: unc("t1", "ood", m, o) for m in X.METHODS} for o in t1_ood}
    ood_unc["path_held"] = {m: unc("t2", "ood", m, "path_held") for m in X.METHODS}

    conf_in_t1 = {m: z[f"t1_conf_in_{m}"] for m in X.METHODS}
    conf_in_t2 = {m: z[f"t2_conf_in_{m}"] for m in X.METHODS}
    conf_ood = {o: {m: z[f"t1_conf_ood_{o}_{m}"] for m in X.METHODS} for o in t1_ood}
    conf_ood["path_held"] = {m: z[f"t2_conf_ood_path_held_{m}"] for m in X.METHODS}

    in_unc_by_ood = {o: in_unc_t1 for o in t1_ood}
    in_unc_by_ood["path_held"] = in_unc_t2
    conf_in_by_ood = {o: conf_in_t1 for o in t1_ood}
    conf_in_by_ood["path_held"] = conf_in_t2

    F = R.FIG_DIR
    P.uncertainty_histograms_multi(in_unc_by_ood, ood_unc, conf_in_by_ood, conf_ood,
                                   res["auroc_matrix"],
                                   os.path.join(F, "01_uncertainty_histograms.png"),
                                   method="ensemble")

    trivial_all = {**res["trivial_baseline"]["track1"], **res["trivial_baseline"]["track2"]}
    P.auroc_matrix(res["auroc_matrix"], trivial_all, os.path.join(F, "02_auroc_matrix.png"))
    P.decomposition_validation(res["decomposition_validation"],
                               os.path.join(F, "03_decomposition_validation.png"))

    sel = {"accuracy": res["selective_prediction"]["accuracy"],
           "oracle_aurc": res["selective_prediction"]["oracle_aurc"], "curves": {}}
    for k, v in res["selective_prediction"]["summary"].items():
        tag = k.replace("/", "_")
        sel["curves"][k] = {"coverage": z[f"rc_{tag}_coverage"],
                            "accuracy": z[f"rc_{tag}_accuracy"], **v}
    P.risk_coverage(sel, os.path.join(F, "04_risk_coverage.png"))
    P.cost_benefit(res["cost_benefit"], os.path.join(F, "05_cost_benefit.png"))
    P.calibration(res["calibration"], res["accuracy"]["track1"],
                  os.path.join(F, "06_calibration.png"))
    P.sample_grid({tag: _Bag(z[f"sample_{tag}"]) for tag in
                   ("in_pneu", "fashion", "derma", "derma_gray", "in_path", "path_held")},
                  os.path.join(F, "07_samples.png"))
    print(f"replotted 7 figures into {F}")


if __name__ == "__main__":
    main()
