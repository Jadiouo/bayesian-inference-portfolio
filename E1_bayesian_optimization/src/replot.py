"""
E1 · 從落盤資料重畫全部圖表：`python src/replot.py`

只讀 `data/E_experiment/e1_plotdata.npz` + `figures/results.json`，
不跑任何 GP 或 BO —— 秒級完成。調配色、標籤、版面都不必重跑 20 分鐘的實驗。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plots as P
import run_all as R

TRACE_KEYS = ("mean", "sd", "median", "q25", "q75", "finals")


def _load_traces(z, prefix: str, scalars: dict) -> dict:
    out = dict(scalars)
    for k in TRACE_KEYS:
        key = f"{prefix}_{k}"
        if key in z:
            out[k] = z[key]
    return out


def main():
    z = np.load(R.PLOTDATA)
    res = json.load(open(R.RESULTS))

    # ── 圖 1：GP 驗證 ────────────────────────────────────────────────────
    demo = {"xs": z["demo_xs"], "mu": z["demo_mu"], "sd": z["demo_sd"],
            "samples": z["demo_samples"], "X": z["demo_X"], "y": z["demo_y"],
            "truth": z["demo_truth"], "kernel": res["gp_demo_hyperparams"]["kernel"],
            "lengthscale": res["gp_demo_hyperparams"]["lengthscale"]}
    P.gp_verification(demo, res["gp_verification"]["vs_sklearn"],
                      res["gp_verification"]["gradients"],
                      os.path.join(R.FIG_DIR, "01_gp_verification.png"))

    # ── 圖 2：收斂曲線 ──────────────────────────────────────────────────
    benchmarks = []
    for bm in res["benchmarks"]:
        methods = {}
        for m, sc in bm["methods"].items():
            methods[m] = _load_traces(z, f"bench_{bm['objective']}_{m.replace('/', '_')}", sc)
        benchmarks.append({**{k: v for k, v in bm.items() if k != "methods"},
                           "methods": methods})
    P.convergence(benchmarks, os.path.join(R.FIG_DIR, "02_convergence.png"))

    # ── 圖 3：κ 效應 ────────────────────────────────────────────────────
    ks = res["kappa_sweep"]
    sweep = {"objective": ks["objective"], "n_total": ks["n_total"],
             "kappas": ks["kappas"],
             "random": _load_traces(z, "kappa_random", ks["random"]),
             "runs": {kn: _load_traces(z, f"kappa_{kn.replace('=', '')}", sc)
                      for kn, sc in ks["runs"].items()}}
    kt = res["kappa_trajectory"]
    traj = {"objective": kt["objective"], "seed": kt["seed"], "n_init": kt["n_init"],
            "x_min": np.asarray(kt["x_min"]) if kt["x_min"] is not None else None,
            "grid_x": z["traj_grid_x"], "Z": z["traj_Z"],
            "trajectories": {kn: {**sc, "X": z[f"traj_X_{kn.replace('=', '')}"],
                                  "y": z[f"traj_y_{kn.replace('=', '')}"]}
                             for kn, sc in kt["trajectories"].items()}}
    P.kappa_effect(sweep, traj, os.path.join(R.FIG_DIR, "03_kappa_effect.png"))

    # ── 圖 4：維度詛咒 ──────────────────────────────────────────────────
    dcr = res["dimension_curse"]
    dc = {k: v for k, v in dcr.items() if k != "per_dim"}
    dc["per_dim"] = {}
    for d, e in dcr["per_dim"].items():
        dc["per_dim"][d] = {**{k: v for k, v in e.items() if k not in ("bo", "random")},
                            "bo": _load_traces(z, f"dim{d}_bo", e["bo"]),
                            "random": _load_traces(z, f"dim{d}_random", e["random"])}
    P.dimension_curse(dc, os.path.join(R.FIG_DIR, "04_dimension_curse.png"))

    # ── 圖 5：機制 ──────────────────────────────────────────────────────
    P.curse_mechanism(res["kernel_concentration"], res["acq_shortfall"],
                      os.path.join(R.FIG_DIR, "05_curse_mechanism.png"))

    # ── 圖 6：核函數 ────────────────────────────────────────────────────
    kcr = res["kernel_comparison"]
    kc = {"kernels": kcr["kernels"], "n_total": kcr["n_total"], "acq": kcr["acq"],
          "per_objective": {}}
    for o, e in kcr["per_objective"].items():
        kc["per_objective"][o] = {
            **{k: v for k, v in e.items() if k != "kernels"},
            "kernels": {kn: _load_traces(z, f"kern_{o}_{kn}", sc)
                        for kn, sc in e["kernels"].items()}}
    P.kernel_comparison(kc, os.path.join(R.FIG_DIR, "06_kernel_comparison.png"))

    # ── 圖 7：混凝土 ────────────────────────────────────────────────────
    cr = res["concrete"]
    ca = {**{k: v for k, v in cr.items() if k not in ("bo", "random")},
          "bo": _load_traces(z, "concrete_bo", cr["bo"]),
          "random": _load_traces(z, "concrete_random", cr["random"])}
    P.concrete(ca, os.path.join(R.FIG_DIR, "07_concrete.png"))

    print(f"replotted 7 figures into {R.FIG_DIR}")


if __name__ == "__main__":
    main()
