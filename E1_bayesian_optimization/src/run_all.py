"""
E1 · 一鍵重現：`python src/run_all.py`

流程：手刻 GP 驗證 → 標準測試函數比較 → κ 敏感度 → 維度詛咒（含機制診斷）
      → 核函數比較 → 混凝土配方最優化 → 落盤 → 出圖

沿用 A2/A3/B2 的設計原則：**推論結果與出圖資料落盤分離**。
繪圖資料寫進 data/E_experiment/e1_plotdata.npz，純量結果寫進 figures/results.json，
之後調整圖表只要 `python src/replot.py`，不必重跑實驗。

執行時間參考（本機 CPU）：約 15–20 分鐘，其中維度詛咒實驗佔一半以上
（20D 的 GP 超參數優化最慢）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bo as B
import experiments as X
import gp as G
import objectives as O
import plots as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "data", "E_experiment"))
FIG_DIR = os.path.join(ROOT, "figures")
PLOTDATA = os.path.join(DATA_DIR, "e1_plotdata.npz")
RESULTS = os.path.join(FIG_DIR, "results.json")

N_TOTAL = 30          # 「我只有 30 次機會」
N_INIT = 5
DIMS = (2, 5, 10, 20, 50)
KAPPAS = (0.0, 0.5, 2.0, 5.0, 20.0)


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


def _store_traces(prefix: str, agg: dict, arrays: dict):
    for k in ("mean", "sd", "median", "q25", "q75", "finals"):
        arrays[f"{prefix}_{k}"] = np.asarray(agg[k])


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    res: dict = {"config": {"n_total": N_TOTAL, "n_init": N_INIT, "dims": list(DIMS),
                            "kappas": list(KAPPAS),
                            "n_seeds_main": X.N_SEEDS_MAIN,
                            "n_seeds_dims": X.N_SEEDS_DIMS}}
    arrays: dict = {}

    # ── 1 · 手刻 GP 的正確性 ─────────────────────────────────────────────
    rng = np.random.default_rng(0)
    Xv = rng.uniform(0, 1, (25, 3))
    yv = np.sin(3 * Xv[:, 0]) + 0.5 * Xv[:, 1] ** 2 - Xv[:, 2] + 0.05 * rng.standard_normal(25)
    checks = [G.verify_against_sklearn(Xv, yv, kernel=k) for k in ("RBF", "Matern52")]
    grads = [G.verify_gradient(Xv, yv, kernel=k) for k in ("RBF", "Matern52")]
    res["gp_verification"] = {"vs_sklearn": checks, "gradients": grads}
    for c in checks:
        _log(f"GP vs sklearn [{c['kernel']:9s}] passed={c['passed']} "
             f"rel_mean={c['max_rel_diff_mean']:.2e} rel_std={c['max_rel_diff_std']:.2e}")
    for g in grads:
        _log(f"GP gradient   [{g['kernel']:9s}] passed={g['passed']} "
             f"max_rel_err={g['max_rel_err']:.2e}")
    assert all(c["passed"] for c in checks), "手刻 GP 與 sklearn 不一致"
    assert all(g["passed"] for g in grads), "解析梯度與數值梯度不一致"

    # 1D 後驗示意（給圖 1）
    xs = np.linspace(0, 1, 300)
    Xd = np.array([[0.08], [0.22], [0.38], [0.55], [0.62], [0.88]])
    truth_fn = lambda t: np.sin(9 * t) * (1 - t) + 0.3 * t
    yd = truth_fn(Xd.ravel())
    gpd = G.GP(kernel="Matern52").fit(Xd, yd, optimize=True, n_restarts=5)
    mu, sd = gpd.predict(xs.reshape(-1, 1))
    samples = gpd.sample_posterior(xs.reshape(-1, 1), n_samples=5, seed=1)
    demo = {"xs": xs, "mu": mu, "sd": sd, "samples": samples, "X": Xd.ravel(), "y": yd,
            "truth": truth_fn(xs), "kernel": "Matern52",
            "lengthscale": gpd.hyperparams["lengthscale"]}
    for k in ("xs", "mu", "sd", "samples", "truth"):
        arrays[f"demo_{k}"] = np.asarray(demo[k])
    arrays["demo_X"] = Xd.ravel()
    arrays["demo_y"] = yd
    res["gp_demo_hyperparams"] = gpd.hyperparams
    _log(f"1D demo GP: ls={gpd.hyperparams['lengthscale']:.3f} "
         f"noise={gpd.hyperparams['noise']:.4f} lml={gpd.hyperparams['lml']:.3f}")

    # ── 2 · 標準測試函數 vs 隨機搜尋 ─────────────────────────────────────
    branin, hart = O.branin(), O.hartmann6()
    ack5 = O.ackley(5)
    benchmarks = []
    for obj in (branin, hart, ack5):
        _log(f"benchmark on {obj.name} ({obj.dim}D), {X.N_SEEDS_MAIN} seeds...")
        bm = X.benchmark_comparison(obj, n_total=N_TOTAL, n_init=N_INIT)
        benchmarks.append(bm)
        for m, agg in bm["methods"].items():
            _store_traces(f"bench_{obj.name}_{m.replace('/', '_')}", agg, arrays)
        best = min(bm["methods"].items(), key=lambda kv: kv[1]["final_mean"])
        _log(f"  {obj.name}: " + ", ".join(
            f"{m}={a['final_mean']:.4f}±{a['final_sd']:.4f}"
            for m, a in bm["methods"].items()) + f"  → best={best[0]}")
    res["benchmarks"] = [
        {k: v for k, v in bm.items() if k != "methods"} |
        {"methods": {m: {kk: vv for kk, vv in a.items() if kk not in
                         ("mean", "sd", "median", "q25", "q75", "finals")}
                     for m, a in bm["methods"].items()}}
        for bm in benchmarks]

    # ── 3 · κ 敏感度 ─────────────────────────────────────────────────────
    _log(f"kappa sweep on Branin, {X.N_SEEDS_MAIN} seeds...")
    sweep = X.kappa_sweep(branin, kappas=KAPPAS, n_total=N_TOTAL, n_init=N_INIT)
    for kname, agg in sweep["runs"].items():
        _store_traces(f"kappa_{kname.replace('=', '')}", agg, arrays)
    _store_traces("kappa_random", sweep["random"], arrays)
    res["kappa_sweep"] = {
        "objective": sweep["objective"], "n_total": sweep["n_total"],
        "kappas": sweep["kappas"],
        "random": {k: v for k, v in sweep["random"].items()
                   if k not in ("mean", "sd", "median", "q25", "q75", "finals")},
        "runs": {kn: {k: v for k, v in a.items()
                      if k not in ("mean", "sd", "median", "q25", "q75", "finals")}
                 for kn, a in sweep["runs"].items()}}
    for kn, a in sweep["runs"].items():
        _log(f"  {kn:11s} final={a['final_mean']:.4f}±{a['final_sd']:.4f} "
             f"explore_radius={a['explore_radius_mean']:.3f} "
             f"unique_frac={a['unique_frac_mean']:.3f}")
    _log(f"  random      final={sweep['random']['final_mean']:.4f}")

    traj = X.kappa_trajectory_2d(branin, kappas=(0.0, 2.0, 20.0), n_total=N_TOTAL,
                                 n_init=N_INIT, seed=3)
    arrays["traj_grid_x"] = traj["grid_x"]
    arrays["traj_Z"] = traj["Z"]
    for kn, tr in traj["trajectories"].items():
        arrays[f"traj_X_{kn.replace('=', '')}"] = tr["X"]
        arrays[f"traj_y_{kn.replace('=', '')}"] = tr["y"]
    res["kappa_trajectory"] = {
        "objective": traj["objective"], "seed": traj["seed"], "n_init": traj["n_init"],
        "x_min": traj["x_min"],
        "trajectories": {kn: {k: v for k, v in tr.items() if k not in ("X", "y")}
                         for kn, tr in traj["trajectories"].items()}}

    # ── 4 · 維度詛咒 + 機制 ──────────────────────────────────────────────
    _log(f"dimension curse on Ackley {DIMS}, {X.N_SEEDS_DIMS} seeds "
         f"(this is the slow part)...")
    dc = X.dimension_curse(dims=DIMS, n_total=50, n_init=10, n_seeds=X.N_SEEDS_DIMS)
    for d in DIMS:
        e = dc["per_dim"][str(d)]
        _store_traces(f"dim{d}_bo", e["bo"], arrays)
        _store_traces(f"dim{d}_random", e["random"], arrays)
        _log(f"  {d:2d}D: BO={e['bo']['final_mean']:.3f}±{e['bo']['final_sd']:.3f} "
             f"random={e['random']['final_mean']:.3f}±{e['random']['final_sd']:.3f} "
             f"paired_gain={e['paired_gain_mean']:+.3f}±{1.96 * e['paired_gain_se']:.3f} "
             f"BO_wins={100 * e['bo_wins_frac']:.0f}%")
    res["dimension_curse"] = {
        k: v for k, v in dc.items() if k != "per_dim"} | {
        "per_dim": {d: {"bo": {k: v for k, v in e["bo"].items()
                               if k not in ("mean", "sd", "median", "q25", "q75", "finals")},
                        "random": {k: v for k, v in e["random"].items()
                                   if k not in ("mean", "sd", "median", "q25", "q75", "finals")},
                        **{k: v for k, v in e.items() if k not in ("bo", "random")}}
                    for d, e in dc["per_dim"].items()}}
    _log(f"  → critical dimension = {dc['critical_dim']}")

    _log("curse mechanism: kernel concentration + acquisition flatness...")
    conc = X.kernel_concentration(dims=DIMS)
    short = X.acq_optimizer_shortfall(dims=DIMS)
    res["kernel_concentration"] = conc
    res["acq_shortfall"] = short
    for r in conc["rows"]:
        _log(f"  {r['dim']:2d}D: k_median={r['kernel_median']:.2e} "
             f"frac(k>0.1)={r['frac_above_0.1']:.4f} dist_rel_sd={r['dist_rel_sd']:.4f}")
    for e in short["per_dim"]:
        _log(f"  {e['dim']:2d}D: mu_spread/y_range={e['mu_spread_over_yrange_mean']:.4f} "
             f"best_EI/y_range={e['best_ei_over_yrange_mean']:.2e} "
             f"inner_shortfall={e['shortfall_at_smallest_mean']:+.5f}")

    # ── 5 · 核函數就是先驗 ───────────────────────────────────────────────
    _log(f"kernel comparison (RBF vs Matern52), {X.N_SEEDS_MAIN} seeds...")
    kc = X.kernel_comparison([branin, ack5], n_total=N_TOTAL, n_init=N_INIT)
    for oname, e in kc["per_objective"].items():
        for kname, agg in e["kernels"].items():
            _store_traces(f"kern_{oname}_{kname}", agg, arrays)
        _log(f"  {oname}: " + ", ".join(
            f"{kn}={a['final_mean']:.4f} (ls={a['final_lengthscale_mean']:.3f})"
            for kn, a in e["kernels"].items()) +
            f"  paired Δ={e['paired_diff_mean']:+.4f}±{1.96 * e['paired_diff_se']:.4f}"
            f"  → {e['better']}")
    res["kernel_comparison"] = {
        "kernels": kc["kernels"], "n_total": kc["n_total"], "acq": kc["acq"],
        "per_objective": {o: {"dim": e["dim"], "f_min": e["f_min"],
                              "paired_diff_mean": e["paired_diff_mean"],
                              "paired_diff_se": e["paired_diff_se"],
                              "better": e["better"],
                              "kernels": {kn: {k: v for k, v in a.items()
                                               if k not in ("mean", "sd", "median",
                                                            "q25", "q75", "finals")}
                                          for kn, a in e["kernels"].items()}}
                          for o, e in kc["per_objective"].items()}}

    # ── 6 · 真實應用：混凝土配方 ─────────────────────────────────────────
    _log(f"concrete mix optimisation, {X.N_SEEDS_MAIN} seeds...")
    ca = X.concrete_application(DATA_DIR, n_total=40, n_init=8, n_seeds=X.N_SEEDS_MAIN)
    _store_traces("concrete_bo", ca["bo"], arrays)
    _store_traces("concrete_random", ca["random"], arrays)
    res["concrete"] = {k: v for k, v in ca.items() if k not in ("bo", "random")} | {
        "bo": {k: v for k, v in ca["bo"].items()
               if k not in ("mean", "sd", "median", "q25", "q75", "finals")},
        "random": {k: v for k, v in ca["random"].items()
                   if k not in ("mean", "sd", "median", "q25", "q75", "finals")}}
    _log(f"  BO {ca['bo_strength_mean']:.2f}±{ca['bo_strength_sd']:.2f} MPa vs "
         f"random {ca['random_strength_mean']:.2f}±{ca['random_strength_sd']:.2f} MPa")
    _log(f"  BO matches random's final level after {ca['evals_bo_to_match_random_final']} "
         f"of {ca['n_total']} evals → saves {ca['evals_saved']} ({ca['saved_pct']:.0f}%)")
    _log(f"  paired gain {ca['paired_gain_mean_mpa']:+.2f}±"
         f"{1.96 * ca['paired_gain_se_mpa']:.2f} MPa, BO wins "
         f"{100 * ca['bo_wins_frac']:.0f}%; best recipe {ca['best_strength_found']:.1f} MPa")

    # ── 落盤 ─────────────────────────────────────────────────────────────
    np.savez_compressed(PLOTDATA, **arrays)
    res["_meta"] = {"runtime_sec": round(time.time() - t0, 1),
                    "plotdata": os.path.relpath(PLOTDATA, ROOT)}
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=_jsonify)
    _log(f"saved {PLOTDATA} ({os.path.getsize(PLOTDATA) / 1e6:.2f} MB) and {RESULTS}")

    make_figures(demo, checks, grads, benchmarks, sweep, traj, dc, conc, short, kc, ca)
    _log(f"done in {res['_meta']['runtime_sec']:.0f}s")


def make_figures(demo, checks, grads, benchmarks, sweep, traj, dc, conc, short, kc, ca):
    P.gp_verification(demo, checks, grads, os.path.join(FIG_DIR, "01_gp_verification.png"))
    P.convergence(benchmarks, os.path.join(FIG_DIR, "02_convergence.png"))
    P.kappa_effect(sweep, traj, os.path.join(FIG_DIR, "03_kappa_effect.png"))
    P.dimension_curse(dc, os.path.join(FIG_DIR, "04_dimension_curse.png"))
    P.curse_mechanism(conc, short, os.path.join(FIG_DIR, "05_curse_mechanism.png"))
    P.kernel_comparison(kc, os.path.join(FIG_DIR, "06_kernel_comparison.png"))
    P.concrete(ca, os.path.join(FIG_DIR, "07_concrete.png"))
    _log("figures written")


if __name__ == "__main__":
    main()
