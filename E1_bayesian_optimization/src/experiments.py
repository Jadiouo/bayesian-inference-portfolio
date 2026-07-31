"""
E1 · 實驗編排。每個函式只回傳純資料（dict / numpy），不畫圖、不落盤。

沿用 A2/A3/B2 的「推論與出圖分離」原則。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

import bo as B
import objectives as O
from acquisition import ACQ_LABEL, optimizer_diagnostics
from gp import GP, KERNELS

N_SEEDS_MAIN = 30      # 計劃書明確要求 30 個隨機種子
N_SEEDS_DIMS = 20      # 維度實驗的成本高（20D 的 GP 超參數優化最慢），降到 20


def _agg(runs_summary: dict) -> dict:
    """只留下繪圖與報告需要的欄位，丟掉逐 run 的完整資料（省下 npz 體積）。"""
    return {k: runs_summary[k] for k in
            ("n_seeds", "mean", "sd", "median", "q25", "q75", "finals",
             "final_mean", "final_sd", "seconds_total")}


# ---------------------------------------------------------------------------
# 1 · 標準測試函數 vs 隨機搜尋（通關標準 1）
# ---------------------------------------------------------------------------


def benchmark_comparison(objective, n_total: int = 30, n_init: int = 5,
                         acqs=("ei", "ucb", "thompson"), kernel: str = "Matern52",
                         n_seeds: int = N_SEEDS_MAIN, kappa: float = 2.0) -> dict:
    """在一個目標函數上比較三種 acquisition 與隨機搜尋。

    所有方法用**完全相同的評估次數預算** `n_total`（BO 的初始設計也計入），
    這是公平比較的前提 —— 真實世界裡初始點也要花實驗次數。
    """
    out = {"objective": objective.name, "dim": objective.dim, "n_total": n_total,
           "n_init": n_init, "kernel": kernel, "kappa": kappa,
           "f_min": objective.f_min, "methods": {}}

    r = B.repeat(lambda seed: B.run_random_search(objective, n_total, seed=seed),
                 n_seeds=n_seeds)
    out["methods"]["random"] = _agg(r)

    for a in acqs:
        r = B.repeat(lambda seed, a=a: B.run_bo(objective, n_total=n_total,
                                                n_init=n_init, acq=a, kappa=kappa,
                                                kernel=kernel, seed=seed),
                     n_seeds=n_seeds)
        out["methods"][f"bo/{a}"] = _agg(r)

    # 相對隨機搜尋的優勢：同一個目標水準下省下幾次評估
    rand_mean = out["methods"]["random"]["mean"]
    targets = {}
    for frac, label in ((0.5, "half"), (0.75, "three_quarters"), (0.9, "ninety")):
        # 目標 = 隨機搜尋用完全部預算所達到的水準的某個比例（以初始水準為基準）
        start = float(rand_mean[0])
        end = float(rand_mean[-1])
        tgt = start - frac * (start - end)
        row = {"target": tgt}
        for m, agg in out["methods"].items():
            row[m] = B.evaluations_to_reach(agg["mean"], tgt)
        targets[label] = row
    out["evals_to_target"] = targets
    return out


# ---------------------------------------------------------------------------
# 2 · κ 敏感度（通關標準 2）
# ---------------------------------------------------------------------------


def kappa_sweep(objective, kappas=(0.0, 0.5, 2.0, 5.0, 20.0), n_total: int = 30,
                n_init: int = 5, kernel: str = "Matern52",
                n_seeds: int = N_SEEDS_MAIN) -> dict:
    """掃 κ。預期 κ=0 卡局部最優、κ=20 退化成近隨機搜尋。

    額外量測兩個能把「卡住」變成數字的指標：
      - **探索半徑**：取樣點與當時最佳點的平均距離。κ=0 應該很小。
      - **不重複點比例**：去重後的點數 / 總點數。κ=0 會反覆取樣同一區域。
    """
    out = {"objective": objective.name, "dim": objective.dim, "kappas": list(kappas),
           "n_total": n_total, "runs": {}}
    rand = B.repeat(lambda seed: B.run_random_search(objective, n_total, seed=seed),
                    n_seeds=n_seeds)
    out["random"] = _agg(rand)

    for k in kappas:
        r = B.repeat(lambda seed, k=k: B.run_bo(objective, n_total=n_total,
                                                n_init=n_init, acq="ucb", kappa=k,
                                                kernel=kernel, seed=seed),
                     n_seeds=n_seeds)
        agg = _agg(r)
        # 探索行為的量化
        radii, uniq = [], []
        for run in r["runs"]:
            X, y = run["X"], run["y"]
            i_best = int(np.argmin(y))
            d = np.linalg.norm(X - X[i_best], axis=1)
            radii.append(float(d[n_init:].mean()))
            # 以 1e-3 為閾值判定「幾乎相同的點」
            keep = 1
            for i in range(1, len(X)):
                if np.min(np.linalg.norm(X[:i] - X[i], axis=1)) > 1e-3:
                    keep += 1
            uniq.append(keep / len(X))
        agg["explore_radius_mean"] = float(np.mean(radii))
        agg["unique_frac_mean"] = float(np.mean(uniq))
        out["runs"][f"kappa={k:g}"] = agg
    return out


def kappa_trajectory_2d(objective, kappas=(0.0, 2.0, 20.0), n_total: int = 30,
                        n_init: int = 5, kernel: str = "Matern52",
                        seed: int = 3, grid: int = 120) -> dict:
    """2D 上的取樣軌跡（供可視化 κ=0 卡在局部最優）。

    只取單一 seed —— 這張圖的目的是展示**機制**，不是統計比較；
    統計比較由 `kappa_sweep` 的 30 seeds 誤差帶負責。
    """
    assert objective.dim == 2, "軌跡圖只用於 2D"
    g = np.linspace(0, 1, grid)
    G1, G2 = np.meshgrid(g, g)
    pts = np.column_stack([G1.ravel(), G2.ravel()])
    Z = objective(pts).reshape(grid, grid)

    out = {"objective": objective.name, "grid_x": g, "Z": Z, "seed": seed,
           "n_init": n_init, "x_min": objective.x_min, "trajectories": {}}
    for k in kappas:
        r = B.run_bo(objective, n_total=n_total, n_init=n_init, acq="ucb",
                     kappa=k, kernel=kernel, seed=seed)
        out["trajectories"][f"kappa={k:g}"] = {
            "X": r["X"], "y": r["y"], "best_value": r["best_value"],
            "best_x": r["best_x"],
            "explore_radius": float(np.linalg.norm(
                r["X"][n_init:] - r["best_x"], axis=1).mean()),
        }
    return out


# ---------------------------------------------------------------------------
# 3 · 維度詛咒（通關標準 3）+ 機制診斷
# ---------------------------------------------------------------------------


def dimension_curse(dims=(2, 5, 10, 20, 50), n_total: int = 50, n_init: int = 10,
                    acq: str = "ei", kernel: str = "Matern52",
                    n_seeds: int = N_SEEDS_DIMS, refit_every: int = 2) -> dict:
    """在不同維度的 Ackley 上比較 BO 與隨機搜尋，找臨界維度。

    固定評估預算 `n_total=50`（不隨維度放大），因為那才對應真實情境：
    「我只有 50 次實驗機會，維度高低不會改變這件事」。
    後果是高維時 `n_init=10 < d`，GP 幾乎沒有資訊 —— 這正是要展示的。

    `refit_every=2` 是成本妥協（20D 的 GP 超參數優化最慢），
    數值記在回傳值裡，README 也標明。
    """
    out = {"dims": list(dims), "n_total": n_total, "n_init": n_init, "acq": acq,
           "kernel": kernel, "refit_every": refit_every, "per_dim": {}}
    for d in dims:
        obj = O.ackley(d)
        rb = B.repeat(lambda seed, o=obj: B.run_bo(o, n_total=n_total, n_init=n_init,
                                                   acq=acq, kernel=kernel, seed=seed,
                                                   refit_every=refit_every),
                      n_seeds=n_seeds)
        rr = B.repeat(lambda seed, o=obj: B.run_random_search(o, n_total, seed=seed),
                      n_seeds=n_seeds)
        # 配對比較：同一個 seed 下 BO 與 random 的最終值差（配對抵消初始設計噪聲）
        paired = rr["finals"] - rb["finals"]        # >0 表示 BO 更好（最小化問題）
        out["per_dim"][str(d)] = {
            "bo": _agg(rb), "random": _agg(rr),
            "paired_gain_mean": float(paired.mean()),
            "paired_gain_sd": float(paired.std(ddof=1)),
            "paired_gain_se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
            "bo_wins_frac": float((paired > 0).mean()),
            "relative_gain_pct": float(100 * paired.mean() /
                                       max(abs(rr["final_mean"]), 1e-12)),
        }
    # 臨界維度：配對優勢的 95% 區間首次涵蓋 0
    crit = None
    for d in dims:
        e = out["per_dim"][str(d)]
        lo = e["paired_gain_mean"] - 1.96 * e["paired_gain_se"]
        if lo <= 0 and crit is None:
            crit = d
    out["critical_dim"] = crit
    return out


def kernel_concentration(dims=(2, 5, 10, 20, 50), n_points: int = 200,
                         kernel: str = "Matern52", lengthscale: float = 0.3,
                         seed: int = 0) -> dict:
    """**維度詛咒的機制**：高維下核值如何集中到 0。

    在 [0,1]^d 均勻抽點，看 pairwise 歐氏距離與核值的分佈。
    高維時距離集中（相對標準差 → 0），且所有距離都變大 →
    核值一起趨近 0 → **每個觀測點都「看不到」其他點**，
    GP 後驗退化成先驗（μ→0、σ→amplitude），acquisition 因此幾乎平坦、
    沒有資訊可用。

    這把「維度詛咒」從一句口號變成可測量的量：
    `frac_kernel_above_0.1` 就是「有效鄰居」的比例。
    """
    k = KERNELS[kernel](log_amp=0.0, log_ls=np.log(lengthscale))
    rows = []
    for d in dims:
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (n_points, d))
        dist = pdist(X)
        K = k(X, X)
        off = K[~np.eye(n_points, dtype=bool)]
        rows.append({
            "dim": int(d),
            "dist_mean": float(dist.mean()), "dist_sd": float(dist.std()),
            "dist_rel_sd": float(dist.std() / dist.mean()),
            "kernel_median": float(np.median(off)),
            "kernel_mean": float(off.mean()),
            "frac_above_0.1": float((off > 0.1).mean()),
            "frac_above_0.01": float((off > 0.01).mean()),
        })
    return {"kernel": kernel, "lengthscale": lengthscale, "n_points": n_points,
            "rows": rows}


def acq_optimizer_shortfall(dims=(2, 5, 10, 20, 50), n_obs: int = 30,
                            acq: str = "ei", kernel: str = "Matern52",
                            n_seeds: int = 8) -> dict:
    """**把「GP 模型不好」與「內層優化解不動」分開。**

    在每個維度上擬合 GP（用 Ackley 的真實觀測），然後用不同的候選點預算
    解同一個 acquisition 最大化問題。若小預算與大預算的結果差很多，
    那失敗有一部分來自**內層優化**，不是 GP 模型 —— 兩者的補救方式
    完全不同（加大內層預算 vs 換核函數 / 換模型）。

    ⚠️ **實測否證了「內層優化失敗」這個假設。** 各維度的 shortfall 幾乎都是 0：
    50 個候選點找到的 acquisition 值與 20000 個相同。真正發生的是更根本的事 ——
    高維下 GP 退化成先驗，**acquisition 變平坦，根本沒有最大值可找**。
    `best_ei` 就是那個證據：它是「最值得試的點能期望改善多少」，
    趨近 0 意味模型認為到處都不值得試，BO 於是退化成隨機搜尋。

    多 seed 平均是必要的：單一 seed 下 GP 超參數優化的隨機性會讓這些量
    出現非單調的跳動（實測 10D 的某個 seed 給出 1e-22 的平坦度，而 20D 給 8e-3）。
    """
    out = []
    for d in dims:
        obj = O.ackley(d)
        per_seed = []
        for s in range(n_seeds):
            X = B.latin_hypercube(n_obs, d, seed=s)
            y = obj(X)
            try:
                gp = GP(kernel=kernel).fit(X, y, optimize=True, n_restarts=3, seed=s)
            except np.linalg.LinAlgError:
                continue
            diag = optimizer_diagnostics(gp, acq, d, float(y.min()), seed=s)
            per_seed.append({**{k: v for k, v in diag.items() if k != "rows"},
                             "gp_lengthscale": gp.hyperparams["lengthscale"],
                             "gp_noise": gp.hyperparams["noise"],
                             "rows": diag["rows"]})
        keys = ("shortfall_at_smallest", "acq_spread", "best_ei", "posterior_sd_mean",
                "posterior_mu_spread", "gp_lengthscale", "gp_noise",
                "y_range_observed")
        agg = {f"{k}_mean": float(np.nanmean([p[k] for p in per_seed])) for k in keys}
        agg.update({f"{k}_sd": float(np.nanstd([p[k] for p in per_seed], ddof=1))
                    for k in ("best_ei", "posterior_sd_mean", "shortfall_at_smallest")})

        # ⚠️ 必須正規化。Ackley 的函數值範圍本身隨維度縮小（餘弦項被平均掉），
        # 所以 best_EI 與 mu_spread 的絕對值下降有一部分只是 y 尺度變小，
        # 不是 GP 退化。除以實際觀測到的 y 範圍後才是「模型解釋力」的乾淨度量。
        #
        # `mu_spread_over_yrange` 是最有說服力的一個：它是「後驗均值的動態範圍
        # 佔觀測值範圍的比例」。趨近 0 表示後驗均值幾乎是常數 ——
        # GP 對這批資料什麼都沒學到，acquisition 因此無方向。
        ratios = {"best_ei_over_yrange": [], "mu_spread_over_yrange": []}
        for p in per_seed:
            yr = max(p["y_range_observed"], 1e-12)
            ratios["best_ei_over_yrange"].append(p["best_ei"] / yr)
            ratios["mu_spread_over_yrange"].append(p["posterior_mu_spread"] / yr)
        for k, v in ratios.items():
            agg[f"{k}_mean"] = float(np.nanmean(v))
            agg[f"{k}_sd"] = float(np.nanstd(v, ddof=1))

        out.append({"dim": int(d), "n_seeds": len(per_seed), **agg,
                    "rows_seed0": per_seed[0]["rows"] if per_seed else []})
    return {"acq": acq, "kernel": kernel, "n_obs": n_obs, "per_dim": out}


# ---------------------------------------------------------------------------
# 4 · 核函數就是先驗（通關標準 4）
# ---------------------------------------------------------------------------


def kernel_comparison(objectives_list, kernels=("RBF", "Matern52"), n_total: int = 30,
                      n_init: int = 5, acq: str = "ei",
                      n_seeds: int = N_SEEDS_MAIN) -> dict:
    """同一套實驗換核函數 —— 展示「核函數就是先驗」（計劃書主題二）。

    同時記錄**學到的 lengthscale**：RBF 假設無限可微，在崎嶇的目標上
    會被迫用很短的 lengthscale 才能擬合，而那讓後驗在觀測點之間
    迅速回到先驗；Matérn 5/2 只要求兩次可微，能用較長的 lengthscale
    描述同樣的資料。這個差異是可以量出來的，不只是理論說法。
    """
    out = {"kernels": list(kernels), "n_total": n_total, "acq": acq, "per_objective": {}}
    for obj in objectives_list:
        entry = {"dim": obj.dim, "f_min": obj.f_min, "kernels": {}}
        for kname in kernels:
            r = B.repeat(lambda seed, k=kname: B.run_bo(obj, n_total=n_total,
                                                        n_init=n_init, acq=acq,
                                                        kernel=k, seed=seed),
                         n_seeds=n_seeds)
            agg = _agg(r)
            ls = [run["hyper_trace"][-1]["lengthscale"] for run in r["runs"]
                  if run["hyper_trace"]]
            noise = [run["hyper_trace"][-1]["noise"] for run in r["runs"]
                     if run["hyper_trace"]]
            agg["final_lengthscale_mean"] = float(np.mean(ls)) if ls else float("nan")
            agg["final_lengthscale_sd"] = float(np.std(ls, ddof=1)) if len(ls) > 1 else 0.0
            agg["final_noise_mean"] = float(np.mean(noise)) if noise else float("nan")
            entry["kernels"][kname] = agg
        # 配對比較（同 seed）
        fa = entry["kernels"][kernels[0]]["finals"]
        fb = entry["kernels"][kernels[1]]["finals"]
        diff = fa - fb
        entry["paired_diff_mean"] = float(diff.mean())
        entry["paired_diff_se"] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        entry["better"] = (kernels[1] if diff.mean() > 0 else kernels[0])
        out["per_objective"][obj.name] = entry
    return out


# ---------------------------------------------------------------------------
# 5 · 真實應用：材料配方（計劃書步驟 5）
# ---------------------------------------------------------------------------


def concrete_application(data_dir: str, n_total: int = 40, n_init: int = 8,
                         acq: str = "ei", kernel: str = "Matern52",
                         n_seeds: int = N_SEEDS_MAIN) -> dict:
    """混凝土配方最優化，並計算「省下多少次實驗」。

    「省下 N 次」的定義：隨機搜尋的**平均**軌跡達到某個強度水準需要
    N_rand 次評估，BO 需要 N_bo 次，省下 N_rand − N_bo。
    水準取隨機搜尋用完全部預算所達到的強度 —— 也就是問
    「BO 要幾次就能達到隨機搜尋花光預算的成果」。
    """
    obj, info = O.concrete_objective(data_dir)
    rb = B.repeat(lambda seed: B.run_bo(obj, n_total=n_total, n_init=n_init, acq=acq,
                                        kernel=kernel, seed=seed), n_seeds=n_seeds)
    rr = B.repeat(lambda seed: B.run_random_search(obj, n_total, seed=seed),
                  n_seeds=n_seeds)

    rand_final = float(rr["mean"][-1])
    n_bo = B.evaluations_to_reach(rb["mean"], rand_final)
    paired = rr["finals"] - rb["finals"]

    # 最佳配方（取所有 BO run 中最好的那個）
    best_run = min(rb["runs"], key=lambda r: r["best_value"])
    lo = np.asarray(info["domain_lo"])
    hi = np.asarray(info["domain_hi"])
    best_recipe = lo + (hi - lo) * best_run["best_x"]

    return {
        "surrogate": info,
        "n_total": n_total, "n_init": n_init, "acq": acq, "kernel": kernel,
        "bo": _agg(rb), "random": _agg(rr),
        "bo_strength_mean": -float(rb["final_mean"]),
        "bo_strength_sd": float(rb["final_sd"]),
        "random_strength_mean": -float(rr["final_mean"]),
        "random_strength_sd": float(rr["final_sd"]),
        "random_final_level_mpa": -rand_final,
        "evals_bo_to_match_random_final": n_bo,
        "evals_saved": None if n_bo is None else n_total - n_bo,
        "saved_pct": None if n_bo is None else 100.0 * (n_total - n_bo) / n_total,
        "paired_gain_mean_mpa": float(paired.mean()),
        "paired_gain_se_mpa": float(paired.std(ddof=1) / np.sqrt(len(paired))),
        "bo_wins_frac": float((paired > 0).mean()),
        "best_recipe": dict(zip(info["features"], best_recipe.round(2).tolist())),
        "best_strength_found": -float(best_run["best_value"]),
    }
