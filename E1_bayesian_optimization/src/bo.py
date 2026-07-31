"""
E1 · 貝葉斯最優化主迴圈，以及它的對照組（隨機搜尋）。

一次 BO run 的結構：

    初始設計（Latin hypercube，n_init 點）
      → 迴圈：擬合 GP → 最大化 acquisition → 評估目標函數 → 更新資料
      → 記錄「到目前為止的最佳值」軌跡

**對照組是隨機搜尋，而且它是個強對手。** Bergstra & Bengio (2012) 的經典
結果就是隨機搜尋在高維超參數搜尋上打敗網格搜尋；本專案的維度詛咒實驗
會量出它在多少維度開始追上 BO。

超參數重擬合的頻率
------------------
每一步都重新優化 GP 超參數（多起點 L-BFGS）是最正確的做法，但成本高。
本模組用 `refit_every`：預設每一步都重擬合（n≤50，成本可接受），
但保留參數讓維度詛咒實驗在大預算下降頻。**這個選擇會影響結果**，
所以 run 的回傳值裡記錄了實際用的頻率。
"""
from __future__ import annotations

import time

import numpy as np

from acquisition import optimize_acquisition
from gp import GP


def latin_hypercube(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Latin hypercube 初始設計 —— 比純隨機更均勻地覆蓋每一維的邊際分佈。

    BO 的初始點品質對結果影響很大：若初始點聚在一角，GP 對其餘區域
    毫無資訊，前幾步會浪費在補基本覆蓋上。LHS 用同樣的預算換到
    每一維都被分層覆蓋。
    """
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n + 1)
    out = np.empty((n, dim))
    for d in range(dim):
        pts = rng.uniform(cut[:n], cut[1:])
        out[:, d] = rng.permutation(pts)
    return out


def running_best(y: np.ndarray) -> np.ndarray:
    """「到第 t 次評估為止的最佳（最小）值」軌跡。"""
    return np.minimum.accumulate(np.asarray(y, dtype=float))


def run_random_search(objective, n_total: int, seed: int = 0) -> dict:
    """隨機搜尋對照組。用與 BO 相同的評估次數預算。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_total, objective.dim))
    t0 = time.time()
    y = objective(X)
    return {"method": "random", "X": X, "y": y, "best_trace": running_best(y),
            "best_value": float(y.min()), "best_x": X[int(np.argmin(y))],
            "seconds": round(time.time() - t0, 2), "objective": objective.name,
            "dim": objective.dim, "seed": seed, "n_total": n_total}


def run_bo(objective, n_total: int = 30, n_init: int = 5, acq: str = "ei",
           kappa: float = 2.0, kernel: str = "Matern52", seed: int = 0,
           n_candidates: int = 2000, n_restarts_acq: int = 5,
           refit_every: int = 1, gp_restarts: int = 3,
           collect_diagnostics: bool = False) -> dict:
    """跑一次貝葉斯最優化。

    `n_total` 含初始設計，所以與隨機搜尋的評估次數預算完全對齊 ——
    這是公平比較的前提（BO 的初始點也是要花實驗次數的）。
    """
    rng = np.random.default_rng(seed)
    X = latin_hypercube(n_init, objective.dim, seed=seed)
    y = objective(X)

    gp = GP(kernel=kernel)
    hyper_trace, acq_trace, diag = [], [], []
    t0 = time.time()
    n_fit_fail = 0

    for it in range(n_total - n_init):
        need_refit = (it % refit_every == 0)
        try:
            gp.fit(X, y, optimize=need_refit, n_restarts=gp_restarts, seed=seed + it)
        except np.linalg.LinAlgError:
            n_fit_fail += 1
            # GP 擬合失敗（K 病態）→ 退回隨機取樣一步，不讓整個 run 死掉
            x_next = rng.uniform(0.0, 1.0, objective.dim)
            X = np.vstack([X, x_next])
            y = np.append(y, objective(x_next.reshape(1, -1))[0])
            continue

        f_best = float(y.min())
        res = optimize_acquisition(gp, acq, objective.dim, f_best, kappa=kappa,
                                   n_candidates=n_candidates,
                                   n_restarts=n_restarts_acq, seed=seed * 1000 + it)
        x_next = res["x"]
        y_next = objective(x_next.reshape(1, -1))[0]

        hyper_trace.append({"iter": it, **gp.hyperparams})
        acq_trace.append({"iter": it, "candidate_best": res["acq_candidate_best"],
                          "refined_best": res["acq_refined_best"],
                          "refine_gain": res["refine_gain"]})
        if collect_diagnostics and it == (n_total - n_init) // 2:
            from acquisition import optimizer_diagnostics
            diag.append(optimizer_diagnostics(gp, acq, objective.dim, f_best,
                                              kappa=kappa, seed=seed))

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    return {"method": f"bo/{acq}", "acq": acq, "kappa": kappa, "kernel": kernel,
            "X": X, "y": y, "best_trace": running_best(y),
            "best_value": float(y.min()), "best_x": X[int(np.argmin(y))],
            "hyper_trace": hyper_trace, "acq_trace": acq_trace,
            "optimizer_diagnostics": diag, "n_gp_fit_failures": n_fit_fail,
            "n_cholesky_retries": gp.n_cholesky_retries,
            "seconds": round(time.time() - t0, 2), "objective": objective.name,
            "dim": objective.dim, "seed": seed, "n_total": n_total,
            "n_init": n_init, "refit_every": refit_every}


def repeat(runner, n_seeds: int = 30, **kw) -> dict:
    """跑多個隨機種子並聚合成平均 ± 標準差的軌跡。

    > ⚠️ 只跑一次的比較沒有意義。**多次重複 + 誤差帶**是這個專案的專業度所在
    > （計劃書 §6 專案 E1 步驟 3）。BO 的單次軌跡受初始設計影響極大，
    > 兩個方法的單次曲線交叉多次是常態。

    回傳每個評估次數 t 上的 mean/sd/median，以及各 run 的最終值分佈。
    """
    runs = []
    for s in range(n_seeds):
        runs.append(runner(seed=s, **kw))
    traces = np.vstack([r["best_trace"] for r in runs])
    finals = np.array([r["best_value"] for r in runs])
    return {
        "n_seeds": n_seeds,
        "mean": traces.mean(axis=0), "sd": traces.std(axis=0, ddof=1),
        "median": np.median(traces, axis=0),
        "q25": np.percentile(traces, 25, axis=0),
        "q75": np.percentile(traces, 75, axis=0),
        "traces": traces, "finals": finals,
        "final_mean": float(finals.mean()), "final_sd": float(finals.std(ddof=1)),
        "seconds_total": round(sum(r["seconds"] for r in runs), 1),
        "runs": runs,
    }


def evaluations_to_reach(trace_mean: np.ndarray, target: float) -> int | None:
    """平均軌跡首次達到 `target` 所需的評估次數（1-indexed）。None = 從未達到。

    這是「省下多少次實驗」的計算基礎：同一個目標水準下，
    隨機搜尋要 N_rand 次、BO 要 N_bo 次，省下 N_rand − N_bo 次。
    """
    idx = np.where(np.asarray(trace_mean) <= target)[0]
    return None if len(idx) == 0 else int(idx[0]) + 1
