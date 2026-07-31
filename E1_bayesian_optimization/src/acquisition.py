"""
E1 · Acquisition functions 與它們的最大化（計劃書主題七第二卡）。

三種策略，對「探索 vs 利用」的處理方式完全不同：

| 名稱 | 公式 | 探索的來源 |
|---|---|---|
| **UCB** | $\\mu - \\kappa\\sigma$（最小化問題） | 顯式的 $\\kappa$ 旋鈕 |
| **EI** | $(f^+ - \\mu)\\Phi(Z) + \\sigma\\phi(Z)$ | 隱式：對改善量取期望 |
| **Thompson** | 從後驗抽一條函數，取它的最小值位置 | 隱式：後驗抽樣本身的隨機性 |

注意本模組全部處理**最小化**問題（objectives.py 的約定），
所以 UCB 是 $\\mu - \\kappa\\sigma$（往低且不確定的地方走），
EI 衡量的是「比目前最佳值 $f^+$ **更低**多少」的期望。
文獻上多寫成最大化形式，符號要對得起來。

> ⚠️ **acquisition 的值不等於 BO 的效能。** BO 每一步都要解一個內層問題
> 「acquisition 的最大值在哪」，而那本身是個非凸全域優化。低維時多起點
> L-BFGS 幾乎總能找到；高維時它自己就失敗了。`optimizer_diagnostics`
> 專門量測這件事，好把「GP 模型不好」與「內層優化解不動」分開 ——
> 這是維度詛咒實驗的必要配套（見 experiments.py）。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Acquisition 值
# ---------------------------------------------------------------------------


def ucb(mu, sd, kappa: float = 2.0, **kw):
    """Lower Confidence Bound（最小化版）：越小越值得去。

    κ=0 → 純利用（只看 μ，完全不管不確定性）。
    κ→∞ → 純探索（只看 σ，退化成「去最沒去過的地方」）。
    """
    return mu - kappa * sd


def ei(mu, sd, f_best: float, xi: float = 0.0, **kw):
    """Expected Improvement（最小化版），回傳**負的** EI 以便統一最小化。

    EI = E[max(f⁺ − f(x), 0)] = (f⁺ − μ)Φ(Z) + σφ(Z)，Z = (f⁺ − μ)/σ

    σ→0 時整式趨於 0，所以 EI 天然地不會重複取樣已知點 ——
    這是它比 UCB 少一個旋鈕的原因。`xi` 是可選的最小改善門檻。
    """
    sd = np.maximum(sd, 1e-12)
    imp = f_best - mu - xi
    Z = imp / sd
    val = imp * norm.cdf(Z) + sd * norm.pdf(Z)
    return -val


def thompson_factory(gp, seed: int = 0):
    """Thompson sampling：從後驗抽**一條**函數，整輪都用它。

    關鍵在「整輪都用同一條」：每次評估 acquisition 時重抽會讓目標函數
    在優化過程中不斷變動，L-BFGS 追不到任何東西。做法是先在一組候選點上
    抽出聯合樣本，再在其上取最小 —— 也因此 Thompson 在本實作裡是
    純候選點式的，不做連續優化。
    """
    def acq_on_candidates(Xc):
        s = gp.sample_posterior(Xc, n_samples=1, seed=seed)[0]
        return s
    return acq_on_candidates


ACQUISITIONS = {"ucb": ucb, "ei": ei, "thompson": None}
ACQ_LABEL = {"ucb": "UCB (μ−κσ)", "ei": "Expected Improvement",
             "thompson": "Thompson sampling", "random": "Random search"}


# ---------------------------------------------------------------------------
# Acquisition 最大化
# ---------------------------------------------------------------------------


def _acq_values(gp, X, kind: str, f_best: float, kappa: float):
    mu, sd = gp.predict(X)
    if kind == "ucb":
        return ucb(mu, sd, kappa=kappa)
    if kind == "ei":
        return ei(mu, sd, f_best=f_best)
    raise ValueError(kind)


def optimize_acquisition(gp, kind: str, dim: int, f_best: float, kappa: float = 2.0,
                         n_candidates: int = 2000, n_restarts: int = 5,
                         seed: int = 0, use_gradient_free: bool = True,
                         ts_max_points: int = 512) -> dict:
    """找 acquisition 的最小值位置（= 下一個要評估的點）。

    兩階段：
      1. Sobol 式的隨機候選點掃描（覆蓋定義域，找出好的起點）
      2. 從最好的幾個候選點出發跑 L-BFGS-B 做局部精修

    第二階段用數值梯度。解析梯度需要對 μ(x)、σ(x) 再對 x 微分，
    那是另一套鏈式法則；本專案的重點在 GP 與 BO 的機制，
    內層優化用有限差分已足夠（`optimizer_diagnostics` 會量測它夠不夠）。

    回傳含診斷：候選點階段的最佳值、精修後的最佳值、改善量。
    """
    rng = np.random.default_rng(seed)
    Xc = rng.uniform(0.0, 1.0, size=(n_candidates, dim))

    if kind == "thompson":
        # ⚠️ Thompson 需要**聯合**後驗樣本，而聯合抽樣要對 m×m 的共變異數做
        # Cholesky → O(m³)。用 m=2000 時實測一次 BO run 要 17.7 秒，
        # 而 UCB/EI 只要 0.4 秒（慢 45 倍），整個專案會被這一項主導。
        #
        # 折衷：在候選點裡隨機取 `ts_max_points` 個做聯合抽樣。
        # 隨機子集仍均勻覆蓋定義域，只是降低了搜尋解析度 ——
        # 這是計算預算的選擇，在 README 的限制章節記錄。
        # （正解是 random Fourier features，能以 O(n·D) 抽出連續樣本函數，
        #   但那是另一套近似方法，超出本專案「手刻 GP」的範圍。）
        if len(Xc) > ts_max_points:
            sub = rng.choice(len(Xc), ts_max_points, replace=False)
            Xc = Xc[sub]
        s = gp.sample_posterior(Xc, n_samples=1, seed=seed)[0]
        i = int(np.argmin(s))
        return {"x": Xc[i], "acq_candidate_best": float(s[i]),
                "acq_refined_best": float(s[i]), "refine_gain": 0.0,
                "n_candidates": len(Xc), "n_restarts": 0,
                "ts_subsampled": bool(len(Xc) == ts_max_points)}

    vals = _acq_values(gp, Xc, kind, f_best, kappa)
    order = np.argsort(vals)
    cand_best = float(vals[order[0]])

    def obj(x):
        return float(_acq_values(gp, x.reshape(1, -1), kind, f_best, kappa)[0])

    best_x, best_v = Xc[order[0]].copy(), cand_best
    if not use_gradient_free:
        return {"x": best_x, "acq_candidate_best": cand_best,
                "acq_refined_best": best_v, "refine_gain": 0.0,
                "n_candidates": n_candidates, "n_restarts": 0}

    for j in range(min(n_restarts, len(order))):
        x0 = Xc[order[j]]
        try:
            r = minimize(obj, x0, method="L-BFGS-B",
                         bounds=[(0.0, 1.0)] * dim,
                         options={"maxiter": 100})
        except Exception:
            continue
        if np.isfinite(r.fun) and r.fun < best_v:
            best_v, best_x = float(r.fun), np.clip(r.x, 0.0, 1.0)

    return {"x": best_x, "acq_candidate_best": cand_best,
            "acq_refined_best": best_v, "refine_gain": cand_best - best_v,
            "n_candidates": n_candidates, "n_restarts": n_restarts}


def optimizer_diagnostics(gp, kind: str, dim: int, f_best: float, kappa: float = 2.0,
                          budgets=(50, 200, 1000, 5000, 20000), n_restarts: int = 5,
                          seed: int = 0) -> dict:
    """**內層優化夠不夠好？** 用不同的候選點預算解同一個 acquisition 問題。

    如果 20000 個候選點找到的 acquisition 值明顯優於 200 個，
    那表示在這個維度下「找到 acquisition 的最大值」本身就沒解好 ——
    BO 的表現差可能不是 GP 模型的錯，而是內層優化的錯。
    這兩件事的補救方式完全不同（換核函數 vs 加大內層預算），
    所以必須分開量測。這是維度詛咒實驗的必要配套。
    """
    out = []
    for b in budgets:
        r = optimize_acquisition(gp, kind, dim, f_best, kappa=kappa,
                                 n_candidates=b, n_restarts=n_restarts, seed=seed)
        out.append({"budget": int(b), "candidate_best": r["acq_candidate_best"],
                    "refined_best": r["acq_refined_best"],
                    "refine_gain": r["refine_gain"]})
    ref = out[-1]["refined_best"]
    for row in out:
        # 相對於最大預算的落後量（>0 表示這個預算沒解到最好）
        row["shortfall_vs_largest"] = float(row["refined_best"] - ref)

    # ── acquisition 的「動態範圍」——比 shortfall 更能診斷高維失敗 ──────
    #
    # 實測發現：高維下 shortfall 幾乎是 0，也就是**內層優化沒有失敗**。
    # 原本的假設（高維下找不到 acquisition 的最大值）被否證了。
    # 真正發生的是更根本的事：GP 退化成先驗後 acquisition **變平坦**，
    # 根本沒有最大值可找 —— 任何預算都會找到同一個平坦值。
    #
    # 所以這裡直接量 acquisition 在候選點上的散佈：
    #   spread → 0 意味「所有候選點看起來一樣好」，選哪個都無所謂，
    #   BO 於是退化成隨機搜尋。這才是高維失敗的機制。
    rng = np.random.default_rng(seed + 1)
    Xc = rng.uniform(0.0, 1.0, size=(4000, dim))
    if kind == "thompson":
        vals = gp.sample_posterior(Xc, n_samples=1, seed=seed)[0]
    else:
        vals = _acq_values(gp, Xc, kind, f_best, kappa)
    mu, sd = gp.predict(Xc)

    # `best_ei` 是有物理意義的量：EI 的最大值 = 「最值得試的那一點，
    # 期望能比目前最佳改善多少」。它趨近 0 就表示模型認為**到處都不值得試**，
    # BO 於是沒有方向可循。
    # （刻意不做「除以 median」的相對化：EI 在多數候選點上本來就趨近 0，
    #   拿它當分母會產生 1e10 量級的無意義數字。）
    best_acq = float(-vals.min()) if kind == "ei" else float(vals.min())
    return {"kind": kind, "dim": dim, "rows": out,
            "shortfall_at_smallest": out[0]["shortfall_vs_largest"],
            "best_value_at_largest_budget": float(ref),
            "acq_spread": float(vals.max() - vals.min()),
            "acq_sd": float(vals.std()),
            "best_ei": best_acq if kind == "ei" else float("nan"),
            "posterior_sd_mean": float(sd.mean()),
            "posterior_mu_spread": float(mu.max() - mu.min()),
            "y_range_observed": float(np.ptp(gp.fit_.y_std) * gp.fit_.y_scale)}
