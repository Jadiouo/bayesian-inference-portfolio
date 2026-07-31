"""
C2 · 貝葉斯 SIR 模型（PyMC）。

三個版本，用來逐步證明每個成分都是必要的：

| 版本 | β | day-of-week | 用途 |
|---|---|---|---|
| `constant`      | 固定 | 有 | 對照組：證明時變 β 是必要的 |
| `timevarying`   | random walk | 有 | 主模型 |
| `tv_no_dow`     | random walk | 無 | 對照組：證明週末效應是必要的 |

先驗設計（計劃書主題二）
------------------------
- **γ（恢復率）用文獻的資訊先驗**：COVID 傳染期約 10 天 → γ≈0.1。
  用 `LogNormal(log(0.1), 0.2)`，95% 區間約 γ∈[0.067,0.148]，即傳染期 6.8–15 天。
  這是本模型唯一的強先驗，而且它必須強：γ 與 β 在似然上高度共線
  （R₀=β/γ 決定曲線形狀，個別值只影響時間尺度），資料本身分不開兩者。
- **β 用弱資訊先驗**：`LogNormal(log(0.3), 0.5)`，涵蓋 R₀ 大約 1–8。
- **時變 β 用 random walk 先驗**：`log β_t ~ N(log β_{t−1}, σ_β)`
  > 🔑 這是主題一（序貫更新）與主題二（先驗設計）的組合 ——
  > random walk 先驗就是在說「我相信 β 會變，但不會突然跳」。
  > σ_β 控制「能跳多快」，它自己也有先驗（`HalfNormal(0.1)`），
  > 所以平滑程度是**從資料學來的**，不是我選的。
- **非中心化參數化**：`log β_t = log β₀ + σ_β · cumsum(z_t)`，`z_t ~ N(0,1)`。
  中心化寫法（直接對 β_t 逐點取樣）在 σ_β 小的時候會產生 funnel，
  正是 C1 專案裡實測出 113 個 divergences 的那個幾何。這裡從一開始就避開。

觀測模型
--------
`cases_t ~ NegativeBinomial(μ_t, α)`，`μ_t = ρ · new_infections_t · dow_t`

- **負二項而非 Poisson**：通報數的變異遠大於均值（群聚事件、批次上報）。
  用 Poisson 會讓 α 的角色被 β_t 的 random walk 吸收 —— 模型會用
  「傳染率劇烈波動」去解釋其實是通報噪聲的東西，R_t 因此變得過度抖動。
- **day-of-week 乘數**：`dow_t = exp(δ_{dow(t)} − mean(δ))`，幾何平均鎖定為 1。
  不減去均值的話 δ 會與 ρ 完全共線（兩者都能整體縮放 μ）。

已知的識別問題（誠實記錄，不假裝解決）
--------------------------------------
**ρ（通報率）與 I₀ 都只影響「觀測到的量級」**，單靠病例數序列無法分開：
把 ρ 減半、I₀ 加倍，前期的 μ_t 幾乎不變。本模型靠 ρ 的資訊先驗
（`Beta(2,8)`，均值 0.2，對應第一波血清學調查估計的通報率量級）
把這個尺度釘住，讓 I₀ 由資料決定。這意味著**I₀ 的後驗在很大程度上
反映的是 ρ 的先驗**，不是資料的證據 —— README 的限制章節有記錄。
"""
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from sir import simulate_pytensor

VERSIONS = ("constant", "timevarying", "tv_no_dow")
VERSION_LABEL = {
    "constant": "Constant β",
    "timevarying": "Time-varying β (random walk) + day-of-week",
    "tv_no_dow": "Time-varying β, no day-of-week",
}


def build(ep, version: str = "timevarying", gamma_mu: float = 0.1,
          gamma_sigma: float = 0.2, beta0_mu: float = 0.3, beta0_sigma: float = 0.5,
          sigma_beta_scale: float = 0.1, I0_mu: float = 500.0, I0_sigma: float = 1.0,
          rho_a: float = 2.0, rho_b: float = 8.0, dow_sigma: float = 0.2) -> pm.Model:
    """建一個貝葉斯 SIR 模型。`ep` 是 data.Epidemic。"""
    if version not in VERSIONS:
        raise ValueError(f"version 必須是 {VERSIONS}")

    n_days = ep.n_days
    N = float(ep.population)
    cases = ep.cases.astype(float)
    dow = ep.dow.astype(int)
    use_dow = version != "tv_no_dow"
    time_varying = version != "constant"

    coords = {"day": np.arange(n_days), "weekday": list(range(7))}
    with pm.Model(coords=coords) as model:
        # ── 流行病學參數 ────────────────────────────────────────────────
        gamma = pm.LogNormal("gamma", mu=np.log(gamma_mu), sigma=gamma_sigma)
        beta0 = pm.LogNormal("beta0", mu=np.log(beta0_mu), sigma=beta0_sigma)
        I0 = pm.LogNormal("I0", mu=np.log(I0_mu), sigma=I0_sigma)
        rho = pm.Beta("rho", alpha=rho_a, beta=rho_b)

        if time_varying:
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=sigma_beta_scale)
            # 非中心化 random walk：z 標準常態、cumsum 後乘 σ_β
            z = pm.Normal("z", 0.0, 1.0, dims="day")
            log_beta = pm.Deterministic(
                "log_beta", pt.log(beta0) + sigma_beta * pt.cumsum(z), dims="day")
            beta_t = pt.exp(log_beta)
        else:
            beta_t = pt.repeat(beta0, n_days)
            pm.Deterministic("log_beta", pt.log(beta_t), dims="day")

        # ── SIR 前向模擬 ────────────────────────────────────────────────
        S0 = N - I0
        S, I, R, new_inf = simulate_pytensor(beta_t, gamma, S0, I0, N, n_days)
        pm.Deterministic("S", S, dims="day")
        pm.Deterministic("I", I, dims="day")
        pm.Deterministic("new_infections", new_inf, dims="day")

        # 有效再生數 R_t = (β_t/γ)(S_t/N)
        pm.Deterministic("R_t", (beta_t / gamma) * (S / N), dims="day")

        # ── 觀測模型 ────────────────────────────────────────────────────
        if use_dow:
            delta = pm.Normal("delta_dow", 0.0, dow_sigma, dims="weekday")
            # 減去均值 → 幾何平均 1，避免與 rho 共線
            dow_mult = pm.Deterministic("dow_effect",
                                        pt.exp(delta - pt.mean(delta)), dims="weekday")
            mult = dow_mult[dow]
        else:
            mult = 1.0

        mu = pm.Deterministic("mu", rho * new_inf * mult + 1e-6, dims="day")
        alpha = pm.Exponential("alpha", lam=1.0 / 10.0)
        pm.NegativeBinomial("cases", mu=mu, alpha=alpha, observed=cases, dims="day")

    return model


def sample(model: pm.Model, draws: int = 1000, tune: int = 1500, chains: int = 4,
           seed: int = 20260730, target_accept: float = 0.9, progressbar: bool = False):
    """NUTS 取樣。

    `init="jitter+adapt_diag"` 在此**保留** PyMC 預設，與 A2 相反 ——
    那裡的問題來自 Weibull 的 `(t/λ)^k` 在 jitter 後溢出；
    這裡的 SIR 遞迴對初始點溫和得多，且 tune 較長。
    收斂仍由 `check_convergence` 守門，不靠假設。
    """
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=target_accept, progressbar=progressbar,
                          idata_kwargs={"log_likelihood": True})
    return idata


def check_convergence(idata, var_names=("gamma", "beta0", "I0", "rho", "alpha"),
                      max_divergent_frac: float = 0.01, max_rhat: float = 1.01,
                      min_ess: float = 400, label: str = "",
                      raise_on_fail: bool = True) -> dict:
    """收斂守門：divergences / r_hat / ESS 三道。

    只檢查純量參數（外加 log_beta 的最差值）。時變模型有 122 個 z，
    逐個檢查會讓單一邊緣參數的雜訊蓋掉真正的問題；
    但 log_beta 是實際進入結論的量，所以它必須過關。
    """
    import arviz as az

    n_div = int(idata.sample_stats.diverging.sum())
    n_tot = int(idata.sample_stats.diverging.size)
    present = [v for v in var_names if v in idata.posterior]
    sm = az.summary(idata, var_names=present)
    lb = az.summary(idata, var_names=["log_beta"]) if "log_beta" in idata.posterior else None

    rep = {"label": label, "divergent": n_div, "divergent_frac": n_div / n_tot,
           "max_rhat_scalars": float(sm["r_hat"].max()),
           "min_ess_scalars": float(sm["ess_bulk"].min())}
    if lb is not None:
        rep["max_rhat_log_beta"] = float(lb["r_hat"].max())
        rep["min_ess_log_beta"] = float(lb["ess_bulk"].min())

    fails = []
    if rep["divergent_frac"] > max_divergent_frac:
        fails.append(f"divergences {n_div}/{n_tot} ({100 * rep['divergent_frac']:.1f}%)")
    if rep["max_rhat_scalars"] > max_rhat:
        fails.append(f"scalar max r_hat {rep['max_rhat_scalars']:.3f}")
    if rep["min_ess_scalars"] < min_ess:
        fails.append(f"scalar min ESS {rep['min_ess_scalars']:.0f}")
    if lb is not None and rep["max_rhat_log_beta"] > max_rhat + 0.01:
        fails.append(f"log_beta max r_hat {rep['max_rhat_log_beta']:.3f}")
    rep["passed"] = not fails
    rep["failures"] = fails
    if fails and raise_on_fail:
        raise RuntimeError(f"convergence check failed [{label}]: " + "; ".join(fails))
    return rep


def posterior_summary(idata, ep) -> dict:
    """把主要純量參數與衍生量整理成可報告的形式。"""
    import arviz as az

    out = {}
    for v in ("gamma", "beta0", "I0", "rho", "alpha", "sigma_beta"):
        if v not in idata.posterior:
            continue
        d = idata.posterior[v].to_numpy().ravel()
        out[v] = {"mean": float(d.mean()), "median": float(np.median(d)),
                  "lo95": float(np.percentile(d, 2.5)),
                  "hi95": float(np.percentile(d, 97.5))}
    # 傳染期 = 1/γ，比 γ 本身好讀
    g = idata.posterior["gamma"].to_numpy().ravel()
    out["infectious_period_days"] = {"median": float(np.median(1.0 / g)),
                                     "lo95": float(np.percentile(1.0 / g, 2.5)),
                                     "hi95": float(np.percentile(1.0 / g, 97.5))}
    if "dow_effect" in idata.posterior:
        de = idata.posterior["dow_effect"].to_numpy().reshape(-1, 7)
        out["dow_effect"] = {"median": np.median(de, axis=0).tolist(),
                            "lo95": np.percentile(de, 2.5, axis=0).tolist(),
                            "hi95": np.percentile(de, 97.5, axis=0).tolist()}
    # 累計感染佔人口比例（判斷群體免疫是否參與了 R_t 下降）
    S = idata.posterior["S"].to_numpy().reshape(-1, ep.n_days)
    attack = (ep.population - S[:, -1]) / ep.population
    out["final_attack_rate"] = {"median": float(np.median(attack)),
                               "lo95": float(np.percentile(attack, 2.5)),
                               "hi95": float(np.percentile(attack, 97.5))}
    return out
