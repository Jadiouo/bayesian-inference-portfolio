"""
A2 · 貝葉斯參數化生存模型（PyMC）。

參數化：AFT（accelerated failure time）
-----------------------------------------
        log λ_i = β₀ + x_i·β

λ_i 是個體 i 的時間尺度參數，係數 β_j 的語意是「x_j 增加一個標準差，
存活時間乘以 exp(β_j)」。這比 PH（比例風險）的語意更直觀：
病人問的是「我還能活多久」，不是「我的瞬時風險比別人高幾倍」。

三個基準分佈（供 WAIC/LOO 比較）：
    Weibull      S(t)=exp(-(t/λ)^k)   —— 風險率單調（k>1 遞增、k<1 遞減）
    Log-Normal   log T ~ N(log λ, σ)  —— 風險率先升後降（有峰值）
    Exponential  S(t)=exp(-t/λ)       —— 風險率恆定，Weibull 的 k=1 特例

Weibull 特別重要：它**同時**是 AFT 也是 PH 模型（唯一同時滿足兩者的
連續分佈族之一），所以可以把 AFT 係數換算成 hazard ratio，直接跟
頻率派 Cox 的輸出並排比較（見 `aft_to_ph`）。

先驗
----
標準化特徵下，β ~ Normal(0, 0.5)：一個標準差的特徵變化造成 exp(±0.5)=
0.6~1.65 倍的存活時間，這涵蓋了臨床上合理的效應大小而排除荒謬值
（β=3 意味 20 倍存活時間）。β₀ ~ Normal(1.5, 1.0)：λ 的先驗中位數約
4.5 年，涵蓋 0.6~12 年，對上 GBSG2 的追蹤範圍。
先驗敏感度（0.25 / 0.5 / 1.0）在 run_all.py 裡實測。
"""
from __future__ import annotations

import numpy as np
import pymc as pm

from censoring import censoring_bounds

KINDS = ("weibull", "lognormal", "exponential")

# 刪失處理策略
CORRECT = "correct"  # 事件用 f(t)、刪失用 S(t) —— 正確做法
AS_EVENT = "as_event"  # 把刪失當成事件 —— 錯誤做法 B
DROP = "drop"  # 丟掉刪失樣本 —— 錯誤做法 A（在 data.drop_censored 先過濾）


def build(
    s,
    kind: str = "weibull",
    censoring: str = CORRECT,
    beta_sd: float = 0.5,
    intercept_mu: float = 1.5,
    intercept_sd: float = 1.0,
) -> pm.Model:
    """建一個 AFT 生存模型。

    `censoring`:
      CORRECT  → 用 pm.Censored，刪失筆貢獻 log S(t)
      AS_EVENT → 不做刪失處理，每筆都貢獻 log f(t)
      DROP     → 同 AS_EVENT，但呼叫端應先用 data.drop_censored 過濾樣本
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind}")

    X, t, event = s.X, s.t, s.event
    n, p = X.shape

    with pm.Model(coords={"feature": s.features, "obs": np.arange(n)}) as model:
        Xd = pm.Data("X", X, dims=("obs", "feature"))
        beta0 = pm.Normal("beta0", mu=intercept_mu, sigma=intercept_sd)
        beta = pm.Normal("beta", mu=0.0, sigma=beta_sd, dims="feature")
        lam = pm.Deterministic("lam", pm.math.exp(beta0 + pm.math.dot(Xd, beta)), dims="obs")

        if kind == "weibull":
            k = pm.LogNormal("k", mu=0.0, sigma=0.5)
            latent = pm.Weibull.dist(alpha=k, beta=lam)
        elif kind == "lognormal":
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            latent = pm.LogNormal.dist(mu=pm.math.log(lam), sigma=sigma)
        else:
            latent = pm.Exponential.dist(scale=lam)

        if censoring == CORRECT:
            ub = censoring_bounds(t, event)
            pm.Censored("obs_t", latent, lower=None, upper=ub, observed=t, dims="obs")
        else:
            # 沒有刪失處理：每筆都當成觀察到的事件時間
            pm.Censored("obs_t", latent, lower=None, upper=np.inf, observed=t, dims="obs")

    return model


def sample(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    seed: int = 20260730,
    progressbar: bool = False,
    init: str = "adapt_diag",
    target_accept: float = 0.95,
):
    """NUTS 取樣，並算好逐點 log-likelihood（WAIC/LOO 需要）。

    ⚠️ `init="adapt_diag"`（而非 PyMC 預設的 `"jitter+adapt_diag"`）是踩坑修正，
    不是風格選擇。預設的 jitter 會把初始點隨機推開，在高刪失率的資料上
    偶發地把某條鏈丟進 `(t/λ)^k` 數值溢出的區域，該鏈就**整條卡死**：
    800 draws 全部 divergent、後驗 sd 膨脹到約等於先驗 sd（0.5），
    而另一條鏈完全正常。因為只影響部分 random seed，單跑一次很容易沒發現，
    但在多 seed 的實驗裡會產生看似「刪失越多越不確定」的假趨勢。
    改用不加 jitter 的初始化 + target_accept=0.95 後，原本失敗的 seed 全部歸零。

    務必搭配 `check_convergence` 使用 —— 靜默的取樣失敗比明顯的錯誤更危險。
    """
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            target_accept=target_accept,
            init=init,
            progressbar=progressbar,
            idata_kwargs={"log_likelihood": True},
        )
    return idata


def check_convergence(idata, max_divergent_frac: float = 0.005, max_rhat: float = 1.01,
                      min_ess: float = 400, label: str = "", raise_on_fail: bool = True) -> dict:
    """收斂守門：divergences、r_hat、ESS 三道檢查。

    存在的理由見 `sample` 的警告：一條卡死的鏈會靜默地污染整個實驗結果。
    任何進入 README 數字的推論都必須先過這一關。
    """
    import arviz as az

    n_div = int(idata.sample_stats.diverging.sum())
    n_total = int(idata.sample_stats.diverging.size)
    var_names = [v for v in ("beta0", "beta", "k", "sigma") if v in idata.posterior]
    sm = az.summary(idata, var_names=var_names)
    rep = {
        "label": label,
        "divergent": n_div,
        "divergent_frac": n_div / n_total,
        "max_rhat": float(sm["r_hat"].max()),
        "min_ess_bulk": float(sm["ess_bulk"].min()),
    }
    fails = []
    if rep["divergent_frac"] > max_divergent_frac:
        fails.append(f"divergences {n_div}/{n_total} ({100 * rep['divergent_frac']:.1f}%)")
    if rep["max_rhat"] > max_rhat:
        fails.append(f"max r_hat {rep['max_rhat']:.3f}")
    if rep["min_ess_bulk"] < min_ess:
        fails.append(f"min ESS {rep['min_ess_bulk']:.0f}")
    rep["passed"] = not fails
    rep["failures"] = fails
    if fails and raise_on_fail:
        raise RuntimeError(f"convergence check failed [{label}]: " + "; ".join(fails))
    return rep


# ---------------------------------------------------------------------------
# 後驗量的換算與預測
# ---------------------------------------------------------------------------


def posterior_flat(idata, name: str) -> np.ndarray:
    """把 (chain, draw, ...) 攤平成 (sample, ...)。"""
    da = idata.posterior[name]
    return da.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def aft_to_ph(idata, kind: str = "weibull") -> np.ndarray | None:
    """Weibull AFT 係數 → PH 的 log hazard ratio：β_PH = −k · β_AFT。

    推導：h(t|x) = k t^(k−1) / λ^k，代入 λ=exp(β₀+xβ_AFT) 得
          h(t|x) = k t^(k−1) exp(−k(β₀+xβ_AFT))
    x 的效應只出現在乘性因子 exp(−k·xβ_AFT)，與 t 無關 → 這正是 PH 結構，
    且 log HR = −k·β_AFT。只有 Weibull（含 Exponential，k=1）能這樣換算；
    Log-Normal 不是 PH 模型，回傳 None。
    """
    if kind == "weibull":
        beta = posterior_flat(idata, "beta")  # (S, p)
        k = posterior_flat(idata, "k")  # (S,)
        return -k[:, None] * beta
    if kind == "exponential":
        return -posterior_flat(idata, "beta")
    return None


def survival_curves(idata, x_new: np.ndarray, t_grid: np.ndarray, kind: str = "weibull",
                    max_draws: int | None = 2000, seed: int = 0) -> np.ndarray:
    """個體化存活曲線的後驗樣本。

    回傳 (n_draws, len(t_grid))：每個後驗抽樣給出一整條 S(t|x_new)。
    這是貝葉斯的獨門好處 —— 不是一條曲線，是一整族曲線，
    取分位數就得到不確定性帶。Cox 的 baseline hazard 是非參數點估計，
    `predict_survival_function` 只會給你一條線。
    """
    beta0 = posterior_flat(idata, "beta0")
    beta = posterior_flat(idata, "beta")
    S_total = len(beta0)
    if max_draws is not None and S_total > max_draws:
        idx = np.random.default_rng(seed).choice(S_total, max_draws, replace=False)
    else:
        idx = np.arange(S_total)

    lam = np.exp(beta0[idx] + beta[idx] @ np.asarray(x_new, dtype=float))  # (S,)
    tg = np.asarray(t_grid, dtype=float)[None, :]

    if kind == "weibull":
        k = posterior_flat(idata, "k")[idx][:, None]
        return np.exp(-((tg / lam[:, None]) ** k))
    if kind == "exponential":
        return np.exp(-tg / lam[:, None])
    if kind == "lognormal":
        from scipy import stats

        sigma = posterior_flat(idata, "sigma")[idx][:, None]
        return stats.lognorm.sf(tg, s=sigma, scale=lam[:, None])
    raise ValueError(kind)


def median_survival(idata, x_new: np.ndarray, kind: str = "weibull") -> np.ndarray:
    """中位存活時間的後驗樣本（年）。Weibull: λ(ln2)^(1/k)。"""
    beta0 = posterior_flat(idata, "beta0")
    beta = posterior_flat(idata, "beta")
    lam = np.exp(beta0 + beta @ np.asarray(x_new, dtype=float))
    if kind == "weibull":
        k = posterior_flat(idata, "k")
        return lam * np.log(2.0) ** (1.0 / k)
    if kind == "exponential":
        return lam * np.log(2.0)
    if kind == "lognormal":
        return lam  # LogNormal 的 λ 就是中位數
    raise ValueError(kind)


def population_survival(idata, X: np.ndarray, t_grid: np.ndarray, kind: str = "weibull",
                        max_draws: int = 500, seed: int = 0) -> np.ndarray:
    """族群平均存活曲線：對樣本中每個人算 S(t|x_i) 再平均。

    這是要跟 Kaplan–Meier 曲線比較的對象。注意不能用「平均特徵的存活曲線」
    代替「存活曲線的平均」—— S 是非線性的，兩者不等（同 A1 的 Jensen 現象）。
    """
    beta0 = posterior_flat(idata, "beta0")
    beta = posterior_flat(idata, "beta")
    S_total = len(beta0)
    idx = (np.random.default_rng(seed).choice(S_total, max_draws, replace=False)
           if S_total > max_draws else np.arange(S_total))

    lam = np.exp(beta0[idx][:, None] + beta[idx] @ np.asarray(X, dtype=float).T)  # (S, n)
    tg = np.asarray(t_grid, dtype=float)[None, None, :]

    if kind == "weibull":
        k = posterior_flat(idata, "k")[idx][:, None, None]
        S = np.exp(-((tg / lam[:, :, None]) ** k))
    elif kind == "exponential":
        S = np.exp(-tg / lam[:, :, None])
    else:
        from scipy import stats

        sigma = posterior_flat(idata, "sigma")[idx][:, None, None]
        S = stats.lognorm.sf(tg, s=sigma, scale=lam[:, :, None])
    return S.mean(axis=1)  # (S, len(t_grid))
