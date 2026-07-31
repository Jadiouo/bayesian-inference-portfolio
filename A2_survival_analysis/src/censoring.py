"""
A2 · 刪失似然 —— 本專案的靈魂。

生存資料的似然不是「每筆資料都貢獻一個密度」。分兩種：

    觀察到事件（t_i 時死亡/復發）  → 貢獻機率密度   f(t_i)
    右刪失（t_i 時仍無事件）        → 貢獻存活函數   S(t_i) = 1 - F(t_i)

        L = ∏_{i: 事件} f(t_i) · ∏_{i: 刪失} S(t_i)

為什麼刪失貢獻 S 而不是 f？因為我們觀察到的事件不是「在 t_i 死亡」，而是
「真實事件時間 T_i > t_i」這個**區間事件**，它的機率就是 S(t_i)。

    🔑 刪失不是遺失資料。它攜帶真實資訊：「至少活到了 t_i」。

本模組手刻三個分佈的 log f 與 log S（純 numpy/scipy，可獨立檢查），
再用 `verify_equivalence` 逐點證明它與 PyMC 的 `pm.Censored` 給出同樣的 logp。
手刻的目的不是取代 PyMC，而是**確認自己真的知道 PyMC 在算什麼** ——
`pm.Censored` 是個黑箱，如果不能重現它，就無法在它出錯時發現。
"""
from __future__ import annotations

import numpy as np
import pymc as pm
from scipy import stats

# ---------------------------------------------------------------------------
# 手刻 log-density 與 log-survival
#
# 參數化一律採 AFT（accelerated failure time）：特徵透過尺度參數 λ 作用，
# 效果是「把時間軸拉長或壓縮」，係數的語意是 log(時間倍率)。
# ---------------------------------------------------------------------------


def weibull_logpdf(t, k, lam):
    """log f(t)；Weibull, shape k, scale λ。S(t)=exp(-(t/λ)^k)。"""
    z = t / lam
    return np.log(k) - np.log(lam) + (k - 1.0) * np.log(z) - z**k


def weibull_logsf(t, k, lam):
    """log S(t) = -(t/λ)^k。這一行就是刪失資料的全部貢獻。"""
    return -((t / lam) ** k)


def lognormal_logpdf(t, sigma, lam):
    """log f(t)；Log-Normal，中位數 λ（即 median survival），log 尺度 σ。"""
    return stats.lognorm.logpdf(t, s=sigma, scale=lam)


def lognormal_logsf(t, sigma, lam):
    return stats.lognorm.logsf(t, s=sigma, scale=lam)


def exponential_logpdf(t, lam):
    """log f(t)；Exponential = Weibull 的 k=1 特例（風險率恆定）。"""
    return -np.log(lam) - t / lam


def exponential_logsf(t, lam):
    return -t / lam


KERNELS = {
    "weibull": (weibull_logpdf, weibull_logsf),
    "lognormal": (lognormal_logpdf, lognormal_logsf),
    "exponential": (exponential_logpdf, exponential_logsf),
}


def censored_loglik(t, event, kind: str, *params) -> np.ndarray:
    """逐點 log-likelihood：事件用 log f，刪失用 log S。

    回傳長度 n 的陣列（不加總），方便檢查是哪些觀測值貢獻異常。
    """
    logpdf, logsf = KERNELS[kind]
    event = np.asarray(event, dtype=bool)
    out = np.empty(len(t), dtype=float)
    out[event] = logpdf(np.asarray(t)[event], *[_sel(p, event) for p in params])
    out[~event] = logsf(np.asarray(t)[~event], *[_sel(p, ~event) for p in params])
    return out


def _sel(p, mask):
    """參數可能是純量（共用）或長度 n 的陣列（每個個體不同的 λ_i）。"""
    p = np.asarray(p)
    return p[mask] if p.ndim > 0 and p.shape[0] == mask.shape[0] else p


# ---------------------------------------------------------------------------
# 與 pm.Censored 的等價驗證
# ---------------------------------------------------------------------------


def censoring_bounds(t, event) -> np.ndarray:
    """`pm.Censored` 的 upper 界：事件筆設 inf（不刪失），刪失筆設觀察時間本身。

    pm.Censored 的語意是「觀察值被夾在 [lower, upper]」：
      - observed < upper → 未被夾住 → logp = log f(observed)
      - observed == upper → 被夾在上界 → logp = log S(upper)
    所以把刪失筆的 upper 設成它自己的觀察時間，就自動切換到存活函數。
    """
    return np.where(np.asarray(event) == 1, np.inf, np.asarray(t, dtype=float))


def _pymc_logp(kind: str, t, event, params):
    """用 PyMC 的 Censored 算逐點 logp（不建完整模型，直接對 dist 求 logp）。"""
    ub = censoring_bounds(t, event)
    if kind == "weibull":
        k, lam = params
        latent = pm.Weibull.dist(alpha=k, beta=lam)
    elif kind == "lognormal":
        sigma, lam = params
        # PyMC 的 LogNormal 用 (mu, sigma) on log scale；median = exp(mu) = λ
        latent = pm.LogNormal.dist(mu=np.log(lam), sigma=sigma)
    elif kind == "exponential":
        (lam,) = params
        latent = pm.Exponential.dist(scale=lam)
    else:
        raise ValueError(kind)
    censored = pm.Censored.dist(latent, lower=None, upper=ub)
    return pm.logp(censored, np.asarray(t, dtype=float)).eval()


def verify_equivalence(t, event, kind: str, params, rtol: float = 1e-8) -> dict:
    """逐點比對手刻 log-lik 與 pm.Censored，回傳最大絕對/相對誤差。

    這是本專案的自我檢查：如果兩者不合，要嘛是我理解錯了刪失似然，
    要嘛是參數化對不上（例如把 PyMC LogNormal 的 mu 當成 median）。
    兩種錯誤都會安靜地產生看似合理但錯誤的後驗。
    """
    mine = censored_loglik(t, event, kind, *params)
    theirs = _pymc_logp(kind, t, event, params)
    abs_err = np.abs(mine - theirs)
    denom = np.maximum(np.abs(theirs), 1e-12)
    rel_err = abs_err / denom
    return {
        "kind": kind,
        "max_abs_err": float(abs_err.max()),
        "max_rel_err": float(rel_err.max()),
        "total_loglik_mine": float(mine.sum()),
        "total_loglik_pymc": float(theirs.sum()),
        "n_events": int(np.sum(np.asarray(event) == 1)),
        "n_censored": int(np.sum(np.asarray(event) == 0)),
        "passed": bool(np.allclose(mine, theirs, rtol=rtol, atol=1e-8)),
    }


def censoring_information_demo(t_obs: float, k: float, lam: float) -> dict:
    """量化「刪失攜帶多少資訊」的一個小示範。

    對同一個觀察時間 t_obs，比較兩種解讀：
      - 當成事件：log f(t_obs)
      - 當成刪失：log S(t_obs)
    並給出「若當成事件，隱含的錯誤」= 真實事件時間必然 > t_obs，
    但事件似然把全部機率質量押在 t_obs 這一點上。
    """
    return {
        "t": t_obs,
        "logpdf_if_event": float(weibull_logpdf(t_obs, k, lam)),
        "logsf_if_censored": float(weibull_logsf(t_obs, k, lam)),
        "survival_prob": float(np.exp(weibull_logsf(t_obs, k, lam))),
    }
