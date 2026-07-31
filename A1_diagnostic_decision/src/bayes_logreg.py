"""
A1 · 貝葉斯邏輯迴歸（PyMC）+ 後驗預測 vs plug-in。

模型：
    β ~ N(0, prior_sd)   （弱資訊先驗；標準化後 prior_sd=2.5 是邏輯迴歸標準選擇）
    α ~ N(0, 5)
    y ~ Bernoulli(logit = α + Xβ)

重點對比（計劃書步驟 2）：
    plug-in       ：用後驗均值參數做「一次」預測 → sigmoid(E[α]+X·E[β])
    後驗預測分佈  ：用「全部後驗樣本」各預測一次再平均 → E[sigmoid(α_s+X·β_s)]
    兩者因 sigmoid 非線性（Jensen 不等式）而不同：後驗預測會被不確定性「拉向 0.5」，
    而且每個病人都帶一條可信區間，plug-in 沒有。
"""
from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm


def fit(X, y, prior_sd: float = 2.5, intercept_sd: float = 5.0,
        draws: int = 2000, tune: int = 1000, chains: int = 4, seed: int = 42):
    """擬合貝葉斯邏輯迴歸，回傳 InferenceData。"""
    with pm.Model() as model:
        beta = pm.Normal("beta", 0.0, prior_sd, shape=X.shape[1])
        alpha = pm.Normal("alpha", 0.0, intercept_sd)
        pm.Bernoulli("y", logit_p=alpha + pm.math.dot(X, beta), observed=y)
        idata = pm.sample(draws, tune=tune, chains=chains, target_accept=0.9,
                          random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": False})
    return idata, model


def _flat_params(idata):
    """把 (chain, draw) 攤平成 (S,) 與 (S, F)。"""
    post = idata.posterior
    alpha = post["alpha"].values.reshape(-1)
    beta = post["beta"].values.reshape(-1, post["beta"].shape[-1])
    return alpha, beta


def posterior_predictive_proba(idata, X) -> np.ndarray:
    """回傳形狀 (S, N) 的後驗預測機率：每個後驗樣本對每個病人各算一次。"""
    alpha, beta = _flat_params(idata)
    logits = alpha[:, None] + beta @ X.T            # (S, N)
    return 1.0 / (1.0 + np.exp(-logits))


def plugin_proba(idata, X) -> np.ndarray:
    """plug-in：用後驗均值參數做單次預測，回傳 (N,)。"""
    alpha, beta = _flat_params(idata)
    logit = alpha.mean() + X @ beta.mean(0)
    return 1.0 / (1.0 + np.exp(-logit))


def predictive_mean(idata, X) -> np.ndarray:
    """後驗預測的均值機率 E[sigmoid(...)]，回傳 (N,)。"""
    return posterior_predictive_proba(idata, X).mean(0)


def predictive_interval(idata, X, cred: float = 0.95):
    """每個病人的後驗預測機率可信區間，回傳 (lo, hi)，各為 (N,)。"""
    p = posterior_predictive_proba(idata, X)
    a = (1 - cred) / 2 * 100
    return np.percentile(p, a, axis=0), np.percentile(p, 100 - a, axis=0)


def convergence(idata) -> dict:
    """回傳最差 r_hat 與最小 ESS，供收斂檢查。"""
    summ = az.summary(idata, var_names=["alpha", "beta"])
    return {"max_rhat": float(summ["r_hat"].max()),
            "min_ess_bulk": float(summ["ess_bulk"].min())}
