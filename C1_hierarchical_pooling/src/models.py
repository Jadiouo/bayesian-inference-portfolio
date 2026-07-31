"""
C1 · 三個 Radon 模型 + Eight Schools（中心化 / 非中心化）。

Radon：log_radon ~ Normal(a[county] + b·floor, sigma)
  - no pooling      ：每郡各自的截距，獨立先驗 a_j ~ N(0,10)——小樣本郡極不穩
  - complete pooling：全州共用一個截距 a——完全忽略郡的差異
  - hierarchical    ：a_j ~ N(mu_a, sigma_a)，超先驗學出母體分佈——部分匯聚

階層模型用**非中心化參數化** a = mu_a + sigma_a·a_raw 避免 funnel（計劃書 §9 / 主題五第三卡）。
"""
from __future__ import annotations

import numpy as np
import pymc as pm

SAMPLE = dict(target_accept=0.9, random_seed=1, progressbar=False,
              idata_kwargs={"log_likelihood": False})


def fit_no_pooling(cidx, floor, y, J, draws=1000, tune=1000, chains=4):
    with pm.Model():
        a = pm.Normal("a", 0, 10, shape=J)
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("y", a[cidx] + b * floor, sigma, observed=y)
        return pm.sample(draws, tune=tune, chains=chains, **SAMPLE)


def fit_complete_pooling(cidx, floor, y, J, draws=1000, tune=1000, chains=4):
    with pm.Model():
        a = pm.Normal("a", 0, 10)
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("y", a + b * floor, sigma, observed=y)
        return pm.sample(draws, tune=tune, chains=chains, **SAMPLE)


def fit_hierarchical(cidx, floor, y, J, draws=1000, tune=1000, chains=4, centered=False):
    with pm.Model():
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 5)
        if centered:
            a = pm.Normal("a", mu_a, sigma_a, shape=J)
        else:
            a = pm.Deterministic("a", mu_a + sigma_a * pm.Normal("a_raw", 0, 1, shape=J))
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("y", a[cidx] + b * floor, sigma, observed=y)
        return pm.sample(draws, tune=tune, chains=chains, **SAMPLE)


def county_intercepts(idata, J: int) -> np.ndarray:
    """回傳各郡截距的後驗均值 (J,)。complete pooling 的純量會廣播成 J 個相同值。"""
    a = idata.posterior["a"]
    if a.ndim == 2:                       # complete pooling：(chain, draw)
        return np.full(J, float(a.mean()))
    return a.mean(("chain", "draw")).values


def predict(idata, cidx_test, floor_test, J):
    """點預測（後驗均值）：a[county] + b·floor。"""
    a = county_intercepts(idata, J)
    b = float(idata.posterior["b"].mean())
    return a[cidx_test] + b * floor_test


# ─────────────── Eight Schools（funnel 示範）───────────────
def fit_eight_schools(y, sigma, centered: bool, draws=1000, tune=1000, chains=4):
    with pm.Model():
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfNormal("tau", 5)
        if centered:
            theta = pm.Normal("theta", mu, tau, shape=len(y))
        else:
            theta = pm.Deterministic("theta", mu + tau * pm.Normal("theta_raw", 0, 1, shape=len(y)))
        pm.Normal("obs", theta, sigma, observed=y)
        return pm.sample(draws, tune=tune, chains=chains, **SAMPLE)
