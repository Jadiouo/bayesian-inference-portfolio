"""
C2 · 評估：後驗預測校準、R_t 不確定性、殘差週期性、模型比較。

三個通關標準各有一個對應的量化函式，刻意不靠目視判斷：
  1. `rt_at_policy_changes`   —— R_t 在政策時點附近有變化
  2. `rt_width_vs_cases`      —— 資料稀疏時不確定性帶變寬
  3. `coverage`               —— 後驗預測帶包住約 95% 的觀測
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from sir import simulate_numpy


def _flat(idata, name: str) -> np.ndarray:
    """(chain, draw, ...) → (sample, ...)"""
    da = idata.posterior[name]
    return da.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


# ---------------------------------------------------------------------------
# 通關標準 3：後驗預測校準
# ---------------------------------------------------------------------------


def posterior_predictive(idata, ep, seed: int = 0, max_draws: int = 2000) -> np.ndarray:
    """從後驗抽負二項觀測樣本 → (n_draws, n_days)。

    重點是抽**觀測**而不是只畫 μ 的區間。μ 的區間只含參數不確定性；
    真正要檢查的是「未來的一天會落在哪」，那還要加上負二項的觀測噪聲。
    只報 μ 的區間會讓覆蓋率看起來遠低於 95%，然後被誤判為模型不好。
    """
    mu = _flat(idata, "mu")
    alpha = _flat(idata, "alpha")
    rng = np.random.default_rng(seed)
    if len(mu) > max_draws:
        idx = rng.choice(len(mu), max_draws, replace=False)
        mu, alpha = mu[idx], alpha[idx]
    # PyMC 的 NegativeBinomial(mu, alpha)：變異數 = mu + mu²/alpha
    p = alpha[:, None] / (alpha[:, None] + mu)
    return rng.negative_binomial(alpha[:, None] * np.ones_like(mu), p)


def coverage(observed, pp_samples, levels=(0.5, 0.8, 0.95)) -> dict:
    """後驗預測帶的實際覆蓋率。

    通關標準是「95% 帶大約包住 95% 的觀測」。同時報 50% 與 80% 帶：
    只看 95% 會漏掉「帶太寬」的問題 —— 一個把所有東西都包住的模型
    在 95% 上滿分，但在 50% 上會嚴重過覆蓋，那代表它其實沒有資訊。
    """
    obs = np.asarray(observed, dtype=float)
    out = {}
    for lv in levels:
        lo = np.percentile(pp_samples, 100 * (1 - lv) / 2, axis=0)
        hi = np.percentile(pp_samples, 100 * (1 + lv) / 2, axis=0)
        inside = (obs >= lo) & (obs <= hi)
        out[f"cov_{int(lv * 100)}"] = float(inside.mean())
        out[f"target_{int(lv * 100)}"] = lv
    out["n_days"] = int(len(obs))
    # 校準誤差：各水準上 |實際 − 名目| 的平均
    out["mean_abs_calib_error"] = float(np.mean(
        [abs(out[f"cov_{int(lv * 100)}"] - lv) for lv in levels]))
    return out


def forecast(idata, ep_train, n_ahead: int, seed: int = 0, max_draws: int = 1000,
             dow_future=None) -> dict:
    """從訓練期末狀態向前模擬 n_ahead 天（**樣本外**預測）。

    關鍵設計：β 繼續走 random walk
        log β_{T+k} = log β_T + σ_β · Σ_{j≤k} z_j,  z_j ~ N(0,1)

    這讓預測帶隨時間**自然變寬** —— 因為模型誠實地說「我不知道未來的
    傳染率會怎麼變」。如果把 β 固定在最後一天的值，預測帶會過窄，
    而覆蓋率檢查就會通過得太漂亮（那是作弊，不是校準）。

    固定 β 的模型沒有 σ_β，此時 β 保持常數 —— 那是它的模型假設，不是簡化。
    """
    rng = np.random.default_rng(seed)
    N = float(ep_train.population)
    n_train = ep_train.n_days

    S_post = _flat(idata, "S")
    I_post = _flat(idata, "I")
    logb = _flat(idata, "log_beta")
    gamma = _flat(idata, "gamma")
    rho = _flat(idata, "rho")
    alpha = _flat(idata, "alpha")
    has_rw = "sigma_beta" in idata.posterior
    sig = _flat(idata, "sigma_beta") if has_rw else None
    has_dow = "dow_effect" in idata.posterior
    dow_eff = _flat(idata, "dow_effect") if has_dow else None

    n = len(gamma)
    idx = rng.choice(n, min(max_draws, n), replace=False)
    cases_pred = np.empty((len(idx), n_ahead))
    newinf_pred = np.empty((len(idx), n_ahead))
    rt_pred = np.empty((len(idx), n_ahead))

    for k, j in enumerate(idx):
        s, i = float(S_post[j, -1]), float(I_post[j, -1])
        lb = float(logb[j, -1])
        if has_rw:
            steps = rng.normal(0.0, float(sig[j]), n_ahead)
            beta_future = np.exp(lb + np.cumsum(steps))
        else:
            beta_future = np.repeat(np.exp(lb), n_ahead)

        sim = simulate_numpy(beta_future, float(gamma[j]), s, i, N, n_ahead)
        newinf_pred[k] = sim["new_infections"]
        rt_pred[k] = (beta_future / float(gamma[j])) * (sim["S"] / N)

        mult = np.ones(n_ahead)
        if has_dow and dow_future is not None:
            mult = dow_eff[j][np.asarray(dow_future)]
        mu = float(rho[j]) * sim["new_infections"] * mult + 1e-6
        a = float(alpha[j])
        cases_pred[k] = rng.negative_binomial(a, a / (a + mu))

    return {"cases": cases_pred, "new_infections": newinf_pred, "R_t": rt_pred,
            "n_ahead": n_ahead, "n_draws": len(idx)}


# ---------------------------------------------------------------------------
# 通關標準 1：R_t 與政策時點
# ---------------------------------------------------------------------------


def rt_trace(idata) -> dict:
    """R_t 的後驗分位數軌跡。"""
    rt = _flat(idata, "R_t")
    return {"median": np.median(rt, axis=0),
            "lo95": np.percentile(rt, 2.5, axis=0),
            "hi95": np.percentile(rt, 97.5, axis=0),
            "lo50": np.percentile(rt, 25, axis=0),
            "hi50": np.percentile(rt, 75, axis=0),
            "p_below_1": (rt < 1.0).mean(axis=0),
            "samples": rt}


def rt_at_policy_changes(idata, ep, window: int = 10) -> dict:
    """量化「R_t 在政策時點附近有變化」。

    對每個政策日 t₀，比較前 `window` 天與後 `window` 天的 R_t 後驗：
    直接對**後驗樣本**算差值分佈，得到 P(R_t 下降) 而不只是點估計的差。

    也報告 R_t 首次「有 95% 後驗機率低於 1」的日期 —— 那是流行病學上
    最有意義的里程碑（疫情開始收縮）。
    """
    rt = _flat(idata, "R_t")
    n = rt.shape[1]
    out = {"policies": {}}
    for label, t0 in ep.policy_indices().items():
        a0, a1 = max(0, t0 - window), t0
        b0, b1 = t0, min(n, t0 + window)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        before = rt[:, a0:a1].mean(axis=1)
        after = rt[:, b0:b1].mean(axis=1)
        d = before - after
        out["policies"][label] = {
            "day_index": t0, "date": str(ep.dates[t0].date()),
            "rt_before_median": float(np.median(before)),
            "rt_after_median": float(np.median(after)),
            "drop_median": float(np.median(d)),
            "drop_lo95": float(np.percentile(d, 2.5)),
            "drop_hi95": float(np.percentile(d, 97.5)),
            "p_decreased": float((d > 0).mean()),
        }
    p_below = (rt < 1.0).mean(axis=0)
    below_idx = np.where(p_below > 0.95)[0]
    out["first_day_rt_below_1_p95"] = (None if len(below_idx) == 0
                                       else {"day_index": int(below_idx[0]),
                                             "date": str(ep.dates[below_idx[0]].date())})
    out["rt_start_median"] = float(np.median(rt[:, 0]))
    out["rt_end_median"] = float(np.median(rt[:, -1]))
    return out


# ---------------------------------------------------------------------------
# 通關標準 2：資料稀疏 → 不確定性變寬
# ---------------------------------------------------------------------------


def rt_width_vs_cases(idata, ep) -> dict:
    """把 R_t 的區間寬度對當日病例數作迴歸 —— 通關標準 2 的量化版。

    ⚠️ **必須用相對（log 尺度）寬度，不能用絕對寬度。** 這是實測踩到的坑：
    R_t 在疫情早期約 3.2、在尾端約 0.6，差 5 倍。絕對寬度會跟著這個
    乘法尺度縮放，於是「尾端病例少但區間窄」看起來像是「資料少反而更確定」——
    完全反直覺，而且是度量的假影。第一版就得到 ratio 0.57×（方向相反）。

    正確的量是 `log(hi95 / lo95)`，即**相對**寬度。理由不只是尺度：
    模型裡 β_t 走的是 **log-scale** 的 random walk，所以不確定性天生
    定義在 log 尺度上，在那裡才可比。

    預期：病例數少的日子（疫情早期與尾端）相對區間明顯更寬 ——
    epistemic 不確定性的直接體現，少的資料就是少的資訊。
    若計數噪聲主導，相對寬度 ∝ 1/√cases（log-log 斜率 −0.5）；
    random walk 先驗讓相鄰日互相「借」資訊，把斜率往 0 拉，
    所以實測值落在 (−0.5, 0) 之間才合理。
    """
    rt = _flat(idata, "R_t")
    lo = np.percentile(rt, 2.5, axis=0)
    hi = np.percentile(rt, 97.5, axis=0)
    width_abs = hi - lo
    # 相對寬度：log(hi/lo)，與 R_t 的整體量級無關
    width_rel = np.log(np.maximum(hi, 1e-12) / np.maximum(lo, 1e-12))
    cases = ep.cases.astype(float)

    m = cases > 0
    slope, intercept, r, p, se = stats.linregress(np.log(cases[m]), np.log(width_rel[m]))
    q1, q3 = np.percentile(cases[m], [25, 75])
    lo_grp = width_rel[m][cases[m] <= q1]
    hi_grp = width_rel[m][cases[m] >= q3]

    # 同時保留絕對寬度的比較，好在 README 裡把「為什麼不能用它」講清楚
    lo_abs = width_abs[m][cases[m] <= q1]
    hi_abs = width_abs[m][cases[m] >= q3]
    return {
        "loglog_slope": float(slope), "loglog_slope_se": float(se),
        "loglog_r": float(r), "loglog_p": float(p),
        "width_rel_lowcases_median": float(np.median(lo_grp)),
        "width_rel_highcases_median": float(np.median(hi_grp)),
        "width_ratio_low_over_high": float(np.median(lo_grp) / np.median(hi_grp)),
        # 絕對寬度的同一比較 —— 展示度量選擇如何翻轉結論
        "width_abs_ratio_low_over_high": float(np.median(lo_abs) / np.median(hi_abs)),
        "cases_q25": float(q1), "cases_q75": float(q3),
        "width_rel": width_rel, "width_abs": width_abs, "cases": cases,
        "rt_median": np.median(rt, axis=0),
    }


# ---------------------------------------------------------------------------
# 週末效應的必要性
# ---------------------------------------------------------------------------


def residual_periodicity(idata, ep, max_lag: int = 21) -> dict:
    """殘差的自相關 —— 「不建模週末效應會留下什麼」的證據。

    用 Pearson 殘差（負二項的變異數是 μ + μ²/α，不能只除以 √μ）。
    若 day-of-week 效應沒被建模，殘差在 lag 7、14、21 應出現明顯正自相關。

    同時做 Ljung–Box 式的檢定量（只看 lag 7 的倍數），
    因為「隨便某個 lag 顯著」和「週期性顯著」是不同的主張。
    """
    mu = _flat(idata, "mu").mean(axis=0)
    alpha = float(_flat(idata, "alpha").mean())
    var = mu + mu**2 / alpha
    resid = (ep.cases - mu) / np.sqrt(np.maximum(var, 1e-12))
    resid = resid - resid.mean()

    n = len(resid)
    denom = float((resid**2).sum())
    acf = np.array([float((resid[:n - k] * resid[k:]).sum()) / denom
                    for k in range(max_lag + 1)])
    weekly_lags = [k for k in (7, 14, 21) if k <= max_lag]
    ci95 = 1.96 / np.sqrt(n)
    return {
        "acf": acf, "lags": np.arange(max_lag + 1), "ci95": float(ci95),
        "acf_at_weekly_lags": {int(k): float(acf[k]) for k in weekly_lags},
        "max_weekly_acf": float(max(abs(acf[k]) for k in weekly_lags)),
        "n_weekly_lags_significant": int(sum(abs(acf[k]) > ci95 for k in weekly_lags)),
        "resid": resid,
    }


# ---------------------------------------------------------------------------
# 模型比較
# ---------------------------------------------------------------------------


def compare_models(idatas: dict) -> dict:
    """WAIC / LOO 比較三個版本。

    與 A2 專案同樣的理由：不用貝氏因子（對先驗太敏感），
    用估計樣本外預測表現的 WAIC/LOO。

    ⚠️ **但對時間序列，逐點 LOO 在概念上就有問題**，這不是實作瑕疵：
    留一個時間點出來，它的鄰居仍在資料裡，而 random walk 先驗讓鄰居
    攜帶了幾乎全部的資訊 —— 於是 LOO 系統性**高估**預測能力。
    實測 `max_pareto_k` 達 0.8 以上（> 0.7 就代表 LOO 的重要性採樣估計
    本身不可靠），正是這個結構問題的徵兆。

    所以本專案把 WAIC/LOO 當**相對**指標用（哪個模型結構更好），
    而「模型能不能預測未來」由 `forecast` + `coverage` 的
    **向前 14 天樣本外檢查**回答。後者才是時間序列該用的判準。
    回傳值裡保留 max_pareto_k，好讓這個限制在報告裡看得見。
    """
    import warnings

    import arviz as az

    rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = az.compare(idatas, ic="waic", scale="log")
        for name, idata in idatas.items():
            w = az.waic(idata, scale="log")
            l = az.loo(idata, scale="log")
            rows[name] = {"elpd_waic": float(w.elpd_waic), "se_waic": float(w.se),
                          "p_waic": float(w.p_waic), "elpd_loo": float(l.elpd_loo),
                          "p_loo": float(l.p_loo),
                          "max_pareto_k": float(np.max(l.pareto_k.to_numpy()))}
    best = max(rows, key=lambda k: rows[k]["elpd_waic"])
    others = {k: {"delta_elpd": rows[best]["elpd_waic"] - rows[k]["elpd_waic"],
                  "dse": float(cmp.loc[k, "dse"]) if k in cmp.index else float("nan")}
              for k in rows if k != best}
    return {"per_model": rows, "best": best, "vs_best": others,
            "table": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                               else str(vv))
                          for kk, vv in cmp.loc[k].items()} for k in cmp.index}}
