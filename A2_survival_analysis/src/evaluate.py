"""
A2 · 模型比較與預測評估。

分兩套工具，各有其位：

**WAIC / LOO-CV（`az.compare`）** —— 用全資料，比較三個基準分佈。
為什麼不用貝氏因子？因為它對先驗太敏感（Lindley 悖論，B2 專案已用
injection–recovery 實測過：先驗放寬 3 倍就能讓結論翻盤）。WAIC/LOO
估計的是**樣本外預測表現**，對弱資訊先驗的具體寬度遠不敏感。

WAIC 的懲罰項 `p_waic` 是「有效參數個數」——不是你寫了幾個參數，
而是資料實際「用掉」多少自由度。它等於逐點 log-likelihood 的後驗變異數之和：
參數若被資料緊緊約束，該點的 log-lik 在後驗上變動小，貢獻的懲罰就小。
這是它比 AIC/BIC（固定懲罰 k 或 k·log n/2）更貼近實情的地方。

**C-index 與 IPCW Brier score** —— 用 5-fold OOF，衡量真正的樣本外預測。
刪失讓評估本身也變棘手：test set 裡有人在 t 之前就刪失了，我們不知道
他在 t 是死是活。Brier score 用 IPCW（inverse probability of censoring
weighting）加權補償這個缺口。C-index 只用可比較的配對（其中一人的事件
時間確定早於另一人），刪失率越高、可用配對越少。
"""
from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

import models as M


# ---------------------------------------------------------------------------
# 貝葉斯模型比較
# ---------------------------------------------------------------------------


def compare(idatas: dict, ic: str = "waic") -> pd.DataFrame:
    """az.compare 排序模型。idatas: {"weibull": idata, ...}"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return az.compare(idatas, ic=ic, scale="log")


def ic_table(idatas: dict) -> pd.DataFrame:
    """把 WAIC 與 LOO 的 elpd、懲罰項、標準誤攤成一張表。

    同時列 p_waic 與 p_loo：兩者都是有效參數數的估計，
    差距大通常暗示有高影響力觀測值（LOO 的 Pareto k 診斷會抓到）。
    """
    rows = []
    for name, idata in idatas.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = az.waic(idata, scale="log")
            l = az.loo(idata, scale="log")
        n_stated = int(idata.posterior["beta"].sizes["feature"]) + 1  # beta + intercept
        n_stated += 1 if "k" in idata.posterior or "sigma" in idata.posterior else 0
        rows.append({
            "model": name,
            "elpd_waic": float(w.elpd_waic),
            "se_waic": float(w.se),
            "p_waic": float(w.p_waic),
            "elpd_loo": float(l.elpd_loo),
            "se_loo": float(l.se),
            "p_loo": float(l.p_loo),
            "n_params_stated": n_stated,
            "max_pareto_k": float(np.max(l.pareto_k.to_numpy())),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("elpd_waic", ascending=False)


# ---------------------------------------------------------------------------
# 樣本外預測評估（處理刪失）
# ---------------------------------------------------------------------------


def _surv_array(t, event):
    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(t, dtype=float))


def c_index(t, event, risk_score) -> tuple[float, int]:
    """Harrell's C-index。risk_score 越大代表風險越高（存活越短）。

    回傳 (c_index, 可比較配對數)。第二個數字值得報出來：
    刪失把大量配對變成不可比較，C-index 的有效樣本比 n(n−1)/2 少很多。
    """
    res = concordance_index_censored(
        np.asarray(event).astype(bool), np.asarray(t, dtype=float), np.asarray(risk_score)
    )
    return float(res[0]), int(res[1] + res[2])  # concordant + discordant


def c_index_bootstrap(t, event, risk_score, n_boot: int = 500, seed: int = 0) -> dict:
    """C-index 的 bootstrap 分佈。

    需要這個是因為 C-index 的差距很容易被過度解讀：三個模型差 0.002
    看起來像排序，但如果 bootstrap 標準誤是 0.02，那就是噪聲。
    重抽時分層在 event 上，保持事件數穩定。
    """
    rng = np.random.default_rng(seed)
    t, event, rs = np.asarray(t, float), np.asarray(event), np.asarray(risk_score)
    ev_idx, cn_idx = np.where(event == 1)[0], np.where(event == 0)[0]
    out = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(ev_idx, len(ev_idx), replace=True),
                            rng.choice(cn_idx, len(cn_idx), replace=True)])
        try:
            out.append(concordance_index_censored(event[b].astype(bool), t[b], rs[b])[0])
        except Exception:
            continue
    out = np.asarray(out)
    return {"mean": float(out.mean()), "se": float(out.std(ddof=1)),
            "lo95": float(np.percentile(out, 2.5)), "hi95": float(np.percentile(out, 97.5)),
            "samples": out}


def paired_c_index_diff(t, event, rs_a, rs_b, n_boot: int = 500, seed: int = 0) -> dict:
    """兩個模型 C-index 差距的 **配對** bootstrap。

    配對很重要：兩個模型在同一批重抽樣本上一起評估，共同的抽樣噪聲被抵消。
    未配對地各自算 CI 再看是否重疊，會嚴重低估區分能力。
    """
    rng = np.random.default_rng(seed)
    t, event = np.asarray(t, float), np.asarray(event)
    rs_a, rs_b = np.asarray(rs_a), np.asarray(rs_b)
    ev_idx, cn_idx = np.where(event == 1)[0], np.where(event == 0)[0]
    diffs = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(ev_idx, len(ev_idx), replace=True),
                            rng.choice(cn_idx, len(cn_idx), replace=True)])
        try:
            ca = concordance_index_censored(event[b].astype(bool), t[b], rs_a[b])[0]
            cb = concordance_index_censored(event[b].astype(bool), t[b], rs_b[b])[0]
        except Exception:
            continue
        diffs.append(ca - cb)
    d = np.asarray(diffs)
    return {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1)),
            "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "p_a_better": float((d > 0).mean())}


def ipcw_brier(t_train, e_train, t_test, e_test, S_pred, times) -> dict:
    """IPCW Brier score 與 integrated Brier score。

    S_pred: (n_test, len(times)) 的存活機率預測。
    times 必須嚴格落在 test set 的 follow-up 範圍內，否則 IPCW 權重無法估計。
    """
    times = np.asarray(times, dtype=float)
    lo, hi = np.min(t_test), np.max(t_test)
    keep = (times > lo) & (times < hi)
    times = times[keep]
    S_pred = np.asarray(S_pred)[:, keep]

    surv_train = _surv_array(t_train, e_train)
    surv_test = _surv_array(t_test, e_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, bs = brier_score(surv_train, surv_test, S_pred, times)
        ibs = integrated_brier_score(surv_train, surv_test, S_pred, times)
    return {"times": times, "brier": np.asarray(bs), "ibs": float(ibs)}


def predict_survival_matrix(idata, X, t_grid, kind="weibull", max_draws=400, seed=0) -> np.ndarray:
    """對一批病人算後驗平均存活曲線矩陣 (n, len(t_grid))。

    用後驗**平均**的 S(t|x)（不是把後驗均值參數代進 S），這才是
    貝葉斯的後驗預測存活機率 —— 同 A1 的 posterior-predictive vs plug-in 區別。
    """
    beta0 = M.posterior_flat(idata, "beta0")
    beta = M.posterior_flat(idata, "beta")
    S_total = len(beta0)
    idx = (np.random.default_rng(seed).choice(S_total, max_draws, replace=False)
           if S_total > max_draws else np.arange(S_total))

    lam = np.exp(beta0[idx][:, None] + beta[idx] @ np.asarray(X, dtype=float).T)  # (S, n)
    tg = np.asarray(t_grid, dtype=float)[None, None, :]

    if kind == "weibull":
        k = M.posterior_flat(idata, "k")[idx][:, None, None]
        S = np.exp(-((tg / lam[:, :, None]) ** k))
    elif kind == "exponential":
        S = np.exp(-tg / lam[:, :, None])
    else:
        from scipy import stats

        sigma = M.posterior_flat(idata, "sigma")[idx][:, None, None]
        S = stats.lognorm.sf(tg, s=sigma, scale=lam[:, :, None])
    return S.mean(axis=0)  # 對後驗抽樣平均 → (n, len(t_grid))


def bayes_risk_score(idata, X, kind="weibull") -> np.ndarray:
    """給 C-index 用的風險分數：−E[log λ_i]。

    AFT 下 λ 越大存活越久，所以風險分數取負號。
    C-index 只看排序，用哪個單調變換都一樣。
    """
    beta = M.posterior_flat(idata, "beta").mean(axis=0)
    return -(np.asarray(X, dtype=float) @ beta)


# ---------------------------------------------------------------------------
# 風險函數形狀 —— WAIC 排序的機制解釋
# ---------------------------------------------------------------------------


def hazard_curve(idata, x_ref, t_grid, kind="weibull", max_draws=400, seed=0) -> np.ndarray:
    """h(t|x) 的後驗樣本 (n_draws, len(t_grid))。

    存活曲線 S(t) 長得像的兩個模型，風險函數 h(t) = f(t)/S(t) 可以差很多，
    而 h(t) 才是「此刻的危險程度」——臨床上真正驅動追蹤頻率決策的量。
    三個基準分佈的 h(t) 形狀被硬性限制住：
        Exponential  h(t) = 1/λ                        恆定
        Weibull      h(t) = (k/λ)(t/λ)^(k−1)           k>1 單調遞增、k<1 單調遞減
        Log-Normal   h(t) = f/S                        先升後降，有單一峰值
    所以模型比較不只是挑分數最高的，而是在問「這個疾病的風險隨時間怎麼走」。
    """
    beta0 = M.posterior_flat(idata, "beta0")
    beta = M.posterior_flat(idata, "beta")
    S_total = len(beta0)
    idx = (np.random.default_rng(seed).choice(S_total, max_draws, replace=False)
           if S_total > max_draws else np.arange(S_total))

    lam = np.exp(beta0[idx] + beta[idx] @ np.asarray(x_ref, dtype=float))[:, None]
    tg = np.asarray(t_grid, dtype=float)[None, :]

    if kind == "weibull":
        k = M.posterior_flat(idata, "k")[idx][:, None]
        return (k / lam) * (tg / lam) ** (k - 1.0)
    if kind == "exponential":
        return np.broadcast_to(1.0 / lam, (len(idx), tg.shape[1])).copy()
    if kind == "lognormal":
        from scipy import stats

        sigma = M.posterior_flat(idata, "sigma")[idx][:, None]
        pdf = stats.lognorm.pdf(tg, s=sigma, scale=lam)
        sf = np.clip(stats.lognorm.sf(tg, s=sigma, scale=lam), 1e-12, None)
        return pdf / sf
    raise ValueError(kind)


def population_hazard_curve(idata, X, t_grid, kind="weibull", max_draws=300, seed=0) -> np.ndarray:
    """**族群層級**（marginal）風險函數：

        h_pop(t) = Σ_i S_i(t) h_i(t) / Σ_i S_i(t)

    這才是能跟 Nelson–Aalen 非參數估計直接比較的量，而個體 h(t|x̄) 不是。
    兩者不同不是誤差，是真實的統計現象：族群裡有未被模型完全解釋的異質性，
    高風險的人先離開風險集合，剩下的是低風險者 → **族群風險率的峰值比任何
    個體的峰值都更早、且下降更快**。這在生存分析裡是經典的
    unobserved-heterogeneity / frailty 效應。

    直接拿 h(t|x̄) 去對 Nelson–Aalen 曲線，會誤判模型與資料不合。
    """
    beta0 = M.posterior_flat(idata, "beta0")
    beta = M.posterior_flat(idata, "beta")
    S_total = len(beta0)
    idx = (np.random.default_rng(seed).choice(S_total, max_draws, replace=False)
           if S_total > max_draws else np.arange(S_total))

    lam = np.exp(beta0[idx][:, None] + beta[idx] @ np.asarray(X, dtype=float).T)  # (S, n)
    tg = np.asarray(t_grid, dtype=float)[None, None, :]
    lam3 = lam[:, :, None]

    if kind == "weibull":
        k = M.posterior_flat(idata, "k")[idx][:, None, None]
        S = np.exp(-((tg / lam3) ** k))
        h = (k / lam3) * (tg / lam3) ** (k - 1.0)
    elif kind == "exponential":
        S = np.exp(-tg / lam3)
        h = np.broadcast_to(1.0 / lam3, S.shape)
    elif kind == "lognormal":
        from scipy import stats

        sigma = M.posterior_flat(idata, "sigma")[idx][:, None, None]
        S = np.clip(stats.lognorm.sf(tg, s=sigma, scale=lam3), 1e-12, None)
        h = stats.lognorm.pdf(tg, s=sigma, scale=lam3) / S
    else:
        raise ValueError(kind)

    return (S * h).sum(axis=1) / S.sum(axis=1)  # (S_draws, T)


def nelson_aalen_hazard(t, event, bandwidth: float = 0.6, t_grid=None, min_at_risk: int = 100):
    """非參數風險率估計：對 Nelson–Aalen 累積風險做核平滑後取斜率。

    這是不假設任何分佈形狀的「地面真相」，用來檢查參數化模型的 h(t)
    形狀是不是資料真的支持的，而不只是 WAIC 分數比較好看。

    ⚠️ 一併回傳每個時間點的 **at-risk 人數**，因為這條曲線的尾端不可信：
    GBSG2 在 t=5.5 年只剩 74 人在風險中、6–7.3 年間總共只有 3 個事件。
    未加標示時，那裡會出現一個假的「第二個風險高峰」，純粹是少數事件
    落在極小風險集合上的結果。任何拿這條曲線當地面真相的論證，
    都必須先說清楚它在哪一段才站得住。
    """
    from lifelines import NelsonAalenFitter

    naf = NelsonAalenFitter()
    naf.fit(t, event_observed=event)
    ch = naf.cumulative_hazard_.iloc[:, 0]
    tt, H = ch.index.to_numpy(dtype=float), ch.to_numpy(dtype=float)
    if t_grid is None:
        t_grid = np.linspace(max(tt.min(), 0.1), np.percentile(t, 95), 60)
    t_grid = np.asarray(t_grid, dtype=float)

    # 高斯核加權的局部線性斜率 dH/dt
    h = np.empty_like(t_grid)
    for i, t0 in enumerate(t_grid):
        w = np.exp(-0.5 * ((tt - t0) / bandwidth) ** 2)
        if w.sum() < 1e-8:
            h[i] = np.nan
            continue
        tw = tt - np.average(tt, weights=w)
        var = np.average(tw**2, weights=w)
        h[i] = np.average(tw * (H - np.average(H, weights=w)), weights=w) / max(var, 1e-12)

    t_arr = np.asarray(t, dtype=float)
    at_risk = np.array([int((t_arr >= t0).sum()) for t0 in t_grid])
    reliable = at_risk >= min_at_risk
    t_cut = float(t_grid[reliable].max()) if reliable.any() else float("nan")
    return t_grid, np.clip(h, 0, None), {"at_risk": at_risk, "reliable": reliable,
                                         "min_at_risk": min_at_risk, "t_cut": t_cut}


# ---------------------------------------------------------------------------
# 5-fold OOF 交叉驗證
# ---------------------------------------------------------------------------


def cross_val(s, kind="weibull", n_splits=5, seed=42, draws=1000, tune=1000, t_grid=None):
    """5-fold OOF：每折重新取樣，用 out-of-fold 預測算 C-index 與 IPCW Brier。

    WAIC/LOO 用全資料估計樣本外表現，CV 是它的實測對照。兩者排序一致
    才有信心；不一致就要說明（WAIC 是逐點的漸近近似，CV 直接但噪聲大）。
    依 event 分層，避免某折的事件數過少。
    """
    from sklearn.model_selection import StratifiedKFold

    from data import Survival

    if t_grid is None:
        t_grid = np.linspace(0.5, np.percentile(s.t, 90), 40)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_risk = np.zeros(s.n)
    oof_S = np.zeros((s.n, len(t_grid)))

    for tr, te in skf.split(s.X, s.event):
        s_tr = Survival(X=s.X[tr], t=s.t[tr], event=s.event[tr], features=s.features)
        idata = M.sample(M.build(s_tr, kind), draws=draws, tune=tune, chains=2, seed=seed)
        oof_risk[te] = bayes_risk_score(idata, s.X[te], kind)
        oof_S[te] = predict_survival_matrix(idata, s.X[te], t_grid, kind)

    ci, n_pairs = c_index(s.t, s.event, oof_risk)
    bs = ipcw_brier(s.t, s.event, s.t, s.event, oof_S, t_grid)
    return {
        "kind": kind,
        "c_index": ci,
        "comparable_pairs": n_pairs,
        "total_pairs": s.n * (s.n - 1) // 2,
        "ibs": bs["ibs"],
        "brier_times": bs["times"],
        "brier": bs["brier"],
        "oof_risk": oof_risk,
        "oof_S": oof_S,
        "t_grid": t_grid,
    }
