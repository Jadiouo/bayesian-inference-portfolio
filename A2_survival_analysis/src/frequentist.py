"""
A2 · 頻率派對照 —— Cox 比例風險、Weibull AFT、Kaplan–Meier。

這一層存在的意義不是「證明貝葉斯比較好」，而是**對照區間的解讀差異**
（計劃書主題三）：

    Cox 的 95% 信賴區間：在「重複這個研究很多次」的想像中，
                         有 95% 的區間會覆蓋真值。對手上這一個區間，
                         不能說「參數有 95% 機率落在裡面」。
    貝葉斯 95% 可信區間：給定這筆資料與先驗，參數落在此區間的
                         後驗機率就是 95%。這是病人真正想問的問題。

數值上兩者在這份資料上會非常接近（n=686、先驗弱）。**這件事本身就是結論**：
資料充足時兩派收斂，貝葉斯的價值不在「答案不同」，而在
(a) 答案能被正確地當成機率解讀、(b) 能自然地傳播到下游決策與個體預測。

Cox 還有一個結構性差異：它把基準風險 h₀(t) 當成無窮維的討厭參數
（nuisance parameter）用 partial likelihood 消掉。好處是不必假設分佈形狀，
代價是**沒有 h₀(t) 的估計就無法直接給出存活曲線的不確定性** ——
`predict_survival_function` 回傳的是把 Breslow 估計當成真值的一條線。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.statistics import proportional_hazard_test


def to_frame(s, duration_col: str = "t", event_col: str = "event") -> pd.DataFrame:
    df = pd.DataFrame(s.X, columns=s.features)
    df[duration_col] = s.t
    df[event_col] = s.event
    return df


def fit_cox(s) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Cox 比例風險模型。回傳 (fitter, 係數表)。

    係數是 log hazard ratio，方向與 AFT 相反：
    風險比 >1（β_PH>0）意味事件來得更快 → 存活時間更短（β_AFT<0）。
    """
    cph = CoxPHFitter()
    cph.fit(to_frame(s), duration_col="t", event_col="event")
    tbl = cph.summary[["coef", "coef lower 95%", "coef upper 95%", "se(coef)", "p"]].copy()
    tbl.columns = ["log_hr", "lo95", "hi95", "se", "p"]
    return cph, tbl


def fit_weibull_aft(s) -> tuple[WeibullAFTFitter, pd.DataFrame]:
    """頻率派 Weibull AFT —— 與貝葉斯模型**同一個似然**，只差先驗與推論方式。

    這是最乾淨的對照組：如果貝葉斯後驗均值和它的 MLE 差很多，
    那差異來自先驗（可解釋）；如果差不多，就證明先驗確實是弱資訊的。
    """
    aft = WeibullAFTFitter()
    aft.fit(to_frame(s), duration_col="t", event_col="event")
    tbl = aft.summary.loc["lambda_"][["coef", "coef lower 95%", "coef upper 95%", "se(coef)", "p"]].copy()
    tbl.columns = ["log_time_ratio", "lo95", "hi95", "se", "p"]
    return aft, tbl


def fit_km(s) -> tuple[KaplanMeierFitter, dict]:
    """Kaplan–Meier 非參數估計 —— 存活曲線的「地面真相」基準。

    KM 本身就正確處理刪失（刪失者在其後的風險集合中被移除，不計為事件），
    所以它是檢查參數化模型是否嚴重失配的好標尺。
    """
    km = KaplanMeierFitter()
    km.fit(s.t, event_observed=s.event)
    med = km.median_survival_time_
    ci = km.confidence_interval_survival_function_
    return km, {
        "median_survival": float(med) if np.isfinite(med) else float("nan"),
        "timeline": km.timeline,
        "survival": km.survival_function_["KM_estimate"].to_numpy(),
        "lo": ci.iloc[:, 0].to_numpy(),
        "hi": ci.iloc[:, 1].to_numpy(),
    }


def km_median(s) -> float:
    """只要 KM 中位存活時間（給三種刪失處理的對照用）。

    注意：把刪失當事件或丟掉刪失時，KM 也會被污染 —— 這正是重點。
    KM 不是萬靈丹，它只是**在刪失被正確標記時**才無偏。
    """
    km = KaplanMeierFitter()
    km.fit(s.t, event_observed=s.event)
    med = km.median_survival_time_
    return float(med) if np.isfinite(med) else float("nan")


def ph_assumption_test(cph: CoxPHFitter, s) -> pd.DataFrame:
    """檢定比例風險假設是否成立（Schoenfeld 殘差）。

    Cox 的整個推論建立在「風險比不隨時間變化」上。這個檢定放進專案是因為
    誠實：如果 PH 假設被違反，Cox 的係數就是某種時間平均，
    而我的 Weibull AFT 也同樣假設了 PH（Weibull 是 PH 模型），一起錯。
    """
    res = proportional_hazard_test(cph, to_frame(s), time_transform="rank")
    return pd.DataFrame({"test_statistic": res.test_statistic, "p": res.p_value})


def cox_survival_at(cph: CoxPHFitter, x_new: np.ndarray, features: list[str],
                    t_grid: np.ndarray) -> np.ndarray:
    """Cox 對單一病人的存活曲線 —— 只有一條線，沒有不確定性帶。

    這正是通關標準 2 的對照組。
    """
    row = pd.DataFrame([dict(zip(features, np.asarray(x_new, dtype=float)))])
    sf = cph.predict_survival_function(row, times=t_grid)
    return sf.to_numpy().ravel()
