"""
A2 · 資料載入與預處理 —— GBSG2 乳癌生存資料（A-D5）。

German Breast Cancer Study Group 2：686 位淋巴結陽性乳癌病人，追蹤至多 7.3 年。
299 人觀察到復發/死亡事件，**387 人（56.4%）在追蹤期結束時仍無事件 → 右刪失**。
這個高刪失率正是本專案的主題：那 387 筆不是「沒事件」，是「我還不知道」。

資料來自 lifelines 內建（`load_gbsg2`），無需下載；為了與其他專案一致，
仍把一份 CSV 快取到統一資料夾 data/A_medical/ 供離線重現與檢查。

時間單位一律換算成**年**（原始為天）。這不只是可讀性：
Weibull 的尺度參數 λ 直接是「特徵時間」，用年為單位時 λ~5 這種量級
才能配上合理的弱資訊先驗；用天則 λ~1800，先驗尺度很難設得有意義。

預處理決策
----------
- `progrec` / `estrec` / `pnodes` / `tsize` 極度右偏（skew 1.8–4.8），做 log1p
  後降到 |skew|<0.6。受體濃度與淋巴結數在臨床上本來就是乘性效應（「多一倍」
  比「多一單位」有意義），log 尺度同時解決統計與語意問題。
- 連續變數標準化，理由同 A1：弱資訊先驗 N(0, σ) 假設預測變數已標準化，
  否則先驗尺度對不同單位的特徵沒有一致意義。
- `tgrade` 是**有序**的（I < II < III），不做 one-hot 而編碼成 0/1/2 再標準化，
  用一個參數換取單調性假設 —— 686 筆資料裡 grade I 只有 81 人，省參數划算。
  這個選擇在 README 的限制章節有記錄。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines.datasets import load_gbsg2

LOG1P = ["tsize", "pnodes", "progrec", "estrec"]
CONTINUOUS = ["age", "tsize", "pnodes", "progrec", "estrec", "tgrade"]
BINARY = ["horTh", "menostat"]
FEATURES = CONTINUOUS + BINARY

# 人類可讀的特徵名（給圖表用，matplotlib 無 CJK → 英文）
LABELS = {
    "age": "Age",
    "tsize": "log Tumor size",
    "pnodes": "log Positive nodes",
    "progrec": "log Progesterone rec.",
    "estrec": "log Estrogen rec.",
    "tgrade": "Tumor grade (ord.)",
    "horTh": "Hormone therapy",
    "menostat": "Postmenopausal",
}

DAYS_PER_YEAR = 365.25


@dataclass
class Survival:
    """一份生存資料：設計矩陣 X、觀察時間 t（年）、事件指示 event（1=事件, 0=右刪失）。"""

    X: np.ndarray
    t: np.ndarray
    event: np.ndarray
    features: list[str]
    df_raw: pd.DataFrame | None = None

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    @property
    def n_censored(self) -> int:
        return int((1 - self.event).sum())

    @property
    def censoring_rate(self) -> float:
        return float(1.0 - self.event.mean())

    def summary(self) -> str:
        return (
            f"n={self.n}, events={self.n_events}, censored={self.n_censored} "
            f"({100 * self.censoring_rate:.1f}%), "
            f"follow-up {self.t.min():.2f}–{self.t.max():.2f} yr"
        )


def _cache_csv(data_dir: str) -> pd.DataFrame:
    """載入 GBSG2 並快取一份 CSV 到統一資料夾（供離線重現）。"""
    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, "gbsg2.csv")
    if os.path.exists(dst):
        return pd.read_csv(dst)
    df = load_gbsg2()
    df.to_csv(dst, index=False)
    return df


def load(data_dir: str, standardize: bool = True) -> tuple[Survival, dict]:
    """載入 GBSG2 並預處理。

    回傳 (Survival, scaler_info)。scaler_info 記錄各連續變數的 mean/std，
    讓「把某個病人的特徵值換算回原始單位」在畫圖與敘事時可行。
    """
    df = _cache_csv(data_dir).copy()

    df["horTh"] = (df["horTh"] == "yes").astype(float)
    df["menostat"] = (df["menostat"] == "Post").astype(float)
    df["tgrade"] = df["tgrade"].map({"I": 0.0, "II": 1.0, "III": 2.0}).astype(float)
    for c in LOG1P:
        df[c] = np.log1p(df[c].astype(float))

    X_df = df[FEATURES].astype(float)
    scaler = {}
    if standardize:
        for c in CONTINUOUS:
            mu, sd = X_df[c].mean(), X_df[c].std(ddof=0)
            scaler[c] = {"mean": float(mu), "std": float(sd)}
            X_df[c] = (X_df[c] - mu) / sd

    t = df["time"].to_numpy(dtype=float) / DAYS_PER_YEAR
    event = df["cens"].to_numpy(dtype=int)  # GBSG2 的 cens: 1=事件, 0=刪失

    return Survival(
        X=X_df.to_numpy(dtype=float),
        t=t,
        event=event,
        features=list(X_df.columns),
        df_raw=df,
    ), scaler


def drop_censored(s: Survival) -> Survival:
    """錯誤做法 A：把刪失樣本整批丟掉（complete-case 分析）。

    直覺上「資料不完整就不要用」，但被丟掉的是**活得比較久**的那群人
    （GBSG2 裡刪失者中位追蹤 3.95 年 vs 事件者中位 1.77 年），
    剩下的樣本系統性偏向短存活。
    """
    m = s.event == 1
    return Survival(X=s.X[m], t=s.t[m], event=s.event[m], features=s.features)


def censored_as_event(s: Survival) -> Survival:
    """錯誤做法 B：把刪失當成事件（「追蹤結束時就算他死了」）。

    每一筆刪失資料的真實事件時間都 > 觀察時間，全部當成事件
    等於系統性把生存時間往下拉。
    """
    return Survival(X=s.X, t=s.t, event=np.ones_like(s.event), features=s.features)


def truncate_followup(s: Survival, horizon: float) -> Survival:
    """把行政刪失時間往前推到 `horizon` 年：t>horizon 的人改記為在 horizon 時刪失。

    樣本數完全不變，**只有事件數下降**。用來展示生存分析的核心事實：
    後驗精度由事件數決定，不是樣本數。
    """
    t = np.minimum(s.t, horizon)
    event = np.where(s.t > horizon, 0, s.event)
    return Survival(X=s.X.copy(), t=t, event=event, features=s.features)
