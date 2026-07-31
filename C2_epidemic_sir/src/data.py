"""
C2 · 資料載入 —— COVID 每日確診數。

⚠️ **偏離計劃書的資料選擇，以及為什麼**
------------------------------------------
計劃書指定 C-D3（Our World in Data）。但 OWID 已把 COVID 資料集**回溯性地改成
週報格式**：整週的總和堆在一天，其餘六天是 0。實測德國 2020-03～06 有
**104/122 天是 0**（例如 2020-03-29 一天記 33,981 例）。

這對本專案是致命的：
- 每日 SIR 觀測模型無法用（六天零、一天暴衝不是傳染動態，是通報格式）
- **週末效應根本不存在了**，而骨架明確要求處理它
- `new_cases_smoothed` 雖是每日值，但在週內是常數（階梯狀），
  用它會人為地讓觀測「太平滑」，過度分散的估計失真

所以主資料改用 **JHU CSSE**（累積確診的每日時間序列，差分得每日新增）。
它保留真正的每日結構：德國 2020 第一波的 day-of-week 平均是
週一 1248 → 週五 1991（差 60%），正是要建模的週期性。

為了不讓「換資料來源」變成一個沒有檢查的決定，`cross_validate_sources()`
會比對 JHU 與 OWID 的**週總和** —— 兩個獨立來源的週總和若一致，
就說明差異只在時間解析度，不是資料本身有問題。

關於通報延遲
------------
骨架提到「通報延遲讓最近幾天資料系統性偏低」。本專案用的是 **2020 年的歷史
資料，已經被兩邊回溯校正過**，所以這個問題在此設定下幾乎不存在。
這與「即時建模」是本質不同的情境 —— 即時做的話最後 1–2 週必須截掉或
顯式建模延遲分佈。README 的限制章節記錄這個差異，因為它讓本專案的
校準結果比即時情境樂觀。
"""
from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

JHU_URL = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
           "csse_covid_19_data/csse_covid_19_time_series/"
           "time_series_covid19_confirmed_global.csv")
OWID_URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# 德國 2020 第一波的政策時點（供 R_t 對照，通關標準 1）
# 來源：德國聯邦政府公告；只取全國性、日期明確的措施。
GERMANY_POLICIES = {
    "2020-03-16": "Schools closed (most states)",
    "2020-03-22": "Nationwide contact ban (Kontaktverbot)",
    "2020-04-20": "First reopening steps",
    "2020-04-27": "Masks mandatory (public transport / shops)",
}


@dataclass
class Epidemic:
    """一段疫情時間序列。"""

    country: str
    dates: pd.DatetimeIndex
    cases: np.ndarray            # 每日新確診（整數）
    population: float
    dow: np.ndarray              # 0=Mon … 6=Sun
    policies: dict = field(default_factory=dict)
    n_negative_clipped: int = 0

    @property
    def n_days(self) -> int:
        return len(self.cases)

    def policy_indices(self) -> dict:
        """把政策日期換成時間序列的索引位置（不在範圍內的略去）。"""
        out = {}
        for d, label in self.policies.items():
            ts = pd.Timestamp(d)
            if ts in self.dates:
                out[label] = int(self.dates.get_loc(ts))
        return out

    def summary(self) -> str:
        return (f"{self.country}: {self.n_days} days "
                f"{self.dates[0].date()}–{self.dates[-1].date()}, "
                f"cases {self.cases.min():.0f}–{self.cases.max():.0f} "
                f"(total {self.cases.sum():.0f}), N={self.population / 1e6:.1f}M")


def _download(url: str, dst: str, timeout: int = 300) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
            f.write(r.read())
    return dst


def load_jhu_daily(data_dir: str, country: str = "Germany") -> pd.Series:
    """JHU CSSE 累積確診 → 每日新增。

    JHU 的原始檔是「每個省/州一列、每天一欄」的寬表，同一國家可能有多列
    （如中國各省），所以要先按國家加總再差分。
    """
    dst = _download(JHU_URL, os.path.join(data_dir, "jhu_confirmed_global.csv"))
    df = pd.read_csv(dst)
    g = df[df["Country/Region"] == country]
    if len(g) == 0:
        raise ValueError(f"JHU 資料裡找不到 {country}")
    cum = g.iloc[:, 4:].sum(axis=0)
    cum.index = pd.to_datetime(cum.index, format="%m/%d/%y")
    return cum.sort_index().diff().dropna()


def load_owid_weekly(data_dir: str, country: str = "Germany") -> pd.DataFrame:
    """OWID 的（週報化）每日欄位，供交叉驗證用。"""
    dst = _download(OWID_URL, os.path.join(data_dir, "owid_compact.csv"))
    df = pd.read_csv(dst, usecols=["country", "date", "new_cases", "population"],
                     parse_dates=["date"])
    g = df[df["country"] == country].set_index("date").sort_index()
    return g


def cross_validate_sources(data_dir: str, country: str, start: str, end: str) -> dict:
    """比對 JHU 與 OWID 的**週總和** —— 換資料來源這個決定的檢查。

    兩邊的每日結構完全不同（JHU 真每日、OWID 週報），但如果它們描述同一場疫情，
    週總和應該接近。差異大就表示至少有一邊的資料有問題，
    那時「換來源」就不是解決方案而是換一個問題。
    """
    jhu = load_jhu_daily(data_dir, country).loc[start:end]
    owid = load_owid_weekly(data_dir, country).loc[start:end, "new_cases"].fillna(0.0)

    jw = jhu.resample("W").sum()
    ow = owid.resample("W").sum()
    common = jw.index.intersection(ow.index)
    jw, ow = jw.loc[common].to_numpy(), ow.loc[common].to_numpy()

    # ⚠️ 用**對稱**相對差（SMAPE 式），不用 |Δ|/OWID。
    # 後者在某一週的 OWID 總和趨近 0 時會爆出無意義的巨大比值
    # （實測 760 倍）——那是**週邊界對齊**的假影：OWID 把整週總和記在某一天，
    # 該天若落在 resample 的週界另一側，就會出現「一週 0、相鄰週雙倍」。
    # 那不是兩個來源在描述不同的疫情，所以指標不該讓它看起來像。
    denom = (np.abs(jw) + np.abs(ow)) / 2.0
    ok = denom > 0
    smape = np.abs(jw[ok] - ow[ok]) / denom[ok]
    return {
        "n_weeks": int(len(jw)),
        "jhu_total": float(jw.sum()), "owid_total": float(ow.sum()),
        # 主要證據：總量幾乎相同、週序列高度相關
        "total_rel_diff": float(abs(jw.sum() - ow.sum()) / max(ow.sum(), 1.0)),
        "corr_weekly": float(np.corrcoef(jw, ow)[0, 1]),
        # 次要：逐週的對稱相對差，反映週邊界對齊誤差而非資料矛盾
        "weekly_smape_median": float(np.median(smape)),
        "weekly_smape_p90": float(np.percentile(smape, 90)),
        "owid_zero_day_fraction": float((owid == 0).mean()),
    }


def load(data_dir: str, country: str = "Germany", start: str = "2020-03-01",
         end: str = "2020-06-30", drop_last_days: int = 0) -> Epidemic:
    """載入一段每日確診序列。

    `drop_last_days`：即時建模時用來截掉通報未完成的尾端。
    本專案用歷史（已回溯校正）資料，預設 0；參數保留以說明正確做法。
    """
    daily = load_jhu_daily(data_dir, country).loc[start:end]
    if drop_last_days > 0:
        daily = daily.iloc[:-drop_last_days]

    vals = daily.to_numpy(dtype=float)
    # JHU 偶有負值（回溯校正造成）。本專案的時段沒有，但仍顯式處理：
    # 負的「新增病例」在觀測模型下無意義（負二項的支撐是非負整數）。
    n_neg = int((vals < 0).sum())
    vals = np.clip(vals, 0.0, None)

    owid = load_owid_weekly(data_dir, country)
    pop = float(owid["population"].dropna().iloc[-1])

    return Epidemic(country=country, dates=daily.index, cases=vals, population=pop,
                    dow=daily.index.dayofweek.to_numpy(),
                    policies=GERMANY_POLICIES if country == "Germany" else {},
                    n_negative_clipped=n_neg)


def day_of_week_profile(ep: Epidemic) -> dict:
    """各星期的平均病例數與相對全體平均的比值 —— 週末效應的描述統計。

    這是「該不該加 day-of-week 項」的第一道證據：
    若各日的比值都接近 1，就不需要那組參數。
    """
    overall = ep.cases.mean()
    prof = {}
    for i, name in enumerate(DOW):
        m = ep.dow == i
        prof[name] = {"mean": float(ep.cases[m].mean()),
                      "ratio": float(ep.cases[m].mean() / overall),
                      "n": int(m.sum())}
    ratios = np.array([prof[d]["ratio"] for d in DOW])
    prof["_spread"] = {"min_ratio": float(ratios.min()), "max_ratio": float(ratios.max()),
                       "max_over_min": float(ratios.max() / ratios.min())}
    return prof


def train_test_split(ep: Epidemic, n_forecast: int = 14) -> tuple[Epidemic, Epidemic]:
    """把尾端 `n_forecast` 天留作**樣本外**預測檢查（通關標準 3）。

    樣本內的後驗預測帶必然偏窄（模型見過那些點），只報告它會高估校準品質。
    真正的檢查是「模型沒見過的未來 14 天，預測帶包不包住真值」。
    """
    k = len(ep.cases) - n_forecast
    a = Epidemic(ep.country, ep.dates[:k], ep.cases[:k], ep.population, ep.dow[:k],
                 ep.policies, ep.n_negative_clipped)
    b = Epidemic(ep.country, ep.dates[k:], ep.cases[k:], ep.population, ep.dow[k:],
                 ep.policies, 0)
    return a, b
