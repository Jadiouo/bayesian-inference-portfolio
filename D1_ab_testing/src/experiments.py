"""
D1 · 序貫測試與「偷看資料」的模擬（計劃書 §5 步驟 3，⭐ 最有說服力的實驗）。

作法：模擬大量 A/A 測試（兩組其實一樣），每天累積資料。在每個「偷看點」用不同
停止規則判斷是否宣布勝利。因為兩組相同，**任何宣布都是假陽性**。

三個規則：
  - 頻率派：雙尾 z 檢定 p < α 就宣布
  - 貝葉斯後驗門檻：max(P(B>A), P(A>B)) > thresh 就宣布
  - 期望損失：領先組的期望損失 < ε 就上線

關鍵誠實點（計劃書明言）：貝葉斯**不是萬靈丹**——用「後驗機率 > 0.95 就停」當
停止規則，長期假陽性率一樣會被偷看膨脹。本模擬把這件事量出來。

效能：每個「模擬×天」的統計量用**常態近似**向量化預計算一次（大樣本下極準，見
run_all 的交叉驗證），之後所有排程與規則都只是查表。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from bayes_ab import Prior


# ─────────────────────── 情境設定與資料生成 ───────────────────────
@dataclass(frozen=True)
class Scenario:
    pA: float
    pB: float
    per_day: int       # 每組每天的曝光數
    days: int          # 追蹤天數（= 最多偷看次數，逐日）
    name: str = ""

    @property
    def is_aa(self) -> bool:
        return self.pA == self.pB


def simulate_cumulative(scn: Scenario, n_sims: int, rng: np.random.Generator):
    """
    回傳每天累積的轉換數 (kA, kB)，形狀皆 (n_sims, days)。
    每天新增 per_day 次曝光，nA = nB = per_day·(day+1)。
    """
    dailyA = rng.binomial(scn.per_day, scn.pA, size=(n_sims, scn.days))
    dailyB = rng.binomial(scn.per_day, scn.pB, size=(n_sims, scn.days))
    return np.cumsum(dailyA, axis=1), np.cumsum(dailyB, axis=1)


# ─────────────────────── 每個偷看點的統計量（向量化）───────────────────────
def _post_stats(k, n, prior: Prior):
    """Beta 後驗均值與標準差（向量化）。"""
    a = prior.alpha + k
    b = prior.beta + (n - k)
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return mean, np.sqrt(var)


@dataclass
class PeekTable:
    """每個模擬×天的決策量。形狀皆 (n_sims, days)。"""

    p_value: np.ndarray        # 頻率派雙尾 p
    prob_ba: np.ndarray        # P(θ_B > θ_A)
    el_leader: np.ndarray      # 領先組的期望損失
    leader_is_b: np.ndarray    # 領先組是否為 B


def build_peek_table(kA, kB, nA_per_day, prior: Prior) -> PeekTable:
    """從累積轉換數，向量化算出每天的 p-value / P(B>A) / 期望損失（常態近似）。"""
    days = kA.shape[1]
    n = nA_per_day * (np.arange(days) + 1)          # 每天的 n（A、B 相同）
    n = n[None, :]                                   # 廣播成 (1, days)
    kA = kA.astype(float)
    kB = kB.astype(float)

    mA, sA = _post_stats(kA, n, prior)
    mB, sB = _post_stats(kB, n, prior)
    mu = mB - mA
    sigma = np.hypot(sA, sB)
    z_diff = mu / sigma

    prob_ba = stats.norm.cdf(z_diff)
    phi = stats.norm.pdf(z_diff)
    ship_loss_b = (-mu) * stats.norm.cdf(-z_diff) + sigma * phi   # E[(θ_A−θ_B)^+]
    ship_loss_a = mu * stats.norm.cdf(z_diff) + sigma * phi       # E[(θ_B−θ_A)^+]
    leader_is_b = mu >= 0
    el_leader = np.where(leader_is_b, ship_loss_b, ship_loss_a)

    # 頻率派雙尾 z 檢定
    pA, pB = kA / n, kB / n
    pooled = (kA + kB) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (pB - pA) / se, 0.0)
    p_value = 2 * stats.norm.sf(np.abs(z))

    return PeekTable(p_value=p_value, prob_ba=prob_ba,
                     el_leader=el_leader, leader_is_b=leader_is_b)


# ─────────────────────── 停止規則與宣布率 ───────────────────────
def _fires_freq(t: PeekTable, cols, alpha: float):
    return t.p_value[:, cols] < alpha


def _fires_bayes(t: PeekTable, cols, thresh: float):
    p = t.prob_ba[:, cols]
    return np.maximum(p, 1 - p) > thresh


def _fires_exploss(t: PeekTable, cols, eps: float):
    return t.el_leader[:, cols] < eps


def declare_rate(fires: np.ndarray) -> float:
    """在給定偷看排程的欄位上，任一次觸發即算「宣布」。回傳宣布比例。"""
    return float(fires.any(axis=1).mean())


def even_schedule(days: int, n_looks: int) -> np.ndarray:
    """在 [0, days) 內取 n_looks 個大致等距的偷看日索引（含最後一天）。"""
    if n_looks >= days:
        return np.arange(days)
    idx = np.unique(np.linspace(0, days - 1, n_looks).round().astype(int))
    return idx


@dataclass
class RuleSpec:
    kind: str          # "freq" | "bayes" | "exploss"
    param: float
    label: str


def fires_for_rule(t: PeekTable, cols, rule: RuleSpec) -> np.ndarray:
    if rule.kind == "freq":
        return _fires_freq(t, cols, rule.param)
    if rule.kind == "bayes":
        return _fires_bayes(t, cols, rule.param)
    if rule.kind == "exploss":
        return _fires_exploss(t, cols, rule.param)
    raise ValueError(rule.kind)


def fpr_vs_looks(t: PeekTable, days: int, looks_grid, rules) -> dict:
    """對每個規則、每個偷看次數 K，算宣布率（A/A 下即假陽性率）。"""
    out = {r.label: [] for r in rules}
    for K in looks_grid:
        cols = even_schedule(days, K)
        for r in rules:
            out[r.label].append(declare_rate(fires_for_rule(t, cols, r)))
    return {"looks": list(looks_grid), "rates": {k: np.array(v) for k, v in out.items()}}


# ─────────────────────── 供圖用的後驗軌跡 ───────────────────────
def prob_ba_trajectories(scn: Scenario, n_sims: int, prior: Prior,
                         rng: np.random.Generator) -> np.ndarray:
    """回傳 (n_sims, days) 的 P(B>A) 每日軌跡，用於畫「偷看會亂走」的示意圖。"""
    kA, kB = simulate_cumulative(scn, n_sims, rng)
    return build_peek_table(kA, kB, scn.per_day, prior).prob_ba
