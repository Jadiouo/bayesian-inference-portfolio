"""
D1 · 貝葉斯 A/B 測試 —— 核心 Beta-Binomial 分析。

轉換率 A/B 測試：每組觀察到 n 次曝光、k 次轉換。
共軛更新（計劃書主題二）：θ ~ Beta(α₀+k, β₀+n−k)，**封閉解，不需 MCMC**——
這也是它在業界受歡迎的原因（即時計算）。

本模組提供「兩個 Beta 比較」的三個關鍵量：
  - prob_b_beats_a()  P(θ_B > θ_A)        ← 老闆真正問的問題（p-value 無法回答）
  - expected_loss_*() E[(θ_A − θ_B)^+]     ← 選錯的期望損失（決策理論，主題七）
  - lift_ci()         相對提升幅度的可信區間

「兩個 Beta 比較」用**網格數值積分**（精確、快、可重現）為主，並提供
蒙地卡羅版本作交叉驗證，以及常態近似版本（大樣本下極準，供模擬迴圈加速用）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ─────────────────────────── 先驗與後驗 ───────────────────────────
@dataclass(frozen=True)
class Prior:
    """Beta 先驗。預設 Beta(1,1)=Uniform（弱資訊）。"""

    alpha: float = 1.0
    beta: float = 1.0

    def label(self) -> str:
        return f"Beta({self.alpha:g}, {self.beta:g})"


def posterior(k: int, n: int, prior: Prior = Prior()) -> tuple[float, float]:
    """Beta-Binomial 共軛後驗參數 (α, β)。"""
    if not (0 <= k <= n):
        raise ValueError(f"需 0 <= k <= n，收到 k={k}, n={n}")
    return prior.alpha + k, prior.beta + n - k


def beta_mean_sd(a: float, b: float) -> tuple[float, float]:
    """Beta(a,b) 的均值與標準差。"""
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return mean, float(np.sqrt(var))


# ─────────────────────── P(θ_B > θ_A)：三種算法 ───────────────────────
def _grid(num: int = 1000) -> np.ndarray:
    # 端點稍微內縮；本專案 α,β≥1，pdf 在 0/1 有界，trapz 精度足夠
    return np.linspace(1e-6, 1 - 1e-6, num)


def prob_b_beats_a(aA, bA, aB, bB, grid: int = 1000) -> float:
    """P(θ_B > θ_A)，網格數值積分：∫ pdf_A(x)·SF_B(x) dx。（主用，精確可重現）"""
    x = _grid(grid)
    return float(np.trapezoid(stats.beta.pdf(x, aA, bA) * stats.beta.sf(x, aB, bB), x))


def prob_b_beats_a_mc(aA, bA, aB, bB, rng: np.random.Generator, size: int = 200_000) -> float:
    """P(θ_B > θ_A)，蒙地卡羅（計劃書 §5 步驟 2 的寫法，用於交叉驗證）。"""
    return float((rng.beta(aB, bB, size) > rng.beta(aA, bA, size)).mean())


def prob_b_beats_a_normal(aA, bA, aB, bB) -> float:
    """P(θ_B > θ_A)，常態近似。大樣本下與精確值誤差 < 0.5pp，供模擬迴圈加速。"""
    mA, sA = beta_mean_sd(aA, bA)
    mB, sB = beta_mean_sd(aB, bB)
    return float(stats.norm.cdf((mB - mA) / np.hypot(sA, sB)))


# ─────────────────────── 期望損失 E[(θ₁ − θ₂)^+] ───────────────────────
def expected_positive_diff(a1, b1, a2, b2, grid: int = 1000) -> float:
    """
    E[(θ₁ − θ₂)^+]，θ ~ Beta。網格積分，1D 化：
        E[(θ₁−θ₂)^+] = ∫ pdf₁(t)·E[(t−θ₂)^+] dt
        E[(t−θ₂)^+]  = t·CDF₂(t) − mean₂·CDF_{Beta(a₂+1,b₂)}(t)
    """
    t = _grid(grid)
    mean2 = a2 / (a2 + b2)
    inner = t * stats.beta.cdf(t, a2, b2) - mean2 * stats.beta.cdf(t, a2 + 1, b2)
    return float(np.trapezoid(stats.beta.pdf(t, a1, b1) * inner, t))


def _expected_positive_diff_normal(a1, b1, a2, b2) -> float:
    """E[(θ₁ − θ₂)^+] 常態近似。D=θ₁−θ₂~N(μ,σ²)，E[D^+]=μΦ(μ/σ)+σφ(μ/σ)。"""
    m1, s1 = beta_mean_sd(a1, b1)
    m2, s2 = beta_mean_sd(a2, b2)
    mu, sigma = m1 - m2, np.hypot(s1, s2)
    if sigma == 0:
        return max(mu, 0.0)
    z = mu / sigma
    return float(mu * stats.norm.cdf(z) + sigma * stats.norm.pdf(z))


def expected_loss_choose_b(aA, bA, aB, bB, grid: int = 1000) -> float:
    """上線 B 的期望損失＝選 B 但其實 A 較好的後悔＝E[(θ_A − θ_B)^+]。"""
    return expected_positive_diff(aA, bA, aB, bB, grid)


def expected_loss_choose_a(aA, bA, aB, bB, grid: int = 1000) -> float:
    """維持 A 的期望損失＝E[(θ_B − θ_A)^+]。"""
    return expected_positive_diff(aB, bB, aA, bA, grid)


# ─────────────────────────── 提升幅度區間 ───────────────────────────
def lift_ci(
    aA, bA, aB, bB, rng: np.random.Generator, cred: float = 0.95, size: int = 200_000
) -> tuple[float, float, float]:
    """相對提升幅度 θ_B/θ_A − 1 的 (下界, 中位數, 上界)，蒙地卡羅。"""
    sa = rng.beta(aA, bA, size)
    sb = rng.beta(aB, bB, size)
    lift = sb / sa - 1.0
    lo, mid, hi = np.percentile(lift, [(1 - cred) / 2 * 100, 50, (1 + cred) / 2 * 100])
    return float(lo), float(mid), float(hi)


# ─────────────────────────── 一次算完的摘要 ───────────────────────────
@dataclass
class ABSummary:
    kA: int
    nA: int
    kB: int
    nB: int
    prior: Prior
    postA: tuple[float, float]
    postB: tuple[float, float]
    rateA: float
    rateB: float
    prob_b_beats_a: float
    prob_a_beats_b: float
    expected_loss_choose_b: float
    expected_loss_choose_a: float
    lift_lo: float
    lift_med: float
    lift_hi: float

    def boss_answer(self) -> str:
        """一段老闆聽得懂的話。"""
        return (
            f"A 版 {self.rateA:.2%}、B 版 {self.rateB:.2%}。"
            f"B 優於 A 的機率 = {self.prob_b_beats_a:.1%}；"
            f"相對提升的 95% 可信區間 = [{self.lift_lo:+.1%}, {self.lift_hi:+.1%}]（中位數 {self.lift_med:+.1%}）；"
            f"若上線 B 但其實 A 較好，期望損失 = {self.expected_loss_choose_b:.4%} 轉換率。"
        )


def summarize(
    kA: int, nA: int, kB: int, nB: int,
    prior: Prior = Prior(), rng: np.random.Generator | None = None,
) -> ABSummary:
    """對一組 A/B 觀測，算出所有決策相關的量。"""
    if rng is None:
        rng = np.random.default_rng(0)
    aA, bA = posterior(kA, nA, prior)
    aB, bB = posterior(kB, nB, prior)
    p_ba = prob_b_beats_a(aA, bA, aB, bB)
    lo, mid, hi = lift_ci(aA, bA, aB, bB, rng)
    return ABSummary(
        kA=kA, nA=nA, kB=kB, nB=nB, prior=prior,
        postA=(aA, bA), postB=(aB, bB),
        rateA=kA / nA, rateB=kB / nB,
        prob_b_beats_a=p_ba, prob_a_beats_b=1 - p_ba,
        expected_loss_choose_b=expected_loss_choose_b(aA, bA, aB, bB),
        expected_loss_choose_a=expected_loss_choose_a(aA, bA, aB, bB),
        lift_lo=lo, lift_med=mid, lift_hi=hi,
    )
