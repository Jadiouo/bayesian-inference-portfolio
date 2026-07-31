"""
C1 · 收縮（shrinkage）的量化。

部分匯聚的每郡估計 ≈ 精確度加權平均（計劃書主題二第二卡）：
    a_partial_j ≈ (1−ω_j)·a_nopool_j + ω_j·a_complete
其中「拉向母體平均的權重」
    ω_j = (1/σ_a²) / (n_j/σ² + 1/σ_a²) = σ² / (σ² + n_j·σ_a²)
n_j 越大 → ω_j 越小 → 收縮越弱。這條理論曲線可以直接和實際後驗估計對照。
"""
from __future__ import annotations

import numpy as np


def empirical_weight(a_nopool, a_partial, a_complete):
    """實際收縮權重 ω_j = (a_nopool − a_partial)/(a_nopool − a_complete)。"""
    denom = a_nopool - a_complete
    return np.where(np.abs(denom) > 1e-9, (a_nopool - a_partial) / denom, np.nan)


def theoretical_weight(n, sigma, sigma_a):
    """理論收縮權重 ω(n) = σ²/(σ² + n·σ_a²)。"""
    n = np.asarray(n, float)
    return sigma**2 / (sigma**2 + n * sigma_a**2)


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))
