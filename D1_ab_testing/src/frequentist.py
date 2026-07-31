"""
D1 · 頻率派對照 —— 兩比例 z 檢定。

用來凸顯貝葉斯的差異：p-value 回答的是「若 A、B 其實一樣，看到這麼極端（或更極端）
資料的機率」，**不是**老闆問的「B 比 A 好的機率」。偷看實驗也需要它作對照組。
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def two_proportion_z_test(kA: int, nA: int, kB: int, nB: int) -> tuple[float, float]:
    """
    合併變異數的兩比例 z 檢定。回傳 (z, 雙尾 p-value)。
    H0: p_A = p_B。z > 0 表示 B 的觀測轉換率較高。
    """
    if nA == 0 or nB == 0:
        return 0.0, 1.0
    pA, pB = kA / nA, kB / nB
    pooled = (kA + kB) / (nA + nB)
    se = np.sqrt(pooled * (1 - pooled) * (1 / nA + 1 / nB))
    if se == 0:
        return 0.0, 1.0
    z = (pB - pA) / se
    p_two_sided = 2 * stats.norm.sf(abs(z))
    return float(z), float(p_two_sided)
