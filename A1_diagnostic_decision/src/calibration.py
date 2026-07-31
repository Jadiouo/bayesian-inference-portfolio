"""
A1 · 校準分析（計劃書步驟 5）。

一個模型「校準良好」是指：在它說「70% 機率」的那些病人裡，真的約 70% 有病。
校準與鑑別度（AUC）是**兩回事**——AUC 很高的模型仍可能校準很差，
而在做決策（門檻、期望損失）時，我們吃的是機率的絕對值，校準差就危險。

  - reliability diagram：把預測機率分箱，比較每箱的預測信心 vs 實際發生率
  - ECE（Expected Calibration Error）：各箱 |實際 − 預測| 的樣本數加權平均
"""
from __future__ import annotations

import numpy as np


def reliability(y_true, p, n_bins: int = 10):
    """回傳 (bin_conf, bin_acc, bin_count)：各箱的平均預測、實際發生率、樣本數。"""
    y_true = np.asarray(y_true, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, int)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b] > 0:
            conf[b] = p[m].mean()
            acc[b] = y_true[m].mean()
    return conf, acc, cnt


def ece(y_true, p, n_bins: int = 10) -> float:
    """Expected Calibration Error。"""
    conf, acc, cnt = reliability(y_true, p, n_bins)
    m = cnt > 0
    w = cnt[m] / cnt.sum()
    return float(np.sum(w * np.abs(acc[m] - conf[m])))
