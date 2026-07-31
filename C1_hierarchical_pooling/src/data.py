"""
C1 · 資料載入。

主資料 Radon（C-D1）：919 戶、85 郡、明尼蘇達州氡氣濃度。階層模型教科書範例。
  - y = log_radon（對數氡濃度）
  - floor：0=地下室、1=一樓（一樓通常較低）
  - county：分組變數（85 郡，樣本數 1–116，中位數 5 → 收縮效果生動）

次資料 Eight Schools（C-D2）：8 所學校的教學成效，Rubin (1981) 經典值。
用來示範 funnel 與非中心化參數化（radon 資料量大、funnel 不明顯，故另用它）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm


@dataclass
class Radon:
    cidx: np.ndarray      # (N,) 郡索引 0..J-1
    floor: np.ndarray     # (N,)
    y: np.ndarray         # (N,) log_radon
    counties: np.ndarray  # (J,) 郡名
    n_j: np.ndarray       # (J,) 各郡樣本數

    @property
    def J(self) -> int:
        return len(self.counties)

    @property
    def N(self) -> int:
        return len(self.y)


def load_radon() -> Radon:
    df = pd.read_csv(pm.get_data("radon.csv"))
    cidx, counties = pd.factorize(df["county"])
    return Radon(
        cidx=cidx.astype(int),
        floor=df["floor"].to_numpy(float),
        y=df["log_radon"].to_numpy(float),
        counties=np.asarray(counties),
        n_j=np.bincount(cidx),
    )


def train_test_split(radon: Radon, test_frac: float = 0.25, seed: int = 0):
    """回傳 (train_mask, test_mask)。每個郡至少保留 1 戶在 train，
    使 no-pooling 對所有測試郡都有估計。"""
    rng = np.random.default_rng(seed)
    is_test = np.zeros(radon.N, bool)
    for j in range(radon.J):
        idx = np.where(radon.cidx == j)[0]
        if len(idx) >= 2:
            n_test = min(int(round(len(idx) * test_frac)), len(idx) - 1)
            if n_test > 0:
                is_test[rng.choice(idx, n_test, replace=False)] = True
    return ~is_test, is_test


def eight_schools():
    """Rubin (1981) 經典值：8 所學校的處理效應 y 與已知標準誤 sigma。"""
    y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])
    names = np.array(list("ABCDEFGH"))
    return y, sigma, names
