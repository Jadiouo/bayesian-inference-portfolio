"""
E1 · 目標函數（E-D1 標準測試函數 + E-D2 真實材料配方）。

全部統一成**最小化**問題、定義域正規化到 [0,1]^d。
正規化是必要的，不是方便：GP 的 isotropic lengthscale 假設各維尺度可比，
若 Branin 的 x₁∈[-5,10]、x₂∈[0,15] 直接餵進去，單一 ℓ 就沒有一致的意義。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Objective:
    """一個最小化問題。`f` 吃 [0,1]^d 的點，回傳純量（越小越好）。"""

    name: str
    dim: int
    f: callable
    f_min: float | None = None      # 已知全域最小值（可驗證）
    x_min: np.ndarray | None = None
    noise_std: float = 0.0

    def __call__(self, X: np.ndarray, rng=None) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        vals = np.array([self.f(x) for x in X])
        if self.noise_std > 0 and rng is not None:
            vals = vals + rng.normal(0, self.noise_std, len(vals))
        return vals

    def regret(self, best_value: float) -> float:
        """與已知最優的差距。沒有已知最優時回傳 nan。"""
        return float("nan") if self.f_min is None else float(best_value - self.f_min)


# ---------------------------------------------------------------------------
# E-D1 標準測試函數
# ---------------------------------------------------------------------------


def branin() -> Objective:
    """Branin-Hoo，2D。三個等價的全域最小值 f*=0.397887。

    選它是因為「多個等價最優」正好用來看 κ=0 的行為：
    純利用的 BO 會鎖死在最先碰到的那一個。
    """
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)

    def f(u):
        x1 = -5.0 + 15.0 * u[0]
        x2 = 0.0 + 15.0 * u[1]
        return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    return Objective("Branin", 2, f, f_min=0.397887,
                     x_min=np.array([(np.pi + 5) / 15, 2.275 / 15]))


def hartmann6() -> Objective:
    """Hartmann-6，6D。f* = −3.32237。標準的中維度 BO 基準。"""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=float)
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]], dtype=float)

    def f(u):
        inner = ((A * (u[None, :] - P) ** 2).sum(axis=1))
        return -float(alpha @ np.exp(-inner))

    return Objective("Hartmann6", 6, f, f_min=-3.32237)


def ackley(dim: int, lo: float = -32.768, hi: float = 32.768) -> Objective:
    """Ackley，任意維度。f* = 0 在原點。

    維度詛咒實驗的主角：它有大量局部最優（餘弦項）疊在一個幾乎平坦的
    漏斗上（指數項）。高維時「幾乎平坦」的區域佔絕大部分體積，
    這正是 GP 的資訊優勢消失的地方。
    """
    a, b, c = 20.0, 0.2, 2 * np.pi

    def f(u):
        x = lo + (hi - lo) * np.asarray(u)
        d = len(x)
        s1 = np.sqrt((x**2).sum() / d)
        s2 = np.cos(c * x).sum() / d
        return float(-a * np.exp(-b * s1) - np.exp(s2) + a + np.e)

    # 原點對應正規化座標 0.5
    return Objective(f"Ackley{dim}D", dim, f, f_min=0.0,
                     x_min=np.full(dim, (0.0 - lo) / (hi - lo)))


STANDARD = {"branin": branin, "hartmann6": hartmann6}


# ---------------------------------------------------------------------------
# E-D2 真實應用：混凝土配方
# ---------------------------------------------------------------------------


def load_concrete(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """UCI Concrete Compressive Strength（1030 筆、8 個配方變數 → 抗壓強度）。

    從 OpenML 取得（data_id=4353），快取成 CSV。
    """
    import pandas as pd

    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, "concrete.csv")
    if os.path.exists(dst):
        df = pd.read_csv(dst)
    else:
        from sklearn.datasets import fetch_openml

        d = fetch_openml(data_id=4353, as_frame=True, parser="auto")
        df = d.frame.copy()
        df.columns = [c.strip() for c in df.columns]
        df.to_csv(dst, index=False)
    target = [c for c in df.columns if "strength" in c.lower()][0]
    feats = [c for c in df.columns if c != target]
    short = ["Cement", "Slag", "FlyAsh", "Water", "Superplast", "CoarseAgg",
             "FineAgg", "Age"]
    names = short if len(feats) == len(short) else feats
    return df[feats].to_numpy(float), df[target].to_numpy(float), names


def concrete_objective(data_dir: str, seed: int = 0) -> tuple[Objective, dict]:
    """把 Concrete 資料集變成一個可查詢的「真實世界」。

    做法：用**梯度提升迴歸**在全部 1030 筆上訓練一個代理模型，當作
    「做一次實驗就能得到的真實強度」。BO 的任務是用最少的查詢次數
    找到強度最高的配方。

    ⚠️ 這個設定的誠實面：代理模型不是真實世界，它平滑掉了真實製程的
    噪聲與不可行區域，所以「省下 N 次實驗」是在這個代理上的數字。
    但它保留了真實資料的維度、變數尺度與交互作用結構，
    比在解析測試函數上宣稱「材料應用」誠實得多。

    定義域取各變數在資料中的 [1%, 99%] 分位數（避免代理模型在
    資料稀疏的極端角落外插出假的高強度），再正規化到 [0,1]^8。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_score

    X, y, names = load_concrete(data_dir)
    model = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
    # ⚠️ 必須 shuffle。Concrete 的原始檔案按配方組別排序，sklearn 的
    # `cv=5` 預設**不** shuffle，於是每個 fold 拿到的是不同配方族群，
    # 實測 R² = 0.43 ± 0.67（標準差比均值還大，某些 fold 為負）。
    # 那個數字反映的是資料排序，不是模型能力，而代理模型的品質
    # 直接決定「省下多少次實驗」這個結論可不可信。
    cv = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=seed),
                         scoring="r2")
    model.fit(X, y)

    lo = np.percentile(X, 1, axis=0)
    hi = np.percentile(X, 99, axis=0)

    def f(u):
        x = lo + (hi - lo) * np.asarray(u)
        # 取負號：統一成最小化問題
        return -float(model.predict(x.reshape(1, -1))[0])

    obj = Objective("Concrete", X.shape[1], f, f_min=None)
    info = {"surrogate_cv_r2_mean": float(cv.mean()), "surrogate_cv_r2_std": float(cv.std()),
            "n_rows": int(len(X)), "features": names,
            "domain_lo": lo.tolist(), "domain_hi": hi.tolist(),
            "best_in_data": float(y.max()), "mean_in_data": float(y.mean())}
    return obj, info
