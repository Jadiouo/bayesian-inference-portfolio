"""
B1 · 物理模型 —— batman 凌日模型 + Kipping (2013) 臨邊昏暗參數化。

待估參數（在摺疊/固定週期下）：
  rp = R_p/R*   （半徑比，即深度）      —— 正數
  a  = a/R*     （縮放半長軸）
  b  = 撞擊參數  （0..1+rp，均勻 ≈ cos i 均勻，幾何先驗）→ i = arccos(b/a)
  t0 = 中天時刻微調
  q1, q2 = Kipping 臨邊昏暗（各 ∈[0,1]，映射到物理上有效的 u1,u2）

Kipping (2013)：u1 = 2√q1·q2，u2 = √q1·(1−2q2)。這個重參數化把二次臨邊昏暗
限制在物理上有效的三角形內，且對 (q1,q2) 是均勻先驗——比直接對 (u1,u2) 設先驗正確。
"""
from __future__ import annotations

import batman
import numpy as np


def kipping_to_u(q1, q2):
    """Kipping (2013) (q1,q2) → 二次臨邊昏暗 (u1,u2)。"""
    sq = np.sqrt(q1)
    return 2 * sq * q2, sq * (1 - 2 * q2)


KEPLER_LC_EXP = 0.0204340   # Kepler 長曝光積分時間（天）≈ 29.4 分鐘


class TransitEvaluator:
    """預建 batman 模型，之後只更新參數 → MCMC 迴圈內快速求值。
    **對長曝光積分**（supersample + exp_time）：Kepler 長曝光 29.4 分鐘會抹平凌日邊緣，
    不積分的話擬合會用極端臨邊昏暗硬湊圓底、把 Rp/R* 偏低。"""

    def __init__(self, phase, P, exp_time=KEPLER_LC_EXP, supersample=11):
        self.P = P
        self.times = phase * P                    # 相位 → 距中天的天數
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a = 0.0, P, 0.0125, 3.5
        p.inc, p.ecc, p.w = 89.0, 0.0, 90.0
        p.u, p.limb_dark = [0.4, 0.2], "quadratic"
        self.params = p
        self.model = batman.TransitModel(p, self.times,
                                         supersample_factor=supersample, exp_time=exp_time)

    def __call__(self, rp, a, b, t0, q1, q2):
        p = self.params
        p.t0, p.rp, p.a = t0, rp, a
        p.inc = np.degrees(np.arccos(np.clip(b / a, 0.0, 1.0)))
        p.u = list(kipping_to_u(q1, q2))
        return self.model.light_curve(p)

    def inclination(self, a, b):
        return float(np.degrees(np.arccos(np.clip(b / a, 0.0, 1.0))))
