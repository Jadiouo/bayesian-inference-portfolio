"""
B2 · 三個競爭模型 —— M0 純雜訊 / M1 行星凌日 / M2 食雙星。

計劃書步驟 1。核心的建模判斷是：**M1 與 M2 的差別要小而物理**，否則比較沒有意義。
這裡的差別只有兩處，兩處都是真正的天文判準：

  1. **rp = R_c/R\* 的先驗範圍**。行星有物理半徑上限（簡併壓讓行星撐不過 ~2 R_Jup，
     即 rp ≲ 0.2）；食雙星的伴星可以跟主星差不多大（rp 到 1）。**先驗就是物理假設**
     —— 這正是貝氏因子與「用同一個模型硬擬合再比 χ²」的根本差異。
  2. **次食**。行星的次食是幾十 ppm 等級（反射+熱輻射），在這個精度下等於沒有；
     食雙星的伴星自己會發光，次食深度是主食的 J 倍（J = 表面亮度比）。
     M2 = M1 + 一個參數 J。**只多一個參數** → 奧卡姆懲罰乾淨可算。

所以 M2 相對 M1 的優勢必須來自資料真的偏好「大伴星 + 有次食」，
而不是來自「參數多所以擬合好」。這就是邊際似然在做的事。

參數順序（θ）：
  M0: [f0, log_jit]
  M1: [f0, rp, a, b, t0, q1, q2, log_jit]
  M2: [f0, rp, a, b, t0, q1, q2, J, log_jit]
"""
from __future__ import annotations

import batman
import numpy as np

KEPLER_LC_EXP = 0.0204340   # Kepler 長曝光積分時間（天）≈ 29.4 分鐘
LOG2PI = float(np.log(2 * np.pi))

# ── 共用先驗界限 ────────────────────────────────────────────────────────────
# 三個模型共用的參數要用**完全相同**的先驗，否則 logZ 的差會混進無關的先驗體積差。
F0_LO, F0_HI = 0.998, 1.002          # 基線
JIT_LO, JIT_HI = -7.0, -3.0          # log10(jitter)
A_LO, A_HI = 2.0, 60.0               # a/R*，log-uniform（尺度參數）
T0_HALF = 0.02                       # 中天相位微調 ±0.02
RP_FLOOR = 1e-4

RP_MAX_PLANET = 0.20                 # M1：行星半徑物理上限（≈2 R_Jup）
RP_MAX_BINARY = 1.00                 # M2：伴星可與主星同級


def kipping_to_u(q1, q2):
    """Kipping (2013) (q1,q2) → 二次臨邊昏暗 (u1,u2)，自動落在物理有效三角形內。"""
    sq = np.sqrt(q1)
    return 2 * sq * q2, sq * (1 - 2 * q2)


def _secondary_phase(phase):
    """把相位映射成「距次食中心的相位距離」。

    次食中心在 ±0.5。注意不能用 sign(phase)*(|phase|-0.5)：phase=0 時 sign=0 會算出 0，
    等於在主食中心也放一個次食。用 where 分段才對。
    """
    return np.where(phase >= 0, phase - 0.5, phase + 0.5)


class TransitShape:
    """預建 batman 模型（主食 + 次食兩組時間網格），MCMC/NS 迴圈內只更新參數。

    對 29.4 分鐘長曝光積分（supersample）——B1 踩過的坑：不積分的話擬合會用極端臨邊昏暗
    硬湊圓底，把 rp 系統性壓低。
    """

    def __init__(self, phase, P, exp_time=KEPLER_LC_EXP, supersample=11):
        self.P = P
        self.phase = np.asarray(phase, float)
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a = 0.0, P, 0.05, 10.0
        p.inc, p.ecc, p.w = 89.0, 0.0, 90.0
        p.u, p.limb_dark = [0.3, 0.2], "quadratic"
        self.params = p
        kw = dict(supersample_factor=supersample, exp_time=exp_time)
        self.m_pri = batman.TransitModel(p, self.phase * P, **kw)
        self.m_sec = batman.TransitModel(p, _secondary_phase(self.phase) * P, **kw)

    def _set(self, rp, a, b, t0, q1, q2):
        p = self.params
        p.rp, p.a, p.t0 = rp, a, t0 * self.P
        p.inc = np.degrees(np.arccos(np.clip(b / a, 0.0, 1.0)))
        p.u = list(kipping_to_u(q1, q2))
        return p

    def primary(self, rp, a, b, t0, q1, q2):
        return self.m_pri.light_curve(self._set(rp, a, b, t0, q1, q2))

    def with_secondary(self, rp, a, b, t0, q1, q2, J):
        """主食 + 次食。次食沿用同一組幾何（圓軌道 → 次食在相位 0.5、時長相同），
        深度乘上表面亮度比 J。這是唯象近似，但抓住了判別上真正重要的自由度。"""
        p = self._set(rp, a, b, t0, q1, q2)
        pri = self.m_pri.light_curve(p)
        sec = self.m_sec.light_curve(p)
        return pri - J * (1.0 - sec)


# ─────────────────────────────────────────────────────────────────────────────
# 模型：每個提供 param_names / prior_transform / loglike，直接餵給 ultranest
# ─────────────────────────────────────────────────────────────────────────────
class Model:
    """競爭模型的共同介面。

    `prior_transform` 把單位立方 [0,1]^d 映射到參數空間 —— 這是 nested sampling 的
    標準介面，也把「先驗是什麼」寫得毫不含糊（先驗體積 = 這個映射的 Jacobian）。
    """

    name = "M?"
    label = "?"
    param_names: list[str] = []

    def __init__(self, shape: TransitShape, flux, err):
        self.shape, self.flux, self.err2 = shape, np.asarray(flux, float), np.asarray(err, float) ** 2

    @property
    def ndim(self):
        return len(self.param_names)

    def _gauss_loglike(self, model, log_jit):
        s2 = self.err2 + (10.0 ** log_jit) ** 2
        return float(-0.5 * np.sum((self.flux - model) ** 2 / s2 + np.log(s2) + LOG2PI))


class M0Noise(Model):
    """M0 · 純雜訊：光曲線只是一條常數基線加白雜訊。沒有凌日。"""

    name, label = "M0", "純雜訊"
    param_names = ["f0", "log_jit"]

    def prior_transform(self, u):
        return np.array([F0_LO + u[0] * (F0_HI - F0_LO),
                         JIT_LO + u[1] * (JIT_HI - JIT_LO)])

    def loglike(self, θ):
        f0, log_jit = θ
        return self._gauss_loglike(np.full_like(self.flux, f0), log_jit)


class M1Planet(Model):
    """M1 · 行星凌日：對稱、平底、**沒有次食**，且 rp 受行星物理上限約束。"""

    name, label = "M1", "行星凌日"
    param_names = ["f0", "rp", "a", "b", "t0", "q1", "q2", "log_jit"]

    def __init__(self, shape, flux, err, rp_max=RP_MAX_PLANET):
        super().__init__(shape, flux, err)
        self.rp_max = rp_max          # 先驗敏感度分析（步驟 5）就是掃這個數

    def prior_transform(self, u):
        rp = RP_FLOOR + u[1] * (self.rp_max - RP_FLOOR)
        return np.array([
            F0_LO + u[0] * (F0_HI - F0_LO),
            rp,
            np.exp(np.log(A_LO) + u[2] * (np.log(A_HI) - np.log(A_LO))),   # log-uniform
            u[3] * (1.0 + rp),        # b ~ U(0, 1+rp)：均勻 b = 均勻 cos i（幾何先驗）
            (2 * u[4] - 1) * T0_HALF,
            u[5], u[6],               # Kipping q1, q2
            JIT_LO + u[7] * (JIT_HI - JIT_LO),
        ])

    def loglike(self, θ):
        f0, rp, a, b, t0, q1, q2, log_jit = θ
        m = f0 * self.shape.primary(rp, a, b, t0, q1, q2)
        if not np.all(np.isfinite(m)):
            return -1e100
        return self._gauss_loglike(m, log_jit)


class M2Binary(Model):
    """M2 · 食雙星：伴星可與主星同級（rp 到 1），且會發光 → 相位 0.5 有次食。

    相對 M1 只多一個參數 J（表面亮度比）。多出來的先驗體積就是奧卡姆懲罰的來源。
    """

    name, label = "M2", "食雙星"
    param_names = ["f0", "rp", "a", "b", "t0", "q1", "q2", "J", "log_jit"]

    def __init__(self, shape, flux, err, rp_max=RP_MAX_BINARY):
        super().__init__(shape, flux, err)
        self.rp_max = rp_max

    def prior_transform(self, u):
        rp = RP_FLOOR + u[1] * (self.rp_max - RP_FLOOR)
        return np.array([
            F0_LO + u[0] * (F0_HI - F0_LO),
            rp,
            np.exp(np.log(A_LO) + u[2] * (np.log(A_HI) - np.log(A_LO))),
            u[3] * (1.0 + rp),
            (2 * u[4] - 1) * T0_HALF,
            u[5], u[6],
            u[7],                     # J ~ U(0,1)：表面亮度比
            JIT_LO + u[8] * (JIT_HI - JIT_LO),
        ])

    def loglike(self, θ):
        f0, rp, a, b, t0, q1, q2, J, log_jit = θ
        m = f0 * self.shape.with_secondary(rp, a, b, t0, q1, q2, J)
        if not np.all(np.isfinite(m)):
            return -1e100
        return self._gauss_loglike(m, log_jit)


def build_all(shape, flux, err):
    """回傳三個模型（順序 M0, M1, M2）。"""
    return [M0Noise(shape, flux, err), M1Planet(shape, flux, err), M2Binary(shape, flux, err)]
