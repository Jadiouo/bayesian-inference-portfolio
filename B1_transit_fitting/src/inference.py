"""
B1 · 貝葉斯推論 —— emcee（天文標準）。

batman 是純 numpy 黑箱函式，用 emcee（只需 log-prob）比包進 PyMC 自訂 Op 乾淨。
參數順序：theta = [rp, a, b, t0, q1, q2, f0]。

先驗（計劃書步驟 4，這裡要動腦）：
  rp ∈ (0, 0.05)      正數（半徑比），均勻
  a  ∈ (1.5, 8)       均勻（刻意不用恆星密度強先驗，讓 rp–a–b 簡併現形）
  b  ∈ (0, 1+rp)      均勻 = cos i 均勻（幾何先驗，不是角度均勻 ← 參數化不變性）
  t0 ∈ (−0.01, 0.01)  中天微調
  q1,q2 ∈ (0, 1)      Kipping 臨邊昏暗
  f0 ∈ (0.998, 1.002) 基線
"""
from __future__ import annotations

import arviz as az
import numpy as np

LABELS = ["rp", "a", "b", "t0", "q1", "q2", "f0", "log_jit"]
#              rp    a    b     t0   q1   q2   f0    log_jit（jitter，ppm 尺度的 log10）
LO = np.array([1e-4, 1.5, 0.0, -0.01, 0.0, 0.0, 0.998, -7.0])
HI = np.array([0.05, 8.0, 1.5, 0.01, 1.0, 1.0, 1.002, -3.0])

# 臨邊昏暗的物理先驗：Kepler-10（Teff≈5627, logg≈4.35, Kepler 波段）的理論二次係數
# u1≈0.38, u2≈0.28 → Kipping q1≈0.44, q2≈0.29。淺凌日「幾乎不約束」LD 且與凌日形狀簡併，
# 故用恆星模型理論值當先驗（Kepler-10 有星震學，恆星參數可靠）；sd=0.10 保留 LD 表的不確定性。
Q1_MU, Q1_SD = 0.44, 0.10
Q2_MU, Q2_SD = 0.29, 0.10


def log_prior(theta):
    rp = theta[0]
    hi = HI.copy()
    hi[2] = 1.0 + rp                         # b < 1+rp（幾何界限）
    if np.any(theta <= LO) or np.any(theta >= hi):
        return -np.inf
    q1, q2 = theta[4], theta[5]              # LD：高斯先驗（其餘均勻，b 均勻→cos i 均勻）
    return -0.5 * ((q1 - Q1_MU) / Q1_SD) ** 2 - 0.5 * ((q2 - Q2_MU) / Q2_SD) ** 2


def log_prob(theta, evaluator, flux, err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    rp, a, b, t0, q1, q2, f0, log_jit = theta
    model = f0 * evaluator(rp, a, b, t0, q1, q2)
    if not np.all(np.isfinite(model)):
        return -np.inf
    s2 = err ** 2 + (10.0 ** log_jit) ** 2   # jitter 加在誤差的平方上（吸收低估的相關雜訊）
    return lp - 0.5 * np.sum((flux - model) ** 2 / s2 + np.log(s2))


def run_emcee(evaluator, flux, err, nwalkers=64, nburn=6000, nprod=30000, seed=42):
    """跑 emcee，回傳 (sampler, idata, tau)。用 DEMove 應付 rp–a–b 強簡併。"""
    import emcee

    ndim = len(LABELS)
    rng = np.random.default_rng(seed)
    guess = np.array([0.0125, 3.5, 0.30, 0.0, 0.44, 0.30, 1.0, -4.5])
    p0 = guess + rng.normal(0, 1, (nwalkers, ndim)) * np.array(
        [0.002, 0.4, 0.1, 0.001, 0.1, 0.1, 2e-4, 0.3])
    p0 = np.clip(p0, LO + 1e-6, HI - 1e-6)

    # DEMove（差分演化）對相關/香蕉形後驗混合遠優於預設 StretchMove
    moves = [(emcee.moves.DEMove(), 0.7), (emcee.moves.DESnookerMove(), 0.3)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(evaluator, flux, err),
                                    moves=moves)
    state = sampler.run_mcmc(p0, nburn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, nprod, progress=False)

    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau = np.full(ndim, np.nan)

    idata = az.from_emcee(sampler, var_names=LABELS)
    return sampler, idata, tau


def flat_samples(sampler, discard=0, thin=1):
    return sampler.get_chain(discard=discard, thin=thin, flat=True)
