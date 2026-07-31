"""
B1 · 一鍵重現：下載/處理 Kepler-10 → 摺疊 → emcee 擬合凌日 → 五張圖。
執行：conda activate bayes && python src/run_all.py
（首次會下載光曲線並快取到 data/B_astro/；之後直接讀快取。）
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arviz as az

import data
import inference as inf
import plots
import transit_model as tm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "..", "data", "B_astro")
os.makedirs(FIG, exist_ok=True)

R_STAR_SUN = 1.065      # Kepler-10 恆星半徑 (太陽半徑)
RSUN_REARTH = 109.1


def sec(t):
    print("\n" + "═" * 70 + f"\n{t}\n" + "═" * 70)


def main():
    Rd = {}
    sec("1 · 資料：下載、去趨勢、相位摺疊")
    d = data.prepare(DATA)
    P, t0 = float(d["P"]), float(d["t0"])
    bp, bf, be, bn = data.bin_fold(d["fold_phase"], d["fold_flux"], window=0.06, n_bins=80)
    print(f"總點數={int(d['n_points'])}  BLS 週期={P:.6f} d（已發表 {data.published_kepler10b()['P']:.6f}）")
    print(f"摺疊窗 {len(d['fold_phase'])} 點 → 分箱 {len(bp)} 箱")
    plots.raw_and_flatten(d["raw_seg_t"], d["raw_seg_raw"], d["raw_seg_flat"],
                          os.path.join(FIG, "01_detrend.png"))

    sec("2 · emcee 擬合凌日（DEMove + 長曝光積分）")
    ev = tm.TransitEvaluator(bp, P)
    sampler, idata, tau = inf.run_emcee(ev, bf, be, seed=42)
    fs = inf.flat_samples(sampler)
    summ = az.summary(idata, var_names=inf.LABELS)
    print(summ[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]].to_string())
    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"\n接受率={acc:.2f}  最大自相關時間={np.nanmax(tau):.0f}  "
          f"鏈長/自相關={30000/np.nanmax(tau):.0f}（>50 收斂）  最大 r_hat={summ['r_hat'].max():.3f}")
    Rd.update(max_rhat=round(float(summ["r_hat"].max()), 3),
              min_ess=int(summ["ess_bulk"].min()), accept=round(acc, 2),
              tau=int(np.nanmax(tau)))

    # 導出量
    rp, a, b = fs[:, 0], fs[:, 1], fs[:, 2]
    inc = np.degrees(np.arccos(np.clip(b / a, 0, 1)))
    rp_earth = rp * R_STAR_SUN * RSUN_REARTH
    med = np.median(fs, axis=0)
    pub = data.published_kepler10b()

    # 圖 2：相位摺疊 + 最佳模型
    fine_ph = np.linspace(-0.06, 0.06, 400)
    ev_fine = tm.TransitEvaluator(fine_ph, P)
    best_fine = med[6] * ev_fine(med[0], med[1], med[2], med[3], med[4], med[5])
    depth_ppm = (1 - best_fine.min()) * 1e6
    plots.phase_fold(d["fold_phase"], d["fold_flux"], bp, bf, be, fine_ph, best_fine,
                     depth_ppm, P, os.path.join(FIG, "02_phase_fold.png"))

    # 圖 3：corner（簡併）
    sec("3 · 參數簡併（corner）")
    corner_samples = np.column_stack([rp, a, b, inc])
    truths = [pub["rp_rs"], pub["a_rs"], pub["b"],
              np.degrees(np.arccos(pub["b"] / pub["a_rs"]))]
    plots.corner_plot(corner_samples, ["R_p/R*", "a/R*", "b", "i (deg)"], truths,
                      os.path.join(FIG, "03_corner.png"))
    print(f"corr(rp,a)={np.corrcoef(rp, a)[0,1]:+.2f}  corr(rp,b)={np.corrcoef(rp, b)[0,1]:+.2f}  "
          f"corr(a,b)={np.corrcoef(a, b)[0,1]:+.2f} → 強簡併（Mean-Field VI 會失敗）")
    Rd.update(corr_rp_a=round(float(np.corrcoef(rp, a)[0, 1]), 2))

    # 圖 4：後驗預測檢查
    sec("4 · 後驗預測檢查")
    idx = np.random.default_rng(0).choice(len(fs), 150, replace=False)
    draws = np.array([fs[i, 6] * ev_fine(fs[i, 0], fs[i, 1], fs[i, 2], fs[i, 3], fs[i, 4], fs[i, 5])
                      for i in idx])
    resid = bf - med[6] * ev(med[0], med[1], med[2], med[3], med[4], med[5])
    chi2_red = np.sum((resid / be) ** 2) / (len(bp) - 8)
    print(f"約化 χ²={chi2_red:.2f}（≈1 表示模型與資料一致）；殘差 std={np.std(resid)*1e6:.0f} ppm")
    plots.posterior_predictive(bp, bf, be, fine_ph, draws, best_fine, P, resid,
                               os.path.join(FIG, "04_posterior_predictive.png"))
    Rd.update(chi2_red=round(float(chi2_red), 2))

    # 圖 5：與已發表值對照
    sec("5 · 與已發表值對照")
    lo, hi = np.percentile(rp, [2.5, 97.5])
    lo_e, hi_e = np.percentile(rp_earth, [2.5, 97.5])
    lo_i, hi_i = np.percentile(inc, [2.5, 97.5])
    print(f"Rp/R* = {np.median(rp):.4f} [{lo:.4f}, {hi:.4f}]   已發表 {pub['rp_rs']}±0.0004"
          f" → {'誤差內一致✓' if hi >= pub['rp_rs']-0.0004 and lo <= pub['rp_rs']+0.0004 else '✗'}")
    print(f"Rp    = {np.median(rp_earth):.2f} R⊕ [{lo_e:.2f}, {hi_e:.2f}]   已發表 {pub['rp_earth']}"
          f" → {'涵蓋✓' if lo_e <= pub['rp_earth'] <= hi_e else '✗'}")
    print(f"傾角 i = {np.median(inc):.1f}° [{lo_i:.1f}, {hi_i:.1f}]")
    plots.published_comparison(rp, rp_earth, pub, os.path.join(FIG, "05_published_comparison.png"))
    Rd.update(rp_med=round(float(np.median(rp)), 4), rp_lo=round(float(lo), 4), rp_hi=round(float(hi), 4),
              rpe_med=round(float(np.median(rp_earth)), 2), rpe_lo=round(float(lo_e), 2),
              rpe_hi=round(float(hi_e), 2), inc_med=round(float(np.median(inc)), 1),
              depth_ppm=round(float(depth_ppm)), P=round(P, 6))

    sec("README 可用數字")
    for k, v in Rd.items():
        print(f"  {k:>12} = {v}")
    print(f"\n五張圖已輸出至：{os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
