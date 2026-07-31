"""
C1 · 一鍵重現：三個模型、收縮圖、超參數、預測驗證、funnel，生成 figures/ 五張圖。
執行：conda activate bayes && python src/run_all.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data
import models
import plots
import shrinkage as shr

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)


def sec(t):
    print("\n" + "═" * 70 + f"\n{t}\n" + "═" * 70)


def county_ci(idata, J):
    a = idata.posterior["a"]
    if a.ndim == 2:                                   # complete pooling
        s = a.values.reshape(-1)
        return (np.full(J, s.mean()), np.full(J, np.percentile(s, 2.5)),
                np.full(J, np.percentile(s, 97.5)))
    s = a.values.reshape(-1, J)
    return s.mean(0), np.percentile(s, 2.5, 0), np.percentile(s, 97.5, 0)


def main():
    R = {}
    rd = data.load_radon()
    print(f"Radon：{rd.N} 戶、{rd.J} 郡；郡樣本數 {rd.n_j.min()}–{rd.n_j.max()}（中位數 {int(np.median(rd.n_j))}）")

    # ─────────── 三個模型（全資料）───────────
    sec("1 · 三個模型：no pooling / complete pooling / hierarchical")
    id_np = models.fit_no_pooling(rd.cidx, rd.floor, rd.y, rd.J)
    id_cp = models.fit_complete_pooling(rd.cidx, rd.floor, rd.y, rd.J)
    id_h = models.fit_hierarchical(rd.cidx, rd.floor, rd.y, rd.J)
    import arviz as az
    div_h = int(id_h.sample_stats["diverging"].sum())
    print(f"階層模型（非中心化）：divergences={div_h}，"
          f"max_rhat={az.summary(id_h, var_names=['mu_a','sigma_a','b'])['r_hat'].max():.3f}")

    a_np = models.county_intercepts(id_np, rd.J)
    a_pp = models.county_intercepts(id_h, rd.J)
    a_cp = models.county_intercepts(id_cp, rd.J)[0]
    sigma = float(id_h.posterior["sigma"].mean())
    sigma_a = float(id_h.posterior["sigma_a"].mean())

    # ─────────── 圖 1：三種做法（示例郡）───────────
    npm, npl, nph = county_ci(id_np, rd.J)
    ppm, ppl, pph = county_ci(id_h, rd.J)
    ex = []
    for t in [1, 2, 5, 12, 30, 70, rd.n_j.max()]:
        j = int(np.argmin(np.abs(rd.n_j - t)))
        if j not in ex:
            ex.append(j)
    ex = np.array(ex)[np.argsort(rd.n_j[np.array(ex)])]
    plots.three_approaches(rd.counties[ex], rd.n_j[ex], npm[ex], npl[ex], nph[ex],
                           ppm[ex], ppl[ex], pph[ex], a_cp,
                           os.path.join(FIG, "01_three_approaches.png"))

    # ─────────── 圖 2：收縮 + 收縮權重 ───────────
    sec("2 · 收縮圖（本專案靈魂）")
    omega = shr.empirical_weight(a_np, a_pp, a_cp)
    mask = np.abs(a_np - a_cp) > 0.1                  # 過濾 denom 太小的不穩點
    n_grid = np.logspace(0, np.log10(rd.n_j.max()), 120)
    omega_th = shr.theoretical_weight(n_grid, sigma, sigma_a)
    small = rd.n_j <= 3
    big = rd.n_j >= 30
    print(f"樣本數≤3 的郡平均收縮權重={np.nanmean(omega[small & mask]):.2f}；"
          f"樣本數≥30 的郡={np.nanmean(omega[big & mask]):.2f}（拉力隨 n 遞減）")
    plots.shrinkage(rd.n_j, a_np, a_pp, a_cp, omega, n_grid, omega_th,
                    os.path.join(FIG, "02_shrinkage.png"))
    R.update(shrink_small=round(float(np.nanmean(omega[small & mask])), 2),
             shrink_big=round(float(np.nanmean(omega[big & mask])), 2))

    # ─────────── 圖 3：超參數（郡間差異）───────────
    sec("3 · 超參數後驗：郡與郡之間差異有多大？")
    mu_a_s = id_h.posterior["mu_a"].values.reshape(-1)
    sig_a_s = id_h.posterior["sigma_a"].values.reshape(-1)
    lo, med, hi = np.percentile(sig_a_s, [2.5, 50, 97.5])
    print(f"σ_a（郡間 SD）後驗：中位數={med:.3f}，95% CI=[{lo:.3f}, {hi:.3f}]"
          f" → 明確 >0，完全匯聚（假設 σ_a=0）無法回答這個問題")
    plots.hyperparameters(mu_a_s, sig_a_s, os.path.join(FIG, "03_hyperparameters.png"))
    R.update(sigma_a_med=round(med, 3), sigma_a_lo=round(lo, 3), sigma_a_hi=round(hi, 3))

    # ─────────── 圖 4：預測驗證（留出測試集，10 次切分平均）───────────
    sec("4 · 預測驗證：三個模型的測試誤差（10 次隨機切分平均）")
    buckets = ["n≤5", "6–20", ">20"]
    order = ["no pooling", "complete pooling", "hierarchical"]

    def bucket_of(n):
        return "n≤5" if n <= 5 else ("6–20" if n <= 20 else ">20")

    ov = {k: [] for k in order}
    bk = {k: {b: [] for b in buckets} for k in order}
    for sd in range(10):
        tr, te = data.train_test_split(rd, test_frac=0.25, seed=sd)
        te_bkt = np.array([bucket_of(rd.n_j[c]) for c in rd.cidx[te]])
        fits = {
            "no pooling": models.fit_no_pooling(rd.cidx[tr], rd.floor[tr], rd.y[tr], rd.J, draws=500, tune=500, chains=2),
            "complete pooling": models.fit_complete_pooling(rd.cidx[tr], rd.floor[tr], rd.y[tr], rd.J, draws=500, tune=500, chains=2),
            "hierarchical": models.fit_hierarchical(rd.cidx[tr], rd.floor[tr], rd.y[tr], rd.J, draws=500, tune=500, chains=2),
        }
        for name, idata in fits.items():
            pred = models.predict(idata, rd.cidx[te], rd.floor[te], rd.J)
            ov[name].append(shr.rmse(pred, rd.y[te]))
            for b in buckets:
                m = te_bkt == b
                if m.any():
                    bk[name][b].append(shr.rmse(pred[m], rd.y[te][m]))
    rmse_all = [float(np.mean(ov[k])) for k in order]
    rmse_bucket = {k: {b: float(np.mean(bk[k][b])) for b in buckets} for k in order}
    for name, r in zip(order, rmse_all):
        print(f"  {name:<18} 測試 RMSE={r:.3f}  |  "
              + "  ".join(f"{b}:{rmse_bucket[name][b]:.3f}" for b in buckets))
    plots.predictive(order, rmse_all, buckets, rmse_bucket,
                     os.path.join(FIG, "04_predictive.png"))
    R.update(rmse_nopool=round(rmse_all[0], 3), rmse_complete=round(rmse_all[1], 3),
             rmse_hier=round(rmse_all[2], 3),
             rmse_hier_small=round(rmse_bucket["hierarchical"]["n≤5"], 3),
             rmse_nopool_small=round(rmse_bucket["no pooling"]["n≤5"], 3),
             rmse_complete_small=round(rmse_bucket["complete pooling"]["n≤5"], 3))

    # ─────────── 圖 5：funnel（Eight Schools）───────────
    sec("5 · funnel 與非中心化（Eight Schools）")
    y8, s8, _ = data.eight_schools()
    id_c = models.fit_eight_schools(y8, s8, centered=True)
    id_nc = models.fit_eight_schools(y8, s8, centered=False)
    ndc = int(id_c.sample_stats["diverging"].sum())
    ndnc = int(id_nc.sample_stats["diverging"].sum())
    print(f"中心化 divergences={ndc}；非中心化 divergences={ndnc} → 重參數化解決 funnel")

    def funnel_coords(idata):
        th = idata.posterior["theta"].values[..., 0].reshape(-1)
        lt = np.log(idata.posterior["tau"].values.reshape(-1))
        dv = idata.sample_stats["diverging"].values.reshape(-1)
        return th, lt, dv
    tc, lc, dc = funnel_coords(id_c)
    tn, ln, dn = funnel_coords(id_nc)
    plots.funnel(tc, lc, dc, tn, ln, dn, ndc, ndnc, os.path.join(FIG, "05_funnel.png"))
    R.update(div_centered=ndc, div_noncentered=ndnc, div_h_radon=div_h)

    # ─────────── README 數字 ───────────
    sec("README 可用數字")
    for k, v in R.items():
        print(f"  {k:>18} = {v}")
    print(f"\n五張圖已輸出至：{os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
