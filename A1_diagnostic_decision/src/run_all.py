"""
A1 · 一鍵重現：擬合貝葉斯邏輯迴歸，走完決策 / 校準 / 敏感度，生成 figures/ 六張圖。
執行：conda activate bayes && python src/run_all.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bayes_logreg as blr
import calibration as cal
import data
import decision as dec
import plots

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "..", "data", "A_medical")
os.makedirs(FIG, exist_ok=True)


def sec(t):
    print("\n" + "═" * 70 + f"\n{t}\n" + "═" * 70)


def main():
    R = {}

    # ─────────── 資料 + 主模型 ───────────
    sec("0 · 資料與貝葉斯邏輯迴歸")
    ds = data.load_heart(DATA, test_size=0.25, seed=0)
    idata, _ = blr.fit(ds.X_train, ds.y_train, prior_sd=2.5, seed=42)
    conv = blr.convergence(idata)
    p_pred = blr.predictive_mean(idata, ds.X_test)
    p_plug = blr.plugin_proba(idata, ds.X_test)
    lr = LogisticRegression(max_iter=2000).fit(ds.X_train, ds.y_train)
    auc_b = roc_auc_score(ds.y_test, p_pred)
    auc_f = roc_auc_score(ds.y_test, lr.predict_proba(ds.X_test)[:, 1])
    print(f"特徵={ds.n_features} train={len(ds.y_train)} test={len(ds.y_test)} "
          f"陽性率={ds.y_train.mean():.1%}")
    print(f"收斂 max_rhat={conv['max_rhat']:.4f} min_ess={conv['min_ess_bulk']:.0f}")
    print(f"AUC 貝葉斯={auc_b:.3f} · 頻率派={auc_f:.3f}")
    R.update(n_feat=ds.n_features, n_test=len(ds.y_test), max_rhat=conv["max_rhat"],
             auc_b=round(auc_b, 3), auc_f=round(auc_f, 3))

    # ─────────── 圖 1：後驗係數 ───────────
    beta = idata.posterior["beta"].values.reshape(-1, ds.n_features)
    b_mean = beta.mean(0)
    b_lo, b_hi = np.percentile(beta, [2.5, 97.5], axis=0)
    plots.coefficients(ds.features, b_mean, b_lo, b_hi,
                       os.path.join(FIG, "01_coefficients.png"))

    # ─────────── 圖 2：後驗預測 vs plug-in ───────────
    sec("2 · 後驗預測分佈 vs plug-in")
    diff = np.abs(p_plug - p_pred)
    P_all = blr.posterior_predictive_proba(idata, ds.X_test)   # (S, N)
    ex = np.argsort(p_pred)[np.linspace(0, len(p_pred) - 1, 7).astype(int)]
    lo, hi = blr.predictive_interval(idata, ds.X_test)
    print(f"|plug-in − 後驗預測| 平均={diff.mean():.4f} 最大={diff.max():.4f}")
    print(f"後驗預測 95%CI 平均寬度={np.mean(hi - lo):.3f}（plug-in 寬度=0）")
    i = int(diff.argmax())
    print(f"最大差異病人：plug-in={p_plug[i]:.3f} 後驗預測={p_pred[i]:.3f} "
          f"CI=[{lo[i]:.3f},{hi[i]:.3f}]")
    plots.predictive_vs_plugin(p_plug, p_pred, P_all[:, ex], p_plug[ex], p_pred[ex],
                               os.path.join(FIG, "02_predictive_vs_plugin.png"))
    R.update(diff_mean=round(diff.mean(), 4), diff_max=round(diff.max(), 4),
             ci_width=round(float(np.mean(hi - lo)), 3))

    # ─────────── 圖 3：最優門檻 ───────────
    sec("3 · 損失矩陣與最優門檻")
    ratios = [1, 5, 20, 100]
    p_stars = [dec.optimal_threshold(1, r) for r in ratios]
    for r, ps in zip(ratios, p_stars):
        print(f"C_FN:C_FP = {r:>3}:1 → 最優門檻 p* = {ps:.4f}")
    ratio_curve = 20
    thr = np.linspace(0.005, 0.8, 300)
    loss_curve = [dec.realized_loss(ds.y_test, p_pred, 1, ratio_curve, t) for t in thr]
    ps_c = dec.optimal_threshold(1, ratio_curve)
    loss_half = dec.realized_loss(ds.y_test, p_pred, 1, ratio_curve, 0.5)
    loss_star = dec.realized_loss(ds.y_test, p_pred, 1, ratio_curve, ps_c)
    print(f"\nC_FN:C_FP={ratio_curve}:1 測試集實際損失：門檻0.5={loss_half:.2f} vs 最優p*={ps_c:.3f}→{loss_star:.2f}"
          f"（降低 {(1-loss_star/loss_half):.0%}）")
    plots.optimal_threshold(ratios, p_stars, thr, loss_curve, ratio_curve, ps_c,
                            loss_half, loss_star, os.path.join(FIG, "03_optimal_threshold.png"))
    R.update(p_star_100=round(p_stars[-1], 4), loss_half=round(loss_half, 2),
             loss_star=round(loss_star, 2), ratio_curve=ratio_curve,
             loss_drop=round(1 - loss_star / loss_half, 3))

    # ─────────── 圖 4：棄權選項 ───────────
    sec("4 · 棄權選項（轉診 / 再檢查，示意用 5:1）")
    c_fp, c_fn = 1, 5
    pg = np.linspace(0, 1, 300)
    treat_loss = dec.expected_loss_treat(pg, c_fp)
    notreat_loss = dec.expected_loss_notreat(pg, c_fn)
    regions = []
    for c_rej in [0.6, 0.4]:
        reg = dec.reject_region(c_fp, c_fn, c_rej)
        _, frac = dec.realized_loss_with_reject(ds.y_test, p_pred, c_fp, c_fn, c_rej)
        regions.append((c_rej, reg, frac))
        print(f"C_reject={c_rej}: 棄權區 p∈[{reg[0]:.3f},{reg[1]:.3f}]（寬 {reg[1]-reg[0]:.3f}），"
              f"測試集 {frac:.0%} 病人落入棄權區")
    plots.reject_option(pg, treat_loss, notreat_loss, regions, c_fn,
                        dec.optimal_threshold(c_fp, c_fn),
                        os.path.join(FIG, "04_reject_option.png"))
    R.update(reject_frac_06=round(regions[0][2], 3), reject_frac_04=round(regions[1][2], 3))

    # ─────────── 圖 5：校準（5-fold OOF）───────────
    sec("5 · 校準分析（5-fold 交叉驗證 OOF）")
    Xr, yr, cont_idx, _ = data.load_raw(DATA)
    skf = StratifiedKFold(5, shuffle=True, random_state=1)
    oof_b = np.zeros(len(yr))
    oof_f = np.zeros(len(yr))
    for k, (tr, te) in enumerate(skf.split(Xr, yr)):
        sc = StandardScaler().fit(Xr[tr][:, cont_idx])
        Xtr, Xte = Xr[tr].copy(), Xr[te].copy()
        Xtr[:, cont_idx] = sc.transform(Xtr[:, cont_idx])
        Xte[:, cont_idx] = sc.transform(Xte[:, cont_idx])
        id_k, _ = blr.fit(Xtr, yr[tr], prior_sd=2.5, draws=800, tune=800, chains=2, seed=100 + k)
        oof_b[te] = blr.predictive_mean(id_k, Xte)
        oof_f[te] = LogisticRegression(max_iter=2000).fit(Xtr, yr[tr]).predict_proba(Xte)[:, 1]
    ece_b, ece_f = cal.ece(yr, oof_b, 10), cal.ece(yr, oof_f, 10)
    conf_b, acc_b, _ = cal.reliability(yr, oof_b, 10)
    conf_f, acc_f, _ = cal.reliability(yr, oof_f, 10)
    print(f"OOF AUC 貝葉斯={roc_auc_score(yr,oof_b):.3f} 頻率派={roc_auc_score(yr,oof_f):.3f}")
    print(f"ECE 貝葉斯={ece_b:.3f} · 頻率派={ece_f:.3f}")
    plots.calibration(conf_b, acc_b, ece_b, conf_f, acc_f, ece_f, oof_b,
                      os.path.join(FIG, "05_calibration.png"))
    R.update(ece_b=round(ece_b, 3), ece_f=round(ece_f, 3))

    # ─────────── 圖 6：先驗敏感度 ───────────
    sec("6 · 先驗敏感度（N(0,1) / N(0,2.5) / N(0,10)）")
    top = np.argsort(np.abs(b_mean))[::-1][:6]
    top_names = [ds.features[i] for i in top]
    prior_sds = [1.0, 2.5, 10.0]
    means, los, his, aucs = [], [], [], []
    for psd in prior_sds:
        idp, _ = blr.fit(ds.X_train, ds.y_train, prior_sd=psd, draws=1000, tune=1000,
                         chains=2, seed=7)
        bp = idp.posterior["beta"].values.reshape(-1, ds.n_features)
        means.append(bp[:, top].mean(0))
        lo_p, hi_p = np.percentile(bp[:, top], [2.5, 97.5], axis=0)
        los.append(lo_p); his.append(hi_p)
        aucs.append(roc_auc_score(ds.y_test, blr.predictive_mean(idp, ds.X_test)))
        print(f"prior_sd={psd:>4}: 測試 AUC={aucs[-1]:.3f}")
    plots.prior_sensitivity(top_names, means, los, his,
                            [f"N(0,{s:g})" for s in prior_sds], aucs,
                            os.path.join(FIG, "06_prior_sensitivity.png"))
    R.update(auc_prior1=round(aucs[0], 3), auc_prior10=round(aucs[2], 3))

    # ─────────── README 數字 ───────────
    sec("README 可用數字")
    for k, v in R.items():
        print(f"  {k:>16} = {v}")
    print(f"\n六張圖已輸出至：{os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
