"""
D1 · 一鍵重現：算出所有數字並生成 figures/ 內四張圖。
執行：conda activate bayes && python src/run_all.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bayes_ab as ab
import experiments as ex
import plots
from bayes_ab import Prior
from frequentist import two_proportion_z_test

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)


def sec(title):
    print("\n" + "═" * 70 + f"\n{title}\n" + "═" * 70)


def main():
    rng = np.random.default_rng(42)
    readme = {}

    # ─────────── 1 · 回答老闆的問題 ───────────
    sec("1 · 回答老闆的問題（Beta-Binomial 封閉解）")
    kA, nA, kB, nB = 208, 4000, 232, 4000            # 5.2% vs 5.8%
    s = ab.summarize(kA, nA, kB, nB, Prior(1, 1), rng)
    print(s.boss_answer())

    aA, bA = s.postA
    aB, bB = s.postB
    p_grid = s.prob_b_beats_a
    p_mc = ab.prob_b_beats_a_mc(aA, bA, aB, bB, rng)
    p_norm = ab.prob_b_beats_a_normal(aA, bA, aB, bB)
    z, p_freq = two_proportion_z_test(kA, nA, kB, nB)
    print(f"\n交叉驗證 P(B>A): 網格={p_grid:.4f} · MC={p_mc:.4f} · 常態近似={p_norm:.4f}")
    print(f"頻率派雙尾 p-value = {p_freq:.4f}  (z={z:.3f})  ← 注意它回答的不是同一個問題")

    lift = rng.beta(aB, bB, 200_000) / rng.beta(aA, bA, 200_000) - 1
    plots.posteriors_and_lift(s, lift, os.path.join(FIG, "01_posteriors_and_lift.png"))
    readme.update(prob_ba=p_grid, p_freq=p_freq, lift_lo=s.lift_lo, lift_hi=s.lift_hi,
                  lift_med=s.lift_med, el_b=s.expected_loss_choose_b, p_norm=p_norm, p_mc=p_mc)

    # ─────────── 2 · 偷看資料（A/A 假陽性）───────────
    sec("2 · 序貫偷看的假陽性率（1000+ 次 A/A 模擬）")
    days, per_day, n_sims = 28, 500, 3000
    aa = ex.Scenario(pA=0.05, pB=0.05, per_day=per_day, days=days, name="A/A")
    kA_c, kB_c = ex.simulate_cumulative(aa, n_sims, rng)
    table = ex.build_peek_table(kA_c, kB_c, per_day, Prior(1, 1))

    # 主打圖只比較「宣稱有差異」的規則（p 值 vs 後驗門檻），才是 apples-to-apples 的
    # 偷看問題。期望損失是另一種哲學（有界後悔），另外報告、以正確框架解讀。
    rules_sig = [
        ex.RuleSpec("freq", 0.05, "Frequentist p<0.05"),
        ex.RuleSpec("bayes", 0.95, "Bayes P>0.95"),
        ex.RuleSpec("bayes", 0.975, "Bayes P>0.975"),
    ]
    looks_grid = [1, 2, 4, 7, 14, 28]
    fpr = ex.fpr_vs_looks(table, days, looks_grid, rules_sig)
    print(f"每組每天 {per_day} 曝光、追蹤 {days} 天、{n_sims} 次 A/A 模擬")
    print(f"{'偷看次數':>8} | " + " | ".join(f"{r.label:>18}" for r in rules_sig))
    for i, K in enumerate(looks_grid):
        print(f"{K:>8} | " + " | ".join(f"{fpr['rates'][r.label][i]*100:>17.1f}%" for r in rules_sig))

    daily = {r.label: fpr["rates"][r.label][-1] for r in rules_sig}   # K=28（逐日偷看）
    fixed = {r.label: fpr["rates"][r.label][0] for r in rules_sig}    # K=1（只看最後一天）
    print(f"\n只看 1 次（固定視野）: freq={fixed['Frequentist p<0.05']:.1%} · "
          f"bayes0.975={fixed['Bayes P>0.975']:.1%}  ← 匹配門檻，皆 ≈5%")
    print(f"逐日偷看 28 次       : freq={daily['Frequentist p<0.05']:.1%} · "
          f"bayes0.975={daily['Bayes P>0.975']:.1%}  ← 仍幾乎一模一樣 → 貝葉斯後驗門檻不免疫")
    print(f"                      bayes0.95={daily['Bayes P>0.95']:.1%}（門檻較鬆，膨脹更嚴重）")

    # 期望損失規則（決策論，另一種「錯」的定義）——回答計劃書「期望損失規則假陽性率是多少」
    eps = 2e-4
    el_rule = ex.RuleSpec("exploss", eps, "Expected loss<ε")
    el_fixed = ex.declare_rate(ex.fires_for_rule(table, ex.even_schedule(days, 1), el_rule))
    el_daily = ex.declare_rate(ex.fires_for_rule(table, np.arange(days), el_rule))
    print(f"\n期望損失規則（ε={eps:.0e}=0.02pp）: 固定={el_fixed:.1%} · 逐日偷看={el_daily:.1%}")
    print("  但它的『上線』是在後悔量已 <ε 時發生——A/A 下兩組本就相同，後悔確實可忽略，")
    print("  這是決策論意義的『不在乎』，與 p 值意義的假陽性不同（詳見決策段落）。")

    traj = ex.prob_ba_trajectories(aa, 400, Prior(1, 1), rng)
    plots.peeking_false_positive(fpr, traj, 0.975,
                                 os.path.join(FIG, "02_peeking_false_positive.png"))
    readme.update(fixed_freq=fixed["Frequentist p<0.05"], fixed_b975=fixed["Bayes P>0.975"],
                  daily_freq=daily["Frequentist p<0.05"], daily_b95=daily["Bayes P>0.95"],
                  daily_b975=daily["Bayes P>0.975"], el_fixed=el_fixed, el_daily=el_daily,
                  eps=eps, n_sims=n_sims, days=days)

    # 誠實的另一面：真有效果時，偷看偵測得更快
    sec("2b · 有真實效果時（power）：偷看的另一面")
    h1 = ex.Scenario(pA=0.05, pB=0.055, per_day=per_day, days=days, name="B 真的+10%")
    kA1, kB1 = ex.simulate_cumulative(h1, n_sims, rng)
    t1 = ex.build_peek_table(kA1, kB1, per_day, Prior(1, 1))
    fires_fixed = ex._fires_freq(t1, ex.even_schedule(days, 1), 0.05)
    fires_daily = ex._fires_freq(t1, np.arange(days), 0.05)
    power_fixed = fires_fixed.any(1).mean()
    power_daily = fires_daily.any(1).mean()
    first_day = np.where(fires_daily.any(1),
                         fires_daily.argmax(1) + 1, days)
    print(f"真實提升 +10%：固定視野偵測率={power_fixed:.1%}；逐日偷看偵測率={power_daily:.1%}；"
          f"偷看平均在第 {first_day.mean():.1f} 天就宣布")
    print("→ 偷看不是一無是處（更快），代價是 A/A 下的假陽性膨脹。這就是取捨。")
    readme.update(power_fixed=power_fixed, power_daily=power_daily, first_day=first_day.mean())

    # ─────────── 3 · 決策成本：何時值得上線 ───────────
    sec("3 · 決策成本：何時值得上線")
    ns = np.unique(np.logspace(np.log10(200), np.log10(30000), 40).astype(int))
    el_b = [ab.expected_loss_choose_b(*ab.posterior(round(0.052 * n), n),
                                      *ab.posterior(round(0.058 * n), n)) for n in ns]
    el_a = [ab.expected_loss_choose_a(*ab.posterior(round(0.052 * n), n),
                                      *ab.posterior(round(0.058 * n), n)) for n in ns]
    eps_levels = [1e-4, 5e-4]
    n_fixed = 4000
    lifts = np.linspace(0.0, 0.30, 40)
    aA0, bA0 = ab.posterior(round(0.052 * n_fixed), n_fixed)
    el_vs_lift = [ab.expected_loss_choose_b(aA0, bA0,
                                            *ab.posterior(round(0.052 * (1 + L) * n_fixed), n_fixed))
                  for L in lifts]
    # 找 ε=5e-4 時「安全上線」所需樣本數
    ship_n = next((n for n, e in zip(ns, el_b) if e < 5e-4), None)
    min_lift = next((L for L, e in zip(lifts, el_vs_lift) if e < 1e-4), None)
    print(f"觀測固定為 5.2% vs 5.8%：期望損失降到 ε=5e-4(0.05pp) 以下需約 {ship_n:,} 樣本/組")
    print(f"n={n_fixed}/組、ε=1e-4(0.01pp) 時，值得上線的最小相對提升 ≈ {min_lift:.0%}")
    plots.when_to_ship(ns, el_b, el_a, eps_levels, lifts, el_vs_lift, n_fixed,
                       os.path.join(FIG, "03_when_to_ship.png"))
    readme.update(ship_n=int(ship_n) if ship_n else None,
                  min_lift=float(min_lift) if min_lift is not None else None)

    # ─────────── 4 · 先驗敏感度（小樣本）───────────
    sec("4 · 先驗敏感度（小樣本，展示『不是萬靈丹』）")
    kAs, nAs, kBs, nBs = 10, 200, 17, 200            # 5.0% vs 8.5%，各 200
    priors = [("Uniform\nBeta(1,1)", Prior(1, 1)),
              ("Jeffreys\nBeta(.5,.5)", Prior(0.5, 0.5)),
              ("Informative\nBeta(5,95)", Prior(5, 95))]
    labels, pbas, els = [], [], []
    for lab, pr in priors:
        ss = ab.summarize(kAs, nAs, kBs, nBs, pr, rng)
        labels.append(lab); pbas.append(ss.prob_b_beats_a); els.append(ss.expected_loss_choose_b)
        print(f"{lab.replace(chr(10),' '):>24}: P(B>A)={ss.prob_b_beats_a:.1%}  "
              f"期望損失={ss.expected_loss_choose_b*100:.3g}pp")
    plots.prior_sensitivity(labels, pbas, els, f"A 10/200=5.0%, B 17/200=8.5%",
                            os.path.join(FIG, "04_prior_sensitivity.png"))
    readme.update(prior_pba_unif=pbas[0], prior_pba_info=pbas[2])

    # ─────────── README 數字彙整 ───────────
    sec("README 可用數字")
    for k, v in readme.items():
        print(f"  {k:>14} = {v}")
    print(f"\n四張圖已輸出至：{os.path.abspath(FIG)}")
    for f in sorted(os.listdir(FIG)):
        if f.endswith(".png"):
            print("   ·", f)


if __name__ == "__main__":
    main()
