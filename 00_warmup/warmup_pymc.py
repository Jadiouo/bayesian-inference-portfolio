"""
熱身測試（對應計劃書 §0.2）——動任何專案前，先確認環境能跑通 PyMC / ArviZ / matplotlib。

場景：一個診斷檢測，20 人受檢、3 人陽性，估計真陽性率 θ。
先驗 Beta(1,1)=Uniform → 後驗 Beta(1+3, 1+17)=Beta(4,18)，眾數=0.15。

通關標準：
  1. r_hat < 1.01（收斂）
  2. 後驗均值 ≈ 0.18（眾數 ≈ 0.15）
  3. 95% HDI ≈ [0.04, 0.33]  ← 注意它有多寬：20 個樣本能告訴你的事情很有限

執行：conda activate bayes && python 00_warmup/warmup_pymc.py
"""
import os
import matplotlib
matplotlib.use("Agg")            # headless：存檔而非開視窗
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    n, k = 20, 3
    with pm.Model():
        theta = pm.Beta("theta", alpha=1, beta=1)          # 先驗
        pm.Binomial("obs", n=n, p=theta, observed=k)       # 似然
        idata = pm.sample(2000, chains=4, random_seed=42, progressbar=False)

    summary = az.summary(idata, hdi_prob=0.95)
    print(summary)

    r_hat = float(summary.loc["theta", "r_hat"])
    mean = float(summary.loc["theta", "mean"])
    lo = float(summary.loc["theta", "hdi_2.5%"])
    hi = float(summary.loc["theta", "hdi_97.5%"])

    az.plot_posterior(idata, hdi_prob=0.95)
    out = os.path.join(HERE, "warmup_posterior.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n後驗圖已存：{out}")

    print("\n── 通關檢核 ──")
    ok_rhat = r_hat < 1.01
    ok_mean = 0.10 < mean < 0.24
    ok_hdi = lo < 0.10 and hi > 0.25
    print(f"[{'✅' if ok_rhat else '❌'}] r_hat = {r_hat:.4f}  (需 < 1.01)")
    print(f"[{'✅' if ok_mean else '❌'}] 後驗均值 = {mean:.3f}  (≈ 0.18，眾數 ≈ 0.15)")
    print(f"[{'✅' if ok_hdi else '❌'}] 95% HDI = [{lo:.3f}, {hi:.3f}]  (≈ [0.04, 0.33])")

    if ok_rhat and ok_mean and ok_hdi:
        print("\n🎉 環境通關：PyMC / ArviZ / matplotlib 一切正常。")
        return 0
    print("\n⚠️  有項目未達標，請檢查安裝。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
