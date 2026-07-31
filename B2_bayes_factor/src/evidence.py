"""
B2 · 邊際似然與貝氏因子 —— ultranest（nested sampling）。

計劃書步驟 2–3。**為什麼不能用 MCMC？**
MCMC 的接受率是 p(θ'|D)/p(θ|D)，比值裡的 p(D) 被消掉了 —— 這正是 Metropolis 不需要
歸一化常數的原因，也正是它**算不出** p(D) 的原因。而模型選擇要的恰恰就是這個被消掉的東西：

    p(M|D) ∝ p(D|M) p(M)，   p(D|M) = ∫ p(D|θ,M) p(θ|M) dθ  ← 邊際似然（證據）

Nested sampling 換一個角度：把 d 維積分改寫成對「先驗體積 X」的一維積分
    Z = ∫ L dX，  X(λ) = ∫_{L(θ)>λ} p(θ) dθ
再用一群活點由外往內收縮來估 X。它是**專門為了直接算 log Z 而設計的**。

副產品：先驗體積的收縮天然帶來奧卡姆因子 —— 參數多、先驗鬆的模型，
高似然區在先驗體積中的占比小，Z 就被壓低。**不需要外加任何懲罰項。**
"""
from __future__ import annotations

import contextlib
import io
import logging
import os

import numpy as np

LOG10 = float(np.log(10.0))

# Jeffreys（1961）尺度，以 log10 B 表示
JEFFREYS = [
    (0.0, "無足輕重"),
    (0.5, "勉強一提"),
    (1.0, "實質"),
    (1.5, "強"),
    (2.0, "非常強"),
    (np.inf, "決定性"),
]


def jeffreys_label(log10_B: float) -> str:
    """把 log10 貝氏因子翻成 Jeffreys 的文字級別（附方向）。"""
    a = abs(log10_B)
    lvl = next(name for thr, name in JEFFREYS if a < thr or thr == np.inf)
    if a < 0.5:
        return f"{lvl}（兩模型難分）"
    return f"{lvl}證據，支持{'前者' if log10_B > 0 else '後者'}"


STEP_SAMPLER_MIN_DIM = 7      # 超過這個維度改用 step sampler（見下）
NSTEPS_FACTOR = 4             # nsteps = 4×ndim。由 nsteps_convergence 的階梯檢查定出：
#                               2×/4×/8× 在 KOI-6017.01 的 M1 上給 1090.68/1091.21/1091.35，
#                               4×→8× 只差 0.14 nat（已收斂），再加倍不划算。


def run_evidence(model, min_live=400, frac_remain=0.01, seed=42, verbose=False,
                 nsteps_factor=None):
    """對一個模型跑 nested sampling，回傳 dict(logz, logzerr, samples, ...)。

    **抽樣器的選擇是這個專案最大的效能關卡。** ultranest 預設的 MLFriends 用橢球包絡
    來拒絕抽樣，維度一高、後驗一簡併，接受率就崩潰。實測 M1（8 維、資料 SNR≈1000）
    用 MLFriends 跑 12 分鐘還沒收斂；換成 SliceSampler（沿隨機方向做切片抽樣，
    每次迭代的成本固定為 nsteps 次似然求值）後 **63 秒**完成，而且活點數還從 200 提到 400。

    這不是調參技巧，是 nested sampling 的已知性質：官方建議 ndim ≳ 7 就該換 step sampler。
    代價是 nsteps 太小會讓鏈沒走遠、logZ 系統性偏高 —— 所以另外有 `nsteps_convergence`
    做階梯檢查（步驟見 run_all）。
    """
    import ultranest
    import ultranest.stepsampler as uss

    if nsteps_factor is None:          # 在模組層讀取，這樣呼叫端可在執行期調整
        nsteps_factor = NSTEPS_FACTOR
    np.random.seed(seed)
    sampler = ultranest.ReactiveNestedSampler(
        model.param_names, model.loglike, model.prior_transform,
        vectorized=False,
    )
    if not verbose:
        # ultranest 在**建構時**才把 handler 掛上 'ultranest' logger，所以只有在這裡
        # 清掉才壓得住；在模組層先設 level 會被它覆蓋，notebook 會被幾十段日誌淹沒。
        _log = logging.getLogger("ultranest")
        _log.handlers.clear()
        _log.setLevel(logging.ERROR)
    if model.ndim >= STEP_SAMPLER_MIN_DIM:
        sampler.stepsampler = uss.SliceSampler(
            nsteps=nsteps_factor * model.ndim,
            generate_direction=uss.generate_mixture_random_direction,
        )
    buf = io.StringIO()
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buf)
    with ctx:
        res = sampler.run(min_num_live_points=min_live, dlogz=0.5,
                          frac_remain=frac_remain, show_status=verbose,
                          viz_callback=False)
    return dict(
        name=model.name, label=model.label, ndim=model.ndim,
        logz=float(res["logz"]), logzerr=float(res["logzerr"]),
        maxloglike=float(np.max(res["weighted_samples"]["logl"])),
        posterior_mean=np.array(res["posterior"]["mean"], float),
        posterior_std=np.array(res["posterior"]["stdev"], float),
        samples=np.array(res["samples"], float),
        param_names=list(model.param_names),
        niter=int(res["niter"]),
        ncall=int(res["ncall"]),
    )


def run_evidence_repeated(model, seeds=(42, 43, 44), min_live=400, **kw):
    """跑多個獨立 seed，回傳合併結果，其中 logzerr 取「實測的 seed 間散布」。

    **為什麼非做不可**：ultranest 回報的 logzerr 只涵蓋 nested sampling 的統計誤差，
    不涵蓋 step sampler 走不夠遠造成的偏差。而本專案最精緻的結論
    —— Kepler-10b 的 M1 只贏 M2 約 2 nat —— 差距小到跟數值誤差同一個量級。
    唯一誠實的做法是把同一個計算重複幾次，用實際的散布當誤差棒；
    若散布蓋過差距，就必須說「分不出來」，而不是報一個好看的數字。
    """
    runs = [run_evidence(model, min_live=min_live, seed=s, **kw) for s in seeds]
    lz = np.array([r["logz"] for r in runs])
    best = runs[int(np.argmax([r["maxloglike"] for r in runs]))]
    out = dict(best)
    out.update(
        logz=float(lz.mean()),
        logzerr=float(lz.std(ddof=1)) if len(lz) > 1 else float(runs[0]["logzerr"]),
        logz_internal=float(np.mean([r["logzerr"] for r in runs])),
        logz_runs=[float(x) for x in lz],
        logz_spread=float(lz.max() - lz.min()),
        nseeds=len(seeds),
    )
    return out


def nsteps_convergence(model, factors=(2, 4, 8), min_live=400, seed=42):
    """階梯檢查：加倍 SliceSampler 的步數，看 logZ 是否已經穩定。

    step sampler 唯一的系統性風險是「鏈走得不夠遠 → 新活點與舊的相關 → logZ 偏高」。
    這個偏差不會反映在 ultranest 回報的 logzerr 裡（那只算統計誤差），
    所以**必須另外檢查**。若 logZ 在 nsteps 加倍後變動遠小於模型間的差距，結論就安全。
    """
    out = []
    for f in factors:
        r = run_evidence(model, min_live=min_live, seed=seed, nsteps_factor=f)
        out.append(dict(factor=f, nsteps=f * model.ndim,
                        logz=r["logz"], logzerr=r["logzerr"], ncall=r["ncall"]))
        print(f"    nsteps={f * model.ndim:>3} ({f}×ndim): logZ = {r['logz']:10.2f} "
              f"± {r['logzerr']:.2f}   ncall={r['ncall']:,}")
    drift = max(o["logz"] for o in out) - min(o["logz"] for o in out)
    print(f"    → logZ 在 nsteps 2×→8× 之間的擺動 = {drift:.2f} nat")
    return out, float(drift)


def bayes_factor(res_num, res_den):
    """log10 B = (logZ_num − logZ_den)/ln10，含誤差傳遞。回傳 (log10_B, err)。"""
    d = res_num["logz"] - res_den["logz"]
    e = np.hypot(res_num["logzerr"], res_den["logzerr"])
    return d / LOG10, e / LOG10


def model_probabilities(results, priors=None):
    """給定各模型的 logZ，算後驗模型機率 p(M|D)（預設模型先驗等權）。"""
    lz = np.array([r["logz"] for r in results], float)
    lp = np.log(np.array(priors, float)) if priors is not None else np.zeros(len(lz))
    x = lz + lp
    x -= x.max()
    w = np.exp(x)
    return w / w.sum()


def summarize(results, target_title):
    """印出一個目標的三模型比較表，回傳可寫進 README 的 dict。"""
    probs = model_probabilities(results)
    by = {r["name"]: r for r in results}
    print(f"\n── {target_title} ─────────────────────────────────────────")
    print(f"{'模型':<6}{'說明':<10}{'參數':>4}{'log Z':>14}{'±(seed)':>10}"
          f"{'±(內部)':>10}{'max logL':>12}{'p(M|D)':>10}")
    for r, p in zip(results, probs):
        print(f"{r['name']:<6}{r['label']:<10}{r['ndim']:>4}{r['logz']:>14.2f}"
              f"{r['logzerr']:>10.2f}{r.get('logz_internal', r['logzerr']):>10.2f}"
              f"{r['maxloglike']:>12.1f}{p:>10.3f}")

    out = {}
    for num, den in (("M1", "M0"), ("M2", "M1"), ("M2", "M0")):
        b, e = bayes_factor(by[num], by[den])
        key = f"log10_B_{num}{den}"
        out[key] = round(float(b), 2)
        print(f"  log10 B({num}/{den}) = {b:+8.2f} ± {e:.2f}   → {jeffreys_label(b)}")

    best = results[int(np.argmax([r["logz"] for r in results]))]
    out["logz_spread"] = {r["name"]: round(r.get("logz_spread", 0.0), 2) for r in results}
    out.update(best=best["name"], best_label=best["label"],
               probs={r["name"]: round(float(p), 4) for r, p in zip(results, probs)},
               logz={r["name"]: round(r["logz"], 2) for r in results},
               logzerr={r["name"]: round(r["logzerr"], 2) for r in results},
               maxlogl={r["name"]: round(r["maxloglike"], 1) for r in results})
    print(f"  → 勝出：{best['name']}（{best['label']}）")
    return out
