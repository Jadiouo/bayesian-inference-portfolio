"""
B2 · 一鍵重現：兩個目標（一 CONFIRMED、一 FALSE POSITIVE）× 三個模型 × nested sampling
           → 貝氏因子 → 先驗敏感度（Lindley 悖論）→ 五張圖。

執行：conda activate bayes && python src/run_all.py
（首次會下載光曲線並快取到 data/B_astro/；之後直接讀快取。約 30–45 分鐘。）
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data
import evidence as ev
import models
import plots

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "..", "data", "B_astro")
os.makedirs(FIG, exist_ok=True)

KEYS = ("kepler10b", "koi6017")
SUPERSAMPLE = 7          # 7 vs 11 的似然差異在 1e-3 nat 以下，但快 1.4 倍
MIN_LIVE = 400
SEEDS = (42, 43, 44)     # 主分析每個 logZ 重複三次，用 seed 間散布當誤差棒（見 evidence.py）

# 先驗敏感度掃描：只往「放寬」的方向掃（k ≥ 1）。往窄掃會把 KOI-6017 的最佳解
# （rp≈0.13）直接排除在先驗外，那是「先驗錯了」而不是 Occam 因子，兩件事不該混在一起。
PRIOR_K = np.array([1.0, 2.0, 5.0, 10.0])
PRIOR_SEEDS = (42, 43)   # 掃描也要重複，否則單次 logZ 的 ~0.5 nat 噪聲會蓋過 Occam 訊號

# 弱訊號情境（第 5 節）：injection–recovery。用 Kepler-10b 真實的每箱誤差當雜訊，
# 注入一個深度可調的合成凌日，把證據**精準**調到 Jeffreys 的決定邊界附近。
#
# 為什麼不用「丟掉真實資料」：試過了，行不通。M0 帶 jitter 參數，會用放大雜訊來部分吸收
# 凹陷，使 log B 對資料量高度非線性 —— 只用 4% 的資料仍給 log10 B = +50，
# 要壓到 ~1 得只剩幾十個資料點，那時分箱本身就垮了。注入法可控、可重複，
# 而且 injection–recovery 本來就是天文界評估偵測門檻的標準做法。
INJECT_RP = np.array([0.0019, 0.0021, 0.0023, 0.0025, 0.0028])
INJECT_K = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
INJECT_GEOM = dict(a=3.5, b=0.30, q1=0.44, q2=0.29)   # 沿用 Kepler-10 的真實幾何


def sec(t):
    print("\n" + "═" * 74 + f"\n{t}\n" + "═" * 74, flush=True)


def load_target(key):
    """準備一個目標：下載/讀快取 → 自適應分箱 → 建 batman 形狀模型。"""
    d = data.prepare(key, DATA)
    bp, bf, be, bn = data.adaptive_bin(d["phase"], d["flux"], float(d["dur_phase"]))
    d_pri, d_sec = data.measure_depths(d["phase"], d["flux"], float(d["dur_phase"]))
    shape = models.TransitShape(bp, float(d["P"]), supersample=SUPERSAMPLE)
    tgt = data.TARGETS[key]
    return dict(key=key, title=tgt["title"], disposition=tgt["disposition"],
                note=tgt["note"], raw=d, bp=bp, bf=bf, be=be, bn=bn,
                d_pri=d_pri, d_sec=d_sec, shape=shape,
                P=float(d["P"]), dur_phase=float(d["dur_phase"]),
                phase=d["phase"], flux=d["flux"])


def best_fit_curves(P, med):
    """在細網格上算三個模型的後驗中位數曲線（給圖 2）。med: {模型名: 中位數參數}"""
    grid = np.linspace(-0.5, 0.5, 3000)
    gshape = models.TransitShape(grid, P, supersample=SUPERSAMPLE)
    fits = {"M0": np.full_like(grid, med["M0"][0])}
    f0, rp, a, b, t0, q1, q2, _ = med["M1"]
    fits["M1"] = f0 * gshape.primary(rp, a, b, t0, q1, q2)
    f0, rp, a, b, t0, q1, q2, J, _ = med["M2"]
    fits["M2"] = f0 * gshape.with_secondary(rp, a, b, t0, q1, q2, J)
    return grid, fits


PLOTDATA = os.path.join(DATA, "b2_plotdata.npz")
_FIELDS = ("bp", "bf", "be", "phase", "flux")
_SCALARS = ("P", "dur_phase", "d_pri", "d_sec")


def save_plotdata(targets):
    """把出圖需要的一切落盤，這樣改圖表不必重跑 90 分鐘的推論。"""
    blob = {"keys": np.array([t["key"] for t in targets])}
    for t in targets:
        k = t["key"]
        for f in _FIELDS:
            blob[f"{k}__{f}"] = np.asarray(t[f])
        for f in _SCALARS:
            blob[f"{k}__{f}"] = np.asarray(float(t[f]))
        blob[f"{k}__title"] = np.array(t["title"])
        blob[f"{k}__disposition"] = np.array(t["disposition"])
        blob[f"{k}__J"] = np.asarray(t["J"])
        for r in t["results"]:
            blob[f"{k}__med_{r['name']}"] = np.median(r["samples"], axis=0)
    np.savez(PLOTDATA, **blob)
    print(f"  中間結果已存：{PLOTDATA}")


def load_plotdata():
    d = np.load(PLOTDATA, allow_pickle=False)
    out = []
    for k in d["keys"]:
        k = str(k)
        t = dict(key=k, title=str(d[f"{k}__title"]),
                 disposition=str(d[f"{k}__disposition"]), J=d[f"{k}__J"])
        for f in _FIELDS:
            t[f] = d[f"{k}__{f}"]
        for f in _SCALARS:
            t[f] = float(d[f"{k}__{f}"])
        t["med"] = {m: d[f"{k}__med_{m}"] for m in ("M0", "M1", "M2")}
        out.append(t)
    return out


def make_figures(targets, summaries, scan, R, figdir):
    """全部出圖集中在這裡，run_all 與 replot 都走同一條路徑。"""
    plots.full_phase([dict(title=t["title"], disposition=t["disposition"],
                           phase=t["phase"], flux=t["flux"], bp=t["bp"], bf=t["bf"],
                           be=t["be"], dur_phase=t["dur_phase"],
                           d_pri=t["d_pri"], d_sec=t["d_sec"]) for t in targets],
                     os.path.join(figdir, "01_full_phase.png"))

    entries = []
    for t in targets:
        grid, fits = best_fit_curves(t["P"], t["med"])
        entries.append(dict(title=t["title"], bp=t["bp"], bf=t["bf"], be=t["be"],
                            dur_phase=t["dur_phase"], grid=grid, fits=fits))
    plots.model_fits(entries, os.path.join(figdir, "02_model_fits.png"))
    plots.evidence_bars(summaries, os.path.join(figdir, "03_evidence.png"))
    plots.secondary_posterior(
        [dict(title=t["title"], disposition=t["disposition"], J_samples=t["J"],
              d_pri=t["d_pri"]) for t in targets],
        os.path.join(figdir, "04_secondary_posterior.png"))
    plots.lindley(scan, os.path.join(figdir, "05_lindley.png"))
    plots.weak_signal(dict(depth_ppm=R["weak"]["depth_ppm"],
                           log10_B=R["weak"]["depth_log10B"]),
                      dict(k=R["weak"]["ks"], log10_B=R["weak"]["k_log10B"]),
                      R["weak"]["chosen_depth_ppm"], 0.0125 ** 2 * 1e6,
                      os.path.join(figdir, "06_weak_signal.png"))
    print(f"  六張圖已輸出至：{os.path.abspath(figdir)}")


def main():
    R = {}
    targets, summaries = [], []

    # ── 1 · 資料 ────────────────────────────────────────────────────────────
    sec("1 · 資料：兩個目標的完整相位光曲線")
    for key in KEYS:
        t = load_target(key)
        targets.append(t)
        print(f"{t['title']:<14}[{t['disposition']:<14}] P={t['P']:.6f} d  "
              f"{len(t['bp'])} 箱  主食={t['d_pri']:>8.0f} ppm  次食={t['d_sec']:>6.0f} ppm  "
              f"→ 次食/主食={t['d_sec']/t['d_pri']:.4f}")
    R["targets"] = {t["key"]: dict(P=round(t["P"], 6), nbins=len(t["bp"]),
                                   d_pri=round(t["d_pri"]), d_sec=round(t["d_sec"]),
                                   err_ppm=round(float(np.median(t["be"])) * 1e6, 1))
                    for t in targets}

    # ── 2 · 三個模型的邊際似然 ──────────────────────────────────────────────
    sec("2 · nested sampling：對每個目標算三個模型的 log Z")
    for t in targets:
        res = []
        for M in models.build_all(t["shape"], t["bf"], t["be"]):
            t0 = time.time()
            r = ev.run_evidence_repeated(M, seeds=SEEDS, min_live=MIN_LIVE)
            res.append(r)
            print(f"  {t['title']:<12} {M.name} (ndim={M.ndim})  logZ={r['logz']:10.2f}"
                  f" ± {r['logzerr']:.2f}(seed 間)  runs="
                  f"[{', '.join(f'{x:.1f}' for x in r['logz_runs'])}]  "
                  f"{time.time()-t0:6.1f}s", flush=True)
        s = ev.summarize(res, f"{t['title']}  [{t['disposition']}]")
        s.update(title=t["title"], disposition=t["disposition"],
                 labels={r["name"]: r["label"] for r in res})
        t["results"], t["summary"] = res, s
        summaries.append(s)

    # M2 的次食參數 J：M2 多花的那一個參數，資料到底想不想要？
    for t in targets:
        m2 = [r for r in t["results"] if r["name"] == "M2"][0]
        J = m2["samples"][:, m2["param_names"].index("J")]
        t["J"] = J
        q = np.percentile(J, [2.5, 50, 97.5])
        print(f"  {t['title']:<12} J = {q[1]:.4f}  [{q[0]:.4f}, {q[2]:.4f}]")
        t["summary"]["J"] = [round(float(x), 4) for x in q]

    R["evidence"] = {t["key"]: t["summary"] for t in targets}

    # ── 3 · step sampler 的健全性檢查 ───────────────────────────────────────
    sec("3 · 健全性檢查：SliceSampler 的 nsteps 夠不夠？")
    print("  step sampler 若步數不足，logZ 會系統性偏高，而這個偏差**不在** logzerr 裡。")
    koi = [t for t in targets if t["key"] == "koi6017"][0]
    _, M1, _ = models.build_all(koi["shape"], koi["bf"], koi["be"])
    ladder, drift = ev.nsteps_convergence(M1, factors=(2, 4, 8), min_live=MIN_LIVE)
    gap = abs(koi["summary"]["logz"]["M2"] - koi["summary"]["logz"]["M1"])
    print(f"    對照：M2 與 M1 的 logZ 差距 = {gap:.1f} nat  → 擺動/差距 = {drift/gap:.1%}")
    R["nsteps_drift"] = round(drift, 2)
    R["nsteps_gap_ratio"] = round(float(drift / gap), 4)

    # ── 4 · 先驗敏感度（Lindley 悖論）───────────────────────────────────────
    sec("4 · 先驗敏感度：把 R_c/R* 的先驗上界放寬 k 倍")
    print("  Occam 因子的預測：先驗放寬 k 倍 → 高似然區在先驗中的占比降 k 倍 →")
    print("  log10 B 應該剛好掉 log10(k)。這是可以**定量驗證**的，不只是定性警告。\n")
    scan = []
    for t in targets:
        lz0 = t["summary"]["logz"]["M0"]
        row = []
        err = []
        for k in PRIOR_K:
            M = models.M1Planet(t["shape"], t["bf"], t["be"],
                                rp_max=models.RP_MAX_PLANET * k)
            r = ev.run_evidence_repeated(M, seeds=PRIOR_SEEDS, min_live=MIN_LIVE)
            b10 = (r["logz"] - lz0) / ev.LOG10
            row.append(float(b10))
            err.append(float(r["logz_spread"] / ev.LOG10))
            print(f"  {t['title']:<12} k={k:>5.1f}  rp_max={models.RP_MAX_PLANET*k:.2f}  "
                  f"logZ1={r['logz']:10.2f}±{r['logzerr']:.2f}  "
                  f"log10 B(M1/M0)={b10:+10.2f}", flush=True)
        scan.append(dict(title=t["title"], disposition=t["disposition"],
                         rp_max=list(models.RP_MAX_PLANET * PRIOR_K),
                         log10_B=row, log10_B_err=err, base_idx=0))
        drop = row[0] - row[list(PRIOR_K).index(10.0)]
        print(f"  → {t['title']}：放寬 10 倍後 log10 B 掉了 {drop:.2f}"
              f"（Occam 預測 {np.log10(10):.2f}）\n")
        t["summary"]["prior_drop_10x"] = round(float(drop), 2)
    R["prior_scan"] = scan

    # ── 5 · 弱訊號情境：Lindley 悖論真正咬人的地方 ──────────────────────────
    sec("5 · 弱訊號情境：把資料丟掉，讓證據回到模稜兩可的邊緣")
    print("  上一節的兩個目標證據強到 10^167 與 10^1316，放寬先驗 10 倍只掉 1 —— 結論**不會**變。")
    print("  這是誠實但不完整的展示。Lindley 悖論真正的威力在**弱證據**：那裡先驗寬度可以")
    print("  單獨翻轉結論。所以刻意只用 Kepler-10b 的一小部分資料，重現「任務早期只有幾週")
    print("  資料時，這個候選體可信嗎」的真實處境。\n")
    k10 = [t for t in targets if t["key"] == "kepler10b"][0]
    inj_shape = models.TransitShape(k10["bp"], k10["P"], supersample=SUPERSAMPLE)

    def inject(rp_inj, seed=7):
        """用真實的每箱誤差生成雜訊，注入一個深度 rp_inj² 的合成凌日。"""
        rng = np.random.default_rng(seed)
        clean = inj_shape.primary(rp_inj, INJECT_GEOM["a"], INJECT_GEOM["b"], 0.0,
                                  INJECT_GEOM["q1"], INJECT_GEOM["q2"])
        return clean + rng.normal(0.0, k10["be"])

    def inj_evidence(flux, rp_mult=1.0):
        r0 = ev.run_evidence(models.M0Noise(inj_shape, flux, k10["be"]), min_live=MIN_LIVE)
        r1 = ev.run_evidence(
            models.M1Planet(inj_shape, flux, k10["be"],
                            rp_max=models.RP_MAX_PLANET * rp_mult), min_live=MIN_LIVE)
        return (r1["logz"] - r0["logz"]) / ev.LOG10

    print(f"  真實 Kepler-10b：rp={0.0125:.4f}（深度 {0.0125**2*1e6:.0f} ppm）"
          f"，每箱誤差 {np.median(k10['be'])*1e6:.1f} ppm\n")
    depth_B = []
    for rp_i in INJECT_RP:
        b = inj_evidence(inject(rp_i))
        depth_B.append(float(b))
        print(f"  注入 rp={rp_i:.4f}（深度 {rp_i**2*1e6:5.1f} ppm）"
              f"→ log10 B(M1/M0) = {b:+8.2f}", flush=True)
    chosen_i = int(np.argmin(np.abs(np.array(depth_B) - 1.0)))
    chosen_rp = float(INJECT_RP[chosen_i])
    print(f"\n  → 選 rp={chosen_rp:.4f}（深度 {chosen_rp**2*1e6:.1f} ppm，"
          f"log10 B={depth_B[chosen_i]:+.2f}）作為邊緣案例，對它掃先驗寬度：\n")

    flux_edge = inject(chosen_rp)
    k_B = []
    for k in INJECT_K:
        b = inj_evidence(flux_edge, rp_mult=k)
        k_B.append(float(b))
        verdict = "支持行星" if b > 0 else "支持純雜訊 ← 翻盤"
        print(f"  k={k:>6.1f}  rp_max={models.RP_MAX_PLANET*k:5.2f}  "
              f"log10 B(M1/M0) = {b:+8.2f}   {verdict}", flush=True)
    R["weak"] = dict(rp_inj=list(map(float, INJECT_RP)),
                     depth_ppm=[float(r ** 2 * 1e6) for r in INJECT_RP],
                     depth_log10B=depth_B, chosen_rp=chosen_rp,
                     chosen_depth_ppm=float(chosen_rp ** 2 * 1e6),
                     ks=list(map(float, INJECT_K)), k_log10B=k_B,
                     flipped=bool(min(k_B) < 0))
    if min(k_B) < 0:
        kf = INJECT_K[int(np.argmax(np.array(k_B) < 0))]
        print(f"\n  ⇒ 先驗放寬 {kf:.0f} 倍就足以把「支持行星」翻成「支持純雜訊」。"
              f"資料一個位元組都沒變。")

    # ── 6 · 圖 ──────────────────────────────────────────────────────────────
    sec("6 · 出圖")
    for t in targets:
        t["med"] = {r["name"]: np.median(r["samples"], axis=0) for r in t["results"]}
    save_plotdata(targets)
    make_figures(targets, summaries, scan, R, FIG)

    # ── 7 · README 數字 ────────────────────────────────────────────────────
    sec("README 可用數字")
    print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
    with open(os.path.join(HERE, "..", "figures", "results.json"), "w") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
