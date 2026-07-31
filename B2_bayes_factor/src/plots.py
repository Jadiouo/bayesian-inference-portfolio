"""
B2 · 圖表（英文標籤；中文敘事在 README / notebook）。函式只負責畫，資料由 run_all 傳入。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
})

C_DATA = "#2e6fdb"
C_REF = "#6c757d"
C_M0 = "#8d99ae"
C_M1 = "#e8873a"
C_M2 = "#d1495b"
C_OK = "#2a9d8f"
MCOL = {"M0": C_M0, "M1": C_M1, "M2": C_M2}
# matplotlib 沒有 CJK 字體，圖上一律用英文（中文敘事在 README / notebook）
MLABEL = {"M0": "pure noise", "M1": "planet transit", "M2": "eclipsing binary"}


def _ppm(x):
    return (np.asarray(x) - 1.0) * 1e6


# ── 圖 1：完整相位光曲線 —— 次食一眼可見 ────────────────────────────────
def full_phase(targets, path):
    """targets: list of dict(title, disposition, phase, flux, bp, bf, be, dur_phase, d_pri, d_sec)"""
    n = len(targets)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.5 * n),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    axes = np.atleast_2d(axes)
    for i, t in enumerate(targets):
        axL, axR = axes[i]
        col = C_OK if t["disposition"] == "CONFIRMED" else C_M2

        axL.plot(t["phase"], _ppm(t["flux"]), ".", ms=1.0, color=C_REF, alpha=0.10)
        axL.errorbar(t["bp"], _ppm(t["bf"]), yerr=np.asarray(t["be"]) * 1e6,
                     fmt="o", ms=3, lw=0.8, color=C_DATA, zorder=3)
        axL.axhline(0, color="k", lw=0.7, alpha=0.5)
        axL.set_xlim(-0.5, 0.5)
        lo = -1.25 * abs(t["d_pri"])
        axL.set_ylim(lo, max(0.25 * abs(t["d_pri"]), 3 * abs(t["d_sec"]) + 50))
        axL.set_ylabel("flux − 1 (ppm)")
        axL.set_title(f"{t['title']}  [{t['disposition']}]   "
                      f"primary {t['d_pri']:.0f} ppm", color=col, fontweight="bold")
        axL.annotate("primary", (0, lo * 0.92), ha="center", fontsize=9, color=C_REF)
        for s in (-0.5, 0.5):
            axL.annotate("secondary?", (s, lo * 0.92), ha="center", fontsize=9, color=C_REF)
        if i == n - 1:
            axL.set_xlabel("Orbital phase")

        # 右：次食區放大 —— 這裡就是行星與食雙星的分水嶺
        # 窗口要夾上限：Kepler-10b 軌道極近、凌日就占週期 8%，4×dur 會寬到 ±0.3 相位
        zoom_w = min(2.5 * t["dur_phase"], 0.15)
        m = np.abs(np.abs(t["bp"]) - 0.5) < zoom_w
        x = np.where(t["bp"][m] > 0, t["bp"][m] - 0.5, t["bp"][m] + 0.5)
        o = np.argsort(x)
        axR.errorbar(x[o], _ppm(t["bf"][m])[o], yerr=np.asarray(t["be"])[m][o] * 1e6,
                     fmt="o", ms=4, lw=1.0, color=C_DATA)
        axR.axhline(0, color="k", lw=0.7, alpha=0.5)
        axR.axvline(0, color=C_REF, lw=0.7, ls=":")
        axR.set_title(f"secondary eclipse zoom:  {t['d_sec']:.0f} ppm", color=col)
        axR.set_ylabel("flux − 1 (ppm)")
        if i == n - 1:
            axR.set_xlabel("Phase from secondary centre")

    fig.suptitle("Full-phase light curves: the secondary eclipse is the giveaway",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 2：三個模型的最佳擬合 ────────────────────────────────────────────
def model_fits(targets, path):
    """每個目標兩欄：主食區、次食區；各畫 M0/M1/M2 的最佳擬合。"""
    n = len(targets)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.6 * n))
    axes = np.atleast_2d(axes)
    for i, t in enumerate(targets):
        w = 2.0 * t["dur_phase"]
        for j, (axis, centre, name) in enumerate(
                [(axes[i, 0], 0.0, "primary"), (axes[i, 1], 0.5, "secondary")]):
            if centre == 0.0:
                sel = np.abs(t["bp"]) < w
                xd = t["bp"][sel]
                xg = t["grid"][np.abs(t["grid"]) < w]
                gm = np.abs(t["grid"]) < w
            else:
                sel = np.abs(np.abs(t["bp"]) - 0.5) < w
                xd = np.where(t["bp"][sel] > 0, t["bp"][sel] - 0.5, t["bp"][sel] + 0.5)
                gm = np.abs(np.abs(t["grid"]) - 0.5) < w
                xg = np.where(t["grid"][gm] > 0, t["grid"][gm] - 0.5, t["grid"][gm] + 0.5)
            od, og = np.argsort(xd), np.argsort(xg)
            axis.errorbar(xd[od], _ppm(t["bf"][sel])[od],
                          yerr=np.asarray(t["be"])[sel][od] * 1e6,
                          fmt="o", ms=3.5, lw=0.9, color=C_DATA, zorder=4, label="binned data")
            for mname in ("M0", "M1", "M2"):
                axis.plot(xg[og], _ppm(t["fits"][mname][gm])[og], "-", lw=2.0,
                          color=MCOL[mname], alpha=0.9,
                          label=f"{mname} {MLABEL[mname]}")
            # y 範圍只看資料與 M1/M2。M0 是一條被凌日拉低的常數線，在次食圖裡遠離資料，
            # 讓它決定 y 範圍會把次食的細節壓扁（它在主食圖裡仍看得到）。
            span = np.concatenate([_ppm(t["bf"][sel]),
                                   _ppm(t["fits"]["M1"][gm]), _ppm(t["fits"]["M2"][gm])])
            pad = 0.15 * (span.max() - span.min() + 1e-9)
            axis.set_ylim(span.min() - pad, span.max() + pad)
            axis.set_title(f"{t['title']} — {name}", fontsize=11)
            axis.set_ylabel("flux − 1 (ppm)")
            if i == n - 1:
                axis.set_xlabel(f"Phase from {name} centre")
    # 圖例放圖外：次食區的資料點才是重點，不能被圖例蓋住
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Best fits of the three competing models", fontsize=13, y=1.003)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 3：邊際似然比較（奧卡姆剃刀現場）────────────────────────────────
def evidence_bars(summaries, path):
    """summaries: list of dict(title, disposition, logz{M0,M1,M2}, logzerr, maxlogl, probs)

    兩行：上行是三個模型的全景（M0 差到幾百∼幾千 nat，一眼看出被排除）；
    下行放大 M1 vs M2 —— 那才是有趣的比較，也是奧卡姆剃刀唯一看得見的地方。
    只畫全景的話，M1/M2 之間幾 nat 的差距會被 M0 的巨大落差壓成一條線。
    """
    n = len(summaries)
    fig, axes = plt.subplots(2, n, figsize=(6.4 * n, 8.4),
                             gridspec_kw={"height_ratios": [1, 1.15]})
    axes = np.atleast_2d(axes.reshape(2, n))
    names = ["M0", "M1", "M2"]

    for j, s in enumerate(summaries):
        col = C_OK if s["disposition"] == "CONFIRMED" else C_M2
        lz = np.array([s["logz"][k] for k in names])
        ml = np.array([s["maxlogl"][k] for k in names])
        err = np.array([s["logzerr"][k] for k in names])
        ref = lz.max()
        best = names[int(np.argmax(lz))]

        # ── 上：全景 ────────────────────────────────────────────────
        axT = axes[0, j]
        x = np.arange(3)
        axT.bar(x, lz - ref, yerr=err, color=[MCOL[k] for k in names],
                alpha=0.85, width=0.55, capsize=4, label="log Z (evidence)")
        axT.plot(x, ml - ref, "k^", ms=9, zorder=5, label="max log L (best fit)")
        axT.axhline(0, color="k", lw=0.8)
        axT.set_xticks(x)
        axT.set_xticklabels([f"{k}\n{MLABEL[k]}" for k in names])
        axT.set_ylabel("log Z − log Z(best)   [nats]")
        axT.set_title(f"{s['title']}  [{s['disposition']}]     winner: {best}",
                      color=col, fontweight="bold")
        axT.legend(fontsize=9, loc="lower right")
        axT.annotate(f"M0 rejected\nby {abs(lz[0]-ref):,.0f} nats",
                     (0, (lz - ref)[0] * 0.5), ha="center", va="center",
                     fontsize=9.5, color="white", fontweight="bold")

        # ── 下：M1 vs M2 放大 —— 奧卡姆剃刀 ─────────────────────────
        axB = axes[1, j]
        dL = s["maxlogl"]["M2"] - s["maxlogl"]["M1"]
        dZ = s["logz"]["M2"] - s["logz"]["M1"]
        vals = [dL, dZ]
        cols = [C_M1 if v < 0 else C_OK for v in vals]
        bars = axB.bar([0, 1], vals, color=cols, alpha=0.85, width=0.5)
        axB.axhline(0, color="k", lw=1.1)
        axB.set_xticks([0, 1])
        axB.set_xticklabels(["Δ max log L\n(does M2 fit better?)",
                             "Δ log Z\n(is M2 more probable?)"])
        axB.set_ylabel("M2 − M1   [nats]")
        for b, v in zip(bars, vals):
            axB.annotate(f"{v:+.1f}", (b.get_x() + b.get_width() / 2, v),
                         textcoords="offset points",
                         xytext=(0, 8 if v > 0 else -16), ha="center",
                         fontsize=11, fontweight="bold")
        pen = dL - dZ
        verdict = ("fits better BUT less probable → not worth it"
                   if dL > 0 > dZ else
                   ("fits better AND more probable → worth it" if dZ > 0 else
                    "no better either way"))
        axB.set_title(f"log$_{{10}}$B(M2/M1) = {s['log10_B_M2M1']:+.2f}   —   {verdict}\n"
                      f"Occam penalty for the extra parameter $J$:  −{pen:.1f} nats",
                      fontsize=10.5, color=col)
        m = max(abs(dL), abs(dZ), 1.0)
        axB.set_ylim(-1.7 * m, 1.7 * m)

    fig.suptitle("Marginal likelihood picks the right model — and the bottom row is "
                 "where max log L and log Z disagree (Occam's razor)",
                 fontsize=12.5, y=1.005)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 4：次食參數 J 的後驗 —— 為什麼 M2 贏 / 輸 ────────────────────────
def secondary_posterior(entries, path):
    """entries: list of dict(title, disposition, J_samples, d_pri)

    J 本身不夠說明問題：兩個目標的 J 中位數其實同一量級（~0.02 vs ~0.03）。
    真正的差別在**相對不確定度**與**換算成 ppm 的絕對深度**，所以兩者都標出來。
    """
    fig, axes = plt.subplots(1, len(entries), figsize=(5.9 * len(entries), 4.4))
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, entries):
        J = np.asarray(e["J_samples"])
        col = C_OK if e["disposition"] == "CONFIRMED" else C_M2
        ax.hist(J, bins=60, color=col, alpha=0.75, density=True)
        q = np.percentile(J, [2.5, 50, 97.5])
        rel = 100 * (q[2] - q[0]) / 2 / q[1]
        dep = q * e["d_pri"]
        ax.axvline(q[1], color="k", lw=1.4)
        ax.axvspan(q[0], q[2], color="k", alpha=0.10)
        ax.set_xlabel("J  (secondary / primary surface-brightness ratio)")
        ax.set_ylabel("posterior density")
        ax.set_title(f"{e['title']}  [{e['disposition']}]\n"
                     f"J = {q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}]   (±{rel:.0f}%)\n"
                     f"secondary depth = {dep[1]:.1f} ppm "
                     f"[{dep[0]:.1f}, {dep[2]:.1f}]", color=col, fontweight="bold",
                     fontsize=11)
        ax.set_xlim(left=0)
    fig.suptitle("The extra parameter M2 buys: does the data actually want a secondary?",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 5：Lindley 悖論 —— 貝氏因子對先驗寬度的敏感度 ────────────────────
def lindley(scan, path):
    """scan: list of dict(title, disposition, rp_max[], log10_B[], log10_B_err[])

    每個目標一個 panel，y 軸各自縮放。共用一個 y 軸是行不通的：
    兩個目標的 log10 B 是 167 與 1316，而要看的變化只有 ~1，畫在一起就是兩條直線。
    """
    n = len(scan)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.8))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, scan):
        col = C_OK if s["disposition"] == "CONFIRMED" else C_M2
        rel = np.asarray(s["rp_max"]) / s["rp_max"][s["base_idx"]]
        base = s["log10_B"][s["base_idx"]]
        ax.errorbar(rel, s["log10_B"], yerr=s.get("log10_B_err"), fmt="o-", color=col,
                    lw=2, ms=7, capsize=4, label="observed", zorder=4)
        ax.plot(rel, base - np.log10(rel), "k--", lw=1.7,
                label="Occam prediction:\n$\\log_{10}B_0-\\log_{10}k$")
        ax.plot(rel[s["base_idx"]], base, "*", color="k", ms=17, zorder=5)
        # 觀測是否真的貼著 Occam 預測走？用最寬的那一點判斷
        dev = s["log10_B"][-1] - (base - np.log10(rel[-1]))
        verdict = ("follows the Occam factor" if abs(dev) < 0.35
                   else f"departs by {dev:+.1f} dex — see README §7")
        ax.set_xscale("log")
        ax.set_xlabel("prior width multiplier $k$  (on $R_c/R_*$ upper bound)")
        ax.set_ylabel("log$_{10}$ B(M1/M0)")
        ax.set_title(f"{s['title']}  [{s['disposition']}]\n"
                     f"★ baseline = {base:,.1f}   —   {verdict}", color=col,
                     fontweight="bold", fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle("Prior sensitivity: the Occam factor holds only while the best fit "
                 "sits *inside* the prior", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 圖 6：弱訊號才是 Lindley 真正咬人的地方 ────────────────────────────
def weak_signal(depth_scan, k_scan, chosen_depth, real_depth, path):
    """depth_scan: dict(depth_ppm[], log10_B[])；k_scan: dict(k[], log10_B[])"""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))

    axL.plot(depth_scan["depth_ppm"], depth_scan["log10_B"], "o-",
             color=C_DATA, lw=2, ms=7)
    axL.axhline(0, color="k", lw=0.9)
    axL.axhspan(-0.5, 0.5, color=C_REF, alpha=0.18)
    axL.axvline(chosen_depth, color=C_M2, ls="--", lw=1.5)
    axL.annotate("inconclusive zone", (axL.get_xlim()[0], 0.5), fontsize=9,
                 color=C_REF, va="bottom", ha="left")
    axL.annotate(f"chosen:\n{chosen_depth:.1f} ppm", (chosen_depth, axL.get_ylim()[1]),
                 textcoords="offset points", xytext=(6, -24), fontsize=9, color=C_M2)
    axL.set_xlabel("injected transit depth (ppm)")
    axL.set_ylabel("log$_{10}$ B(M1/M0)")
    axL.set_title("Injection–recovery: dial the signal down to the\n"
                  f"decision boundary (real Kepler-10b is {real_depth:.0f} ppm, "
                  "at $10^{167}$)")

    k = np.asarray(k_scan["k"], float)
    B = np.asarray(k_scan["log10_B"], float)
    axR.plot(k, B, "o-", color=C_M2, lw=2.2, ms=7)
    axR.axhline(0, color="k", lw=1.2)
    axR.axhspan(axR.get_ylim()[0] if B.min() < 0 else -1, 0, color=C_M2, alpha=0.10)
    for thr, lab in [(2, "decisive"), (1, "strong"), (0.5, "substantial")]:
        axR.axhline(thr, color=C_REF, ls=":", lw=0.9)
        axR.annotate(lab, (k.max(), thr), fontsize=8, color=C_REF, va="bottom", ha="right")
    flip = B < 0
    if flip.any():
        axR.annotate("conclusion flips:\nnow favours pure noise",
                     (k[flip][0], B[flip][0]), textcoords="offset points",
                     xytext=(70, 42), fontsize=9.5, color=C_M2, ha="center",
                     arrowprops=dict(arrowstyle="->", color=C_M2, lw=1.4))
    axR.set_xscale("log")
    axR.set_xlabel("prior width multiplier $k$  (on $R_c/R_*$ upper bound)")
    axR.set_ylabel("log$_{10}$ B(M1/M0)")
    axR.set_title(f"Same data ({chosen_depth:.0f} ppm signal), same likelihood —\n"
                  "only the prior width changed")

    fig.suptitle("Lindley's paradox bites at the margin: with weak evidence, "
                 "a wider prior alone can reverse the verdict", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
