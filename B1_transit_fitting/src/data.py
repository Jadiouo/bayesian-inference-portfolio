"""
B1 · 資料管線 —— Kepler-10 光曲線（B-D1）。

流程（計劃書步驟 1–2）：
  lightkurve 下載所有季度 → stitch → 去 NaN/離群 → flatten（去恆星自轉趨勢）
  → BLS 找週期/中天時刻 → 遮罩凌日再 flatten 一次（避免凹陷被稀釋）→ 相位摺疊。

結果快取到 data/B_astro/kepler10b.npz，之後 run_all / notebook 免重新下載。
"""
from __future__ import annotations

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def published_kepler10b() -> dict:
    """Kepler-10b 已發表參數（Dumusque et al. 2014 / NASA Exoplanet Archive）——拿來對答案。"""
    return dict(
        P=0.8374907,          # 週期 (天)
        rp_rs=0.01247,        # 行星/恆星半徑比
        rp_earth=1.47,        # 行星半徑 (地球半徑)
        a_rs=3.51,            # 縮放半長軸 a/R*
        b=0.30,               # 撞擊參數（近似）
        depth_ppm=155,        # 凌日深度 (ppm)
        source="Dumusque et al. 2014 / NASA Exoplanet Archive",
    )


def prepare(cache_dir: str, force: bool = False) -> dict:
    """下載並處理 Kepler-10，快取結果。回傳 dict（含摺疊資料、原始片段、BLS 參數）。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "kepler10b.npz")
    if os.path.exists(cache) and not force:
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    import lightkurve as lk

    sr = lk.search_lightcurve("Kepler-10", author="Kepler", cadence="long")
    lc = sr.download_all().stitch().remove_nans().remove_outliers(sigma=5)

    # 第一遍 flatten → BLS 找週期
    flat1 = lc.flatten(window_length=301)
    period = np.linspace(0.82, 0.86, 8000)
    bls = flat1.to_periodogram(method="bls", period=period, duration=0.06)
    P = float(bls.period_at_max_power.value)
    t0 = float(bls.transit_time_at_max_power.value)
    dur = float(bls.duration_at_max_power.value)

    # 遮罩凌日後再 flatten 一次（避免凹陷被 Savitzky-Golay 稀釋）
    intransit = lc.create_transit_mask(period=P, transit_time=t0, duration=dur * 1.6)
    flat = lc.flatten(window_length=301, mask=intransit)

    # 原始 vs flatten 的一小段（供圖用）：取中段約 3 天
    t = lc.time.value
    seg = (t > t[len(t) // 2]) & (t < t[len(t) // 2] + 3.0)
    raw_seg_t = t[seg]
    raw_seg_raw = lc.flux.value[seg] / np.nanmedian(lc.flux.value[seg])
    raw_seg_flat = flat.flux.value[seg]

    # 相位摺疊，取凌日窗
    folded = flat.fold(period=P, epoch_time=t0)
    ph = folded.phase.value
    fx = folded.flux.value
    win = np.abs(ph) < 0.08
    order = np.argsort(ph[win])

    out = dict(
        P=P, t0=t0, dur=dur,
        fold_phase=ph[win][order].astype(float),
        fold_flux=fx[win][order].astype(float),
        raw_seg_t=raw_seg_t.astype(float),
        raw_seg_raw=raw_seg_raw.astype(float),
        raw_seg_flat=raw_seg_flat.astype(float),
        n_points=len(lc),
    )
    np.savez(cache, **out)
    return out


def bin_fold(phase, flux, window: float = 0.06, n_bins: int = 80):
    """把摺疊資料在 |phase|<window 內分箱，回傳 (bin_phase, bin_flux, bin_err, bin_n)。"""
    m = np.abs(phase) < window
    ph, fx = phase[m], flux[m]
    edges = np.linspace(-window, window, n_bins + 1)
    idx = np.digitize(ph, edges) - 1
    bp, bf, be, bn = [], [], [], []
    for i in range(n_bins):
        sel = idx == i
        if sel.sum() >= 3:
            bp.append((edges[i] + edges[i + 1]) / 2)
            bf.append(fx[sel].mean())
            be.append(fx[sel].std() / np.sqrt(sel.sum()))
            bn.append(int(sel.sum()))
    return (np.array(bp), np.array(bf), np.array(be), np.array(bn))
