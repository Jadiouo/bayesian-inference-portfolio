"""
B2 · 資料管線 —— 兩個目標的「完整相位」光曲線。

與 B1 的關鍵差異有三個，都是為了模型選擇服務：

1. **摺疊要覆蓋完整相位 [-0.5, 0.5]**。B1 只要凌日窗（|phase|<0.08）就能估參數；
   B2 要判斷「是不是食雙星」，而食雙星的決定性證據是**次食（相位 0.5 附近的第二個凹陷）**。
   只看凌日窗 = 把最關鍵的證據丟掉。
2. **flatten 要同時遮罩主食與次食**。不遮罩的話 Savitzky-Golay 會把凹陷當成趨勢吸收掉，
   次食本來就只有幾百 ppm，一稀釋就沒了。
3. **sigma clip 只能砍上方**（見下方 `remove_outliers`）。

參數見 TARGETS：一個 CONFIRMED（Kepler-10b）、一個真的 FALSE POSITIVE（KOI-6017.01）。
結果快取到 data/B_astro/b2_<key>.npz。
"""
from __future__ import annotations

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 兩個目標：一真一假。假的那個是從 NASA Exoplanet Archive 的 KOI 累積表挑出來的，
# 條件是 koi_disposition='FALSE POSITIVE' 且 koi_fpflag_ss=1（Stellar Eclipse 旗標），
# 並同時具備 DEEP_V_SHAPED（V 字形）與 HAS_SEC_TCE（偵測到次食）兩個註記
# —— 也就是說，官方判定它是假行星的理由，正是我們的 M2 模型要抓的兩個特徵。
# ─────────────────────────────────────────────────────────────────────────────
TARGETS = {
    "kepler10b": dict(
        search="Kepler-10",
        title="Kepler-10b",
        disposition="CONFIRMED",
        P0=0.8374907,          # 已發表週期（Dumusque et al. 2014），BLS 以此為中心精修
        koi_depth_ppm=155,
        note="已確認的岩質行星（Kepler 任務第一顆確認的岩石行星）",
    ),
    "koi6017": dict(
        search="KIC 5961350",
        title="KOI-6017.01",
        disposition="FALSE POSITIVE",
        P0=5.262652,           # KOI 累積表週期
        koi_depth_ppm=22650,
        note="官方 FALSE POSITIVE；旗標 koi_fpflag_ss=1，註記 DEEP_V_SHAPED + HAS_SEC_TCE",
    ),
}


def prepare(key: str, cache_dir: str, force: bool = False) -> dict:
    """下載並處理一個目標，回傳完整相位摺疊資料。結果快取。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"b2_{key}.npz")
    if os.path.exists(cache) and not force:
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    import lightkurve as lk

    tgt = TARGETS[key]
    lc = lk.search_lightcurve(tgt["search"], author="Kepler", cadence="long").download_all()
    lc = lc.stitch().remove_nans()

    # ── 踩坑 1：sigma clip 只砍上方 ──────────────────────────────────────────
    # 對稱的 remove_outliers(sigma=5) 會把 2% 深的食**當成離群值刪掉**（實測砍掉 1391 點，
    # 剛好就是食內的點）。凹陷是訊號不是雜訊，只砍上方（閃焰、宇宙射線）才對。
    lc = lc.remove_outliers(sigma_upper=5, sigma_lower=50)

    # ── 第一遍 flatten → BLS 精修週期 ───────────────────────────────────────
    # 週期搜尋本身在 B1 已完整展示（那裡從盲搜到對上已發表值 4 ppm）；B2 的主題是模型選擇，
    # 所以這裡以目錄值為中心做 ±3% 精修，把算力留給 nested sampling。
    flat1 = lc.flatten(window_length=901)
    P0 = tgt["P0"]
    grid = np.linspace(P0 * 0.97, P0 * 1.03, 20000)
    # duration 網格必須**隨週期縮放**：兩個目標週期差 6 倍，固定的天數網格會讓短週期目標
    # 的凌日時長被系統性高估（Kepler-10b 軌道極近，凌日占週期的 ~10%）。
    bls = flat1.to_periodogram(method="bls", period=grid,
                               duration=P0 * np.array([0.02, 0.04, 0.06, 0.08, 0.12]))
    P = float(bls.period_at_max_power.value)
    t0 = float(bls.transit_time_at_max_power.value)
    dur = float(bls.duration_at_max_power.value)

    # ── 踩坑 2：遮罩主食「與次食」後再 flatten ──────────────────────────────
    m_pri = lc.create_transit_mask(period=P, transit_time=t0, duration=dur * 2.0)
    m_sec = lc.create_transit_mask(period=P, transit_time=t0 + P / 2, duration=dur * 2.0)
    flat = lc.flatten(window_length=901, mask=(m_pri | m_sec))

    # ── 完整相位摺疊 ────────────────────────────────────────────────────────
    folded = flat.fold(period=P, epoch_time=t0)
    ph = folded.phase.value / P          # 轉成 [-0.5, 0.5] 的無因次相位
    fx = folded.flux.value
    good = np.isfinite(ph) & np.isfinite(fx)
    ph, fx = ph[good], fx[good]
    order = np.argsort(ph)

    # 給圖用的一小段原始 vs 去趨勢曲線（約 3 個週期）
    t = lc.time.value
    span = min(3 * P, 12.0)
    seg = (t > t[len(t) // 2]) & (t < t[len(t) // 2] + span)

    out = dict(
        P=P, t0=t0, dur=dur, dur_phase=dur / P,
        phase=ph[order].astype(float),
        flux=fx[order].astype(float),
        seg_t=t[seg].astype(float),
        seg_raw=(lc.flux.value[seg] / np.nanmedian(lc.flux.value[seg])).astype(float),
        seg_flat=flat.flux.value[seg].astype(float),
        n_points=len(lc),
    )
    np.savez(cache, **out)
    return out


def bin_window(dur_phase, pad=1.5, cap=0.16):
    """細分箱區的半寬：凌日全寬的 pad 倍，但夾在 cap 以下。

    cap 是必要的：Kepler-10b 週期只有 20 小時、a/R*≈3.5，凌日就占了週期的 ~10%，
    不設上限的話主食細箱區會跟次食細箱區重疊（相位空間只有 1.0 可分）。
    """
    return float(min(pad * dur_phase, cap))


def adaptive_bin(phase, flux, dur_phase, n_fine=80, n_sec=44, n_coarse=40, pad=1.5):
    """自適應分箱：主食與次食附近細分箱，其餘平坦區粗分箱。

    為什麼不用均勻分箱：均勻細箱會讓多數資料點落在毫無資訊的平坦區，
    nested sampling 的似然求值成本卻照付。自適應分箱把解析度放在有訊號的地方
    （形狀 → 區分 U 形/V 形；次食 → 區分行星/食雙星），總箱數壓在 ~180。

    回傳 (bin_phase, bin_flux, bin_err, bin_n)。誤差用箱內標準誤 SEM。
    """
    w = bin_window(dur_phase, pad)
    edges = np.concatenate([
        np.linspace(-0.5, -0.5 + w, n_sec // 2 + 1),     # 次食左半（相位 −0.5 側）
        np.linspace(-0.5 + w, -w, n_coarse // 2 + 1)[1:],
        np.linspace(-w, w, n_fine + 1)[1:],              # 主食
        np.linspace(w, 0.5 - w, n_coarse // 2 + 1)[1:],
        np.linspace(0.5 - w, 0.5, n_sec // 2 + 1)[1:],   # 次食右半
    ])
    assert np.all(np.diff(edges) > 0), "分箱邊界重疊（dur_phase 過大？）"
    idx = np.digitize(phase, edges) - 1
    bp, bf, be, bn = [], [], [], []
    for i in range(len(edges) - 1):
        sel = idx == i
        n = int(sel.sum())
        if n >= 5:
            bp.append(0.5 * (edges[i] + edges[i + 1]))
            bf.append(float(flux[sel].mean()))
            be.append(float(flux[sel].std(ddof=1) / np.sqrt(n)))
            bn.append(n)
    return np.array(bp), np.array(bf), np.array(be), np.array(bn)


def measure_depths(phase, flux, dur_phase):
    """粗量主食與次食深度（ppm）——給圖說與健全性檢查用，不進似然。"""
    half = 0.35 * dur_phase
    pri = np.abs(phase) < half
    sec = np.abs(np.abs(phase) - 0.5) < half
    edge = bin_window(dur_phase)      # 基線只取兩個食「之外」的平坦區
    base = np.median(flux[(np.abs(phase) > edge) &
                          (np.abs(np.abs(phase) - 0.5) > edge)])
    d_pri = (base - np.median(flux[pri])) * 1e6 if pri.sum() else np.nan
    d_sec = (base - np.median(flux[sec])) * 1e6 if sec.sum() else np.nan
    return float(d_pri), float(d_sec)
