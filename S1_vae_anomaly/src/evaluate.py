"""
S1 · 評估指標 —— AUROC、bootstrap 區間、以及「控制複雜度後」的分層 AUROC。

為什麼需要分層 AUROC
--------------------
VAE 異常偵測有一個結構性的病:**ELBO 強烈依賴影像複雜度**。
平坦、低對比的圖容易重建(ELBO 高),紋理豐富的圖難重建(ELBO 低)。
如果 pneumonia 影像剛好比 normal 更複雜,那麼一個完全沒學到病理的模型
也會有漂亮的 AUROC —— 它只是在測量「這張圖多難壓縮」。

`stratified_auroc` 把樣本按複雜度分箱、**箱內**算 AUROC 再加權平均。
箱內的正常/異常影像複雜度相當,所以剩下的判別力不可能來自複雜度。
這是本專案最重要的誠實檢查,對應 A3 專案的 trivial-baseline 精神。
"""
from __future__ import annotations

import io

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC。`labels` 1 = 異常;`scores` 越大越像異常。

    用 rank 公式(Mann–Whitney U)而不是梯形積分,平手會被正確地算成 0.5。
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # 平手取平均 rank
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_ci(scores, labels, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC 的 bootstrap 95% 區間(對正負兩組**分層**重抽,保持組大小)。"""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        boot[b] = auroc(s[idx], y[idx])
    return {"auroc": auroc(s, y),
            "lo95": float(np.percentile(boot, 2.5)),
            "hi95": float(np.percentile(boot, 97.5)),
            "boot_sd": float(boot.std(ddof=1))}


def paired_auroc_ci(s_a, s_b, labels, n_boot: int = 2000, seed: int = 0) -> dict:
    """兩個分數的 AUROC 差(A − B)的**配對** bootstrap 區間。

    配對(每次重抽用同一組索引評兩個分數)才能消掉「這批樣本剛好好判」
    的共同變異,區間會比兩個獨立區間相減窄得多。
    """
    a, b, y = np.asarray(s_a, float), np.asarray(s_b, float), np.asarray(labels).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        d[i] = auroc(a[idx], y[idx]) - auroc(b[idx], y[idx])
    return {"delta": auroc(a, y) - auroc(b, y),
            "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)),
            "sd": float(d.std(ddof=1)),
            "win_rate": float((d > 0).mean())}


def stratified_auroc(scores, labels, control, n_bins: int = 5,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """**控制 `control` 之後**的 AUROC:按 control 的分位數分箱,箱內算 AUROC。

    加權方式用箱內的配對數 n_pos·n_neg —— 那正是該箱對整體 Mann–Whitney
    統計量的貢獻權重,所以在「control 與標籤無關」的極限下,
    分層 AUROC 會回到原始 AUROC。

    箱內若某一類為空就跳過該箱(並回報跳過數)—— 那代表在該複雜度區間
    根本沒有可比較的對照,硬算會得到 nan 或以極少數樣本主導的假數字。
    """
    s, y, c = (np.asarray(scores, float), np.asarray(labels).astype(int),
               np.asarray(control, float))
    edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins, skipped = [], 0
    for k in range(n_bins):
        m = (c >= edges[k]) & (c < edges[k + 1])
        if m.sum() < 2 or (y[m] == 1).sum() == 0 or (y[m] == 0).sum() == 0:
            skipped += 1
            continue
        w = int((y[m] == 1).sum()) * int((y[m] == 0).sum())
        bins.append({"k": k, "n": int(m.sum()), "n_pos": int((y[m] == 1).sum()),
                     "auroc": auroc(s[m], y[m]), "weight": w,
                     "control_range": [float(edges[k]), float(edges[k + 1])]})
    if not bins:
        return {"auroc": float("nan"), "bins": [], "n_bins_skipped": skipped}

    W = sum(b["weight"] for b in bins)
    pooled = sum(b["auroc"] * b["weight"] for b in bins) / W

    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        num = den = 0.0
        for k in range(n_bins):
            m = (c[idx] >= edges[k]) & (c[idx] < edges[k + 1])
            if m.sum() < 2 or (y[idx][m] == 1).sum() == 0 or (y[idx][m] == 0).sum() == 0:
                continue
            w = int((y[idx][m] == 1).sum()) * int((y[idx][m] == 0).sum())
            num += auroc(s[idx][m], y[idx][m]) * w
            den += w
        if den > 0:
            boot.append(num / den)
    boot = np.asarray(boot)
    return {"auroc": float(pooled), "bins": bins, "n_bins_skipped": skipped,
            "lo95": float(np.percentile(boot, 2.5)) if len(boot) else float("nan"),
            "hi95": float(np.percentile(boot, 97.5)) if len(boot) else float("nan")}


def separation(a: np.ndarray, b: np.ndarray) -> dict:
    """兩個分佈的可分離程度 —— 通關標準 1 說「直方圖可分離」,這是它的量化。

    Cohen's d 假設同變異數,而 ELBO 在兩組的散佈常常差很多,
    所以同時報 **重疊係數**(直方圖交集面積,不依賴任何分佈假設)。
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / max(len(a) + len(b) - 2, 1))
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, 61)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    overlap = float(np.minimum(pa, pb).sum() * (edges[1] - edges[0]))
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1)),
            "cohens_d": float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan"),
            "overlap_coefficient": overlap,
            "median_gap": float(np.median(a) - np.median(b))}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相關 —— ELBO 與複雜度的關係是單調但非線性的,用秩才對。"""
    def rank(v):
        v = np.asarray(v, float)
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def png_bits(x: np.ndarray) -> np.ndarray:
    """每張影像用 PNG 無損壓縮後的**位元數** —— 通用壓縮器的複雜度代理。

    這是 Serrà et al. (2020) 的做法:深度生成模型的 log-likelihood 被
    影像複雜度支配,而一個通用壓縮器 L(x) 是 log-likelihood 的
    「無模型」上界估計。用 S = −ELBO − L(x) 把複雜度成分扣掉,
    剩下的才是模型真正貢獻的訊息(見 experiments.complexity_corrected)。

    用 PNG 而不是 JPEG:必須無損,否則扣掉的是「有損壓縮的失真」而非資訊量。
    """
    from PIL import Image

    out = np.empty(len(x), dtype=np.float64)
    for i, img in enumerate(x):
        arr = np.clip(img[0] * 255.0, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, mode="L").save(buf, format="PNG", optimize=True)
        out[i] = buf.getbuffer().nbytes * 8.0
    return out
