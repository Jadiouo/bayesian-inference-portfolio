"""
A3 · 評估：OOD 偵測（AUROC）、選擇性預測（risk–coverage）、校準（ECE）。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------------------------
# OOD 偵測
# ---------------------------------------------------------------------------


def auroc(score_in: np.ndarray, score_ood: np.ndarray, allow_flip: bool = False) -> dict:
    """AUROC：把 in-distribution 與 OOD 分開的能力。分數越大越像 OOD。

    `allow_flip=True` 時回報 max(auc, 1−auc)，用於方向未知的 trivial baseline
    （例如 OOD 影像可能更亮也可能更暗）。這對 baseline 是**有利**的設定
    ——讓對照更嚴格，避免用「方向猜錯」來讓 baseline 看起來很差。
    """
    y = np.concatenate([np.zeros(len(score_in)), np.ones(len(score_ood))])
    s = np.concatenate([score_in, score_ood])
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    a = float(roc_auc_score(y, s))
    flipped = False
    if allow_flip and a < 0.5:
        a, flipped = 1.0 - a, True
    ap = float(average_precision_score(y, -s if flipped else s))
    return {"auroc": a, "auprc": ap, "flipped": flipped,
            "n_in": int(len(score_in)), "n_ood": int(len(score_ood))}


def auroc_bootstrap(score_in, score_ood, n_boot: int = 300, seed: int = 0,
                    allow_flip: bool = False) -> dict:
    """AUROC 的 bootstrap 區間。

    需要它是因為 OOD 集大小差異很大（PathMNIST 留出類別只有 1654 張），
    而 0.02 的 AUROC 差距在小樣本上可能純粹是噪聲。
    """
    rng = np.random.default_rng(seed)
    si, so = np.asarray(score_in), np.asarray(score_ood)
    vals = []
    for _ in range(n_boot):
        bi = rng.choice(len(si), len(si), replace=True)
        bo = rng.choice(len(so), len(so), replace=True)
        try:
            vals.append(auroc(si[bi], so[bo], allow_flip=allow_flip)["auroc"])
        except ValueError:
            continue
    v = np.asarray(vals)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1)),
            "lo95": float(np.percentile(v, 2.5)), "hi95": float(np.percentile(v, 97.5))}


def paired_auroc_diff(score_in_a, score_ood_a, score_in_b, score_ood_b,
                      n_boot: int = 300, seed: int = 0) -> dict:
    """兩種方法 AUROC 差距的**配對** bootstrap。

    配對很重要：兩種方法在同一批重抽樣本上一起評估，共同的抽樣噪聲被抵消。
    各自算 CI 再看是否重疊會嚴重低估區分能力。
    """
    rng = np.random.default_rng(seed)
    sia, soa = np.asarray(score_in_a), np.asarray(score_ood_a)
    sib, sob = np.asarray(score_in_b), np.asarray(score_ood_b)
    diffs = []
    for _ in range(n_boot):
        bi = rng.choice(len(sia), len(sia), replace=True)
        bo = rng.choice(len(soa), len(soa), replace=True)
        try:
            a = auroc(sia[bi], soa[bo])["auroc"]
            b = auroc(sib[bi], sob[bo])["auroc"]
        except ValueError:
            continue
        diffs.append(a - b)
    d = np.asarray(diffs)
    return {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1)),
            "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "p_a_better": float((d > 0).mean())}


# ---------------------------------------------------------------------------
# 選擇性預測（接 A1 的決策層）
# ---------------------------------------------------------------------------


def risk_coverage(uncertainty: np.ndarray, correct: np.ndarray, n_points: int = 101) -> dict:
    """risk–coverage 曲線：「放棄最不確定的 X% 樣本後，剩下的準確率是多少？」

    這是 A1 的棄權選項在影像上的版本：epistemic 超過門檻 → 轉人工判讀。
    臨床上這才是不確定性的用途——不是報告一個漂亮的數字，
    而是決定哪些病例需要人看。

    回傳 coverage（保留比例）、accuracy（保留樣本的準確率）、risk（1−accuracy），
    以及 AURC（risk 曲線下面積，越小越好）。
    """
    u = np.asarray(uncertainty, dtype=float)
    c = np.asarray(correct, dtype=bool)
    order = np.argsort(u)  # 最確定的排前面
    c_sorted = c[order]
    n = len(u)

    coverages = np.linspace(1.0 / n, 1.0, n_points)
    accs, risks = [], []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        accs.append(c_sorted[:k].mean())
        risks.append(1.0 - c_sorted[:k].mean())
    accs, risks = np.asarray(accs), np.asarray(risks)
    return {"coverage": coverages, "accuracy": accs, "risk": risks,
            "aurc": float(np.trapezoid(risks, coverages)),
            "acc_at_100": float(accs[-1]),
            "acc_at_80": float(accs[np.argmin(np.abs(coverages - 0.8))]),
            "acc_at_50": float(accs[np.argmin(np.abs(coverages - 0.5))])}


def oracle_aurc(correct: np.ndarray) -> float:
    """完美排序（所有錯誤都排在最後）的 AURC —— risk–coverage 的理論下界。

    報告它是因為單看 AURC 無法判斷好壞：一個準確率 99% 的模型，
    即使排序完全隨機，AURC 也很小。要看的是實際 AURC 與 oracle 的差距。
    """
    c = np.sort(np.asarray(correct, dtype=bool))[::-1]  # 全對排前面
    n = len(c)
    cov = np.linspace(1.0 / n, 1.0, 101)
    risks = [1.0 - c[: max(1, int(round(v * n)))].mean() for v in cov]
    return float(np.trapezoid(risks, cov))


# ---------------------------------------------------------------------------
# 校準（接 A1 的 ECE）
# ---------------------------------------------------------------------------


def ece(probs_mean: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """Expected Calibration Error（等寬分箱）。

    A1 專案已經做過表格資料的校準；這裡放進來是因為 OOD 偵測與校準是
    兩件**不同**的事，而它們常被混為一談：一個模型可以在 in-distribution
    上校準得很好，卻完全偵測不到 OOD（反之亦然）。分開報告才看得出來。
    """
    conf = np.asarray(probs_mean).max(axis=-1)
    pred = np.asarray(probs_mean).argmax(axis=-1)
    correct = (pred == np.asarray(y)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)

    total = 0.0
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append({"bin": b, "n": 0, "conf": np.nan, "acc": np.nan})
            continue
        c_mean, a_mean = conf[m].mean(), correct[m].mean()
        total += (m.sum() / len(conf)) * abs(a_mean - c_mean)
        rows.append({"bin": b, "n": int(m.sum()), "conf": float(c_mean), "acc": float(a_mean)})
    return {"ece": float(total), "bins": rows,
            "accuracy": float(correct.mean()), "mean_conf": float(conf.mean())}


def accuracy(dec: dict, y: np.ndarray) -> float:
    return float((dec["pred"] == np.asarray(y)).mean())
