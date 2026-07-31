"""
A1 · 決策理論：從機率到行動（計劃書主題七第一卡）。

損失矩陣（治療 vs 不治療）：
                 有病 y=1      沒病 y=0
    治療           0            C_FP      ← 誤診：不必要的治療
    不治療        C_FN           0        ← 漏診：錯過疾病（醫療上通常 C_FN ≫ C_FP）

給定 P(有病)=p：
    E[loss | 治療]   = (1−p)·C_FP
    E[loss | 不治療] =    p ·C_FN
    → 治療 iff (1−p)C_FP < p·C_FN  iff  p > C_FP/(C_FP+C_FN) =: p*

所以最優門檻 p* 由**成本比**決定，不是預設的 0.5。C_FN/C_FP 越大，p* 越小
（漏診越貴 → 越傾向治療 → 門檻越低）。
"""
from __future__ import annotations

import numpy as np


def optimal_threshold(c_fp: float, c_fn: float) -> float:
    """兩行動的最優門檻 p* = C_FP / (C_FP + C_FN)。"""
    return c_fp / (c_fp + c_fn)


def expected_loss_treat(p, c_fp):
    return (1 - np.asarray(p, float)) * c_fp


def expected_loss_notreat(p, c_fn):
    return np.asarray(p, float) * c_fn


def reject_region(c_fp: float, c_fn: float, c_reject: float):
    """棄權（轉診/再檢查，固定成本 C_reject）被選中的 p 區間 (lo, hi)；若無則 None。
    棄權 iff C_reject 同時小於治療與不治療的期望損失。"""
    lo = c_reject / c_fn            # p 需大於此，不治療才夠貴
    hi = 1 - c_reject / c_fp        # p 需小於此，治療才夠貴
    return (lo, hi) if lo < hi else None


def decide(p, c_fp, c_fn, c_reject: float | None = None):
    """回傳每個病人的行動：0=不治療, 1=治療, 2=棄權（若提供 C_reject）。"""
    p = np.asarray(p, float)
    lt = expected_loss_treat(p, c_fp)
    ln = expected_loss_notreat(p, c_fn)
    if c_reject is None:
        return (lt < ln).astype(int)
    lr = np.full_like(p, c_reject)
    return np.stack([ln, lt, lr], 0).argmin(0)     # 0=不治療,1=治療,2=棄權


def realized_loss(y_true, p, c_fp, c_fn, threshold: float) -> float:
    """在測試集上，用固定門檻決策時的平均實際損失（用真實標籤）。"""
    y_true = np.asarray(y_true)
    treat = np.asarray(p) > threshold
    loss = np.where(treat & (y_true == 0), c_fp,
                    np.where(~treat & (y_true == 1), c_fn, 0.0))
    return float(loss.mean())


def realized_loss_with_reject(y_true, p, c_fp, c_fn, c_reject: float):
    """含棄權選項時的平均實際損失，與落入棄權區的病人比例。"""
    y_true = np.asarray(y_true)
    act = decide(p, c_fp, c_fn, c_reject)
    loss = np.zeros(len(y_true))
    loss[(act == 1) & (y_true == 0)] = c_fp
    loss[(act == 0) & (y_true == 1)] = c_fn
    loss[act == 2] = c_reject
    return float(loss.mean()), float((act == 2).mean())
