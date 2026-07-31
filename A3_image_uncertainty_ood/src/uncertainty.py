"""
A3 · 不確定性分解（計劃書主題六）與 OOD 分數。

熵分解
------
給定一組預測分佈樣本 $\\{p_t(y|x)\\}_{t=1}^T$（來自 MC Dropout 的不同 mask，
或 Deep Ensemble 的不同成員）：

    total      = H[ E_t[p_t] ]              整體不確定性（預測平均分佈的熵）
    aleatoric  = E_t[ H[p_t] ]              各次熵的平均：資料本身的模糊程度
    epistemic  = total − aleatoric          = 互資訊 I(y; w)

> 🔑 分類問題的分解用**熵**而不是變異數。`epistemic = 總熵 − 平均熵` 這個量
> 在資訊理論裡就是**互資訊**——「知道權重 $w$ 能減少多少對 $y$ 的不確定性」。
> 它 ≥ 0（Jensen 不等式），且只有在所有 $p_t$ 完全相同時為 0。

直覺：
- 一張本質模糊的影像（兩類都像）→ 每個 $p_t$ 都接近均勻 → 平均熵高、彼此一致
  → **aleatoric 高、epistemic 低**。再多資料也救不了。
- 一張沒見過的影像 → 每個 $p_t$ 都很尖銳但**指向不同類別** → 平均熵低、
  平均後的分佈很平 → **epistemic 高**。這是「我不知道」，可以靠更多資料改善。

這個直覺是否成立**必須驗證**，不能假設 —— 見 `experiments.decomposition_validation`。

三種取得預測分佈的方式
----------------------
1. **單一模型 softmax**：T=1，退化情形。`epistemic ≡ 0`，只能用 max-prob 或熵當分數。
   這是最常見的 baseline，也是計劃書要求的對照。
2. **MC Dropout**：同一組權重、T 個不同 dropout mask。便宜（一次訓練），
   但所有樣本共享同一個權重鄰域，多樣性受 mask 能表達的範圍限制。
3. **Deep Ensemble**：M 個獨立訓練的網路。貴 M 倍，但成員可以落在
   完全不同的損失盆地，多樣性通常大得多。
"""
from __future__ import annotations

import numpy as np
import torch

from model import enable_dropout_only

EPS = 1e-12


# ---------------------------------------------------------------------------
# 熵分解
# ---------------------------------------------------------------------------


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """H[p] = −Σ p log p，nat 為單位。"""
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def decompose(probs: np.ndarray) -> dict:
    """probs: (T, N, C) —— T 個預測分佈樣本。回傳逐樣本的三個量 (N,)。

    `epistemic` 理論上非負；數值上可能出現 −1e-16 量級的值，clip 到 0
    並回報 clip 前的最小值，讓數值問題可見而不是被藏起來。
    """
    probs = np.asarray(probs, dtype=np.float64)
    mean_p = probs.mean(axis=0)                       # (N, C)
    total = entropy(mean_p)                           # H[E[p]]
    aleatoric = entropy(probs, axis=-1).mean(axis=0)  # E[H[p]]
    epistemic_raw = total - aleatoric
    return {
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": np.clip(epistemic_raw, 0.0, None),
        "epistemic_min_raw": float(epistemic_raw.min()),
        "mean_prob": mean_p,
        "max_prob": mean_p.max(axis=-1),
        "pred": mean_p.argmax(axis=-1),
        "T": int(probs.shape[0]),
    }


# ---------------------------------------------------------------------------
# 三種預測分佈的產生方式
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_deterministic(model, x: np.ndarray, dev, batch_size: int = 512) -> np.ndarray:
    """Dropout 關閉的單次前向 → (1, N, C)。單一模型 baseline。"""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(dev)
        out.append(model(xb).softmax(-1).cpu().numpy())
    return np.concatenate(out)[None, ...]


@torch.no_grad()
def predict_mc_dropout(model, x: np.ndarray, dev, T: int = 50, batch_size: int = 512,
                       seed: int = 0) -> np.ndarray:
    """MC Dropout → (T, N, C)。

    用 `enable_dropout_only` 而不是 `model.train()`：只要 Dropout 的隨機性，
    不要任何依賴 batch 的正規化行為（見 model.py 的說明）。
    """
    enable_dropout_only(model)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probs = np.empty((T, len(x), _n_classes(model)), dtype=np.float32)
    for t in range(T):
        out = []
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(dev)
            out.append(model(xb).softmax(-1).cpu().numpy())
        probs[t] = np.concatenate(out)
    model.eval()
    return probs


@torch.no_grad()
def predict_ensemble(models, x: np.ndarray, dev, batch_size: int = 512) -> np.ndarray:
    """Deep Ensemble → (M, N, C)。每個成員 Dropout 關閉，各出一個確定性預測。

    Dropout 在這裡只是訓練期的正則化；多樣性來自不同的權重解，
    不需要（也不該）在推論時再加 dropout 噪聲——那會把兩種機制混在一起，
    使「Ensemble vs MC Dropout」的比較不再乾淨。
    """
    return np.concatenate([predict_deterministic(m, x, dev, batch_size) for m in models])


def _n_classes(model) -> int:
    for mod in reversed(list(model.modules())):
        if isinstance(mod, torch.nn.Linear):
            return mod.out_features
    raise RuntimeError("找不到輸出層")


# ---------------------------------------------------------------------------
# OOD 分數
# ---------------------------------------------------------------------------


def ood_scores(dec: dict) -> dict:
    """從分解結果取出可當 OOD 分數的量（越大越像 OOD）。

    `max_prob` 要取負號：in-distribution 的樣本 max-prob 高。
    """
    return {
        "epistemic": dec["epistemic"],
        "total_entropy": dec["total"],
        "aleatoric": dec["aleatoric"],
        "neg_max_prob": -dec["max_prob"],
    }


def trivial_ood_scores(image_set) -> dict:
    """**誠實檢查**：只用低階影像統計當 OOD 分數，完全不看模型。

    平均亮度、對比、L2 norm、通道間標準差 —— 四個最笨的量。
    如果它們的 AUROC 就已經很高，那「模型的 epistemic 不確定性偵測到了 OOD」
    這個說法必須大幅打折：偵測到的可能只是「這批影像的像素分佈不一樣」。

    大部分作品集不做這個對照，於是把資料集之間的低階差異當成
    模型的語意理解能力。本專案的第 2、3 張圖就是為了讓這件事無法被含混過去。

    分數方向未知（例如 OOD 可能更亮也可能更暗），所以在算 AUROC 時
    取 max(auc, 1−auc) —— 相當於允許 baseline 事後選擇最有利的方向，
    這對 baseline 是**有利**的設定，讓對照更嚴格而非更寬鬆。
    """
    from data import raw_pixel_stats

    st = raw_pixel_stats(image_set)
    return {f"pixel_{k}": v for k, v in st.items()}
