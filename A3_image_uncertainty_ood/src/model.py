"""
A3 · 帶 Dropout 的 CNN。

為什麼是 GroupNorm 而不是 BatchNorm
------------------------------------
MC Dropout 的做法是「測試時保持 Dropout 開啟」，最直接的寫法是 `model.train()`。
但 `train()` 是全域開關，它會**連帶**把 BatchNorm 也切到訓練模式，帶來兩個
安靜但嚴重的錯誤：

1. BatchNorm 在訓練模式用**當前 batch** 的統計量做正規化 → 一個樣本的預測
   會依賴同一個 batch 裡剛好有哪些其他樣本。預測不再是單樣本的函數，
   而「同一張影像餵兩次得到不同答案」會被誤認為是模型的 epistemic 不確定性。
2. 每次 MC 前向傳播都會更新 running statistics → 跑 50 次 MC Dropout
   等於偷偷用測試資料訓練了 50 步正規化層。**這是測試集洩漏。**

GroupNorm 完全不依賴 batch，也沒有 running statistics，
所以 train/eval 兩個模式下的行為相同 —— 這讓「只開 Dropout」變得安全。

本模組另外提供 `enable_dropout_only()`：`model.eval()` 之後只把 Dropout
子模組切回訓練模式。用了 GroupNorm 之後這與 `model.train()` 等價，
但寫明意圖比依賴「剛好沒有 BatchNorm」更穩固 —— 之後若有人加了 BatchNorm，
這個函式仍然是對的。`sanity_check_mode_invariance()` 會驗證這件事。
"""
from __future__ import annotations

import copy
import os

# ⚠️ 必須在 import torch **之前**設定：`torch.use_deterministic_algorithms(True)`
# 之下，部分 cuBLAS 操作需要這個 workspace 設定才有決定性實作，否則直接報錯。
# CUDA context 一旦建立就讀不到這個變數了，所以放在模組最頂端。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn


class DropoutCNN(nn.Module):
    """小型 CNN：3 個 conv block + FC head，conv 用 Dropout2d、head 用 Dropout。

    Dropout2d（整個 feature map 一起丟）在卷積層比逐元素 Dropout 更有意義：
    相鄰像素高度相關，逐元素丟掉幾乎不減少資訊，等於沒有正則化效果，
    也就不會產生有意義的函數空間後驗樣本。
    """

    def __init__(self, n_classes: int, in_ch: int = 3, width: int = 32,
                 p_conv: float = 0.2, p_fc: float = 0.5, groups: int = 8):
        super().__init__()
        w = width

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(min(groups, cout), cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(min(groups, cout), cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(p_conv),
            )

        self.features = nn.Sequential(
            block(in_ch, w),       # 28 → 14
            block(w, w * 2),       # 14 → 7
            block(w * 2, w * 4),   # 7 → 3
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p_fc),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def enable_dropout_only(model: nn.Module) -> None:
    """`eval()` 之後只把 Dropout 子模組切回訓練模式 —— MC Dropout 的正確開法。

    比 `model.train()` 精確：明確表達「我只要 Dropout 的隨機性，
    不要任何依賴 batch 的正規化行為」。
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout1d, nn.AlphaDropout)):
            m.train()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def sanity_check_mode_invariance(model: nn.Module, x: torch.Tensor,
                                 tol: float = 1e-6, batch_tol: float = 1e-5) -> dict:
    """驗證本模型沒有 BatchNorm 式的兩個病徵，而不是口頭聲稱。

    檢查一「模式不變性」：關掉 Dropout 後，`train()` 與 `eval()` 的 logits 必須相同。
        BatchNorm 會違反它（train 用 batch 統計、eval 用 running 統計）。

    檢查二「batch 組成不變性」：一次餵 64 張，與逐張餵，單張的 logits 必須相同。
        BatchNorm 會違反它（預測依賴同 batch 的其他樣本）。

    ⚠️ **檢查二必須在 CPU 上做。** 在 CUDA 上 cuDNN 會依 batch size 選不同的
    卷積演算法，浮點累加順序跟著變，即使模型完全沒有 batch 依賴也會出現
    ~1e-4 的差異（實測 CUDA 1.7e-4 vs CPU 6.3e-7）。那是數值噪聲，不是
    結構問題；在 GPU 上跑這個檢查只會測到 cuDNN 的實作細節。
    """
    drops = [m for m in model.modules()
             if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout1d))]
    saved = [m.p for m in drops]
    for m in drops:
        m.p = 0.0
    try:
        # 檢查一：同一個 batch、同一個裝置，只切模式
        model.eval()
        out_eval = model(x).clone()
        model.train()
        out_train = model(x).clone()
        max_diff = (out_eval - out_train).abs().max().item()

        # 檢查二：搬到 CPU，避開 cuDNN 的 batch-size-dependent kernel 選擇
        cpu_model = copy.deepcopy(model).to("cpu").train()
        xc = x.detach().to("cpu")
        k = min(8, len(xc))
        out_full = cpu_model(xc)[:k]
        out_single = torch.cat([cpu_model(xc[i : i + 1]) for i in range(k)])
        max_batch_diff = (out_full - out_single).abs().max().item()
    finally:
        for m, p in zip(drops, saved):
            m.p = p
        model.eval()

    has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    return {
        "max_train_eval_diff": max_diff,
        "max_batch_composition_diff_cpu": max_batch_diff,
        "passed": bool(max_diff <= tol and max_batch_diff <= batch_tol and not has_bn),
        "n_dropout_layers": len(drops),
        "has_batchnorm": has_bn,
        "n_groupnorm": sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm)),
    }


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """設定隨機種子，並（預設）要求 cuDNN 使用決定性演算法。

    ⚠️ 只設 seed **不足以**讓訓練可重現，這是實測踩出來的：

    只設 seed 時，兩次「相同」的完整執行給出的 PneumoniaMNIST ensemble
    測試準確率是 0.885 與 0.901。原因是 cuDNN 預設自動挑選最快的卷積演算法，
    其中部分非決定性（浮點歸約順序隨執行變動），微小差異在 30 個 epoch 裡
    累積後改變了 early stopping 選到哪一個 epoch —— 對一個承諾「一鍵重現」
    的專案，這是不能接受的。

    加上這裡的三個設定（`cudnn.deterministic`、`cudnn.benchmark=False`、
    `use_deterministic_algorithms`，搭配模組頂端的 `CUBLAS_WORKSPACE_CONFIG`）
    之後，連續兩次完整執行的**每一個**數字完全相同 —— 準確率、AUROC、
    選擇性比值、cost-benefit 掃描、risk–coverage 全部逐位一致。

    誠實聲明：我沒有逐一隔離這三個旗標各自的貢獻，只驗證了整組設定有效。
    `use_deterministic_algorithms` 需要 `CUBLAS_WORKSPACE_CONFIG`，
    且會讓沒有決定性實作的 op 直接報錯（本專案的模型不觸發任何一個）。

    代價是訓練略慢（本專案模型小，實測影響可忽略）。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
