"""
A3 · 訓練迴圈與 Deep Ensemble。

Deep Ensemble（Lakshminarayanan et al. 2017）的做法刻意簡單：
**M 個獨立隨機初始化 + 獨立的資料順序**，各自訓練到收斂，預測時平均。
不需要 bagging（子抽樣資料反而通常變差），因為多樣性主要來自
非凸損失函數的不同局部解 —— 隨機初始化與 SGD 噪聲就足夠。

這是本專案的重要對照組：它是「函數空間後驗」的一個粗糙但有效的近似，
而 MC Dropout 是另一個。計劃書預期 Ensemble 通常勝出，理由是
MC Dropout 的所有樣本共享同一組權重的鄰域，多樣性受限於 dropout mask
所能表達的範圍；Ensemble 的成員可以落在完全不同的損失盆地。
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import DropoutCNN, count_params, device, set_seed


def make_loader(s, batch_size: int = 128, shuffle: bool = False, seed: int | None = None):
    x = torch.from_numpy(s.x)
    y = torch.from_numpy(s.y)
    g = None
    if shuffle and seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      generator=g, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate_plain(model: nn.Module, loader, dev) -> dict:
    """確定性評估（Dropout 關閉）—— 用來監控訓練與報告基準準確率。"""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(-1) == yb).sum().item()
        total += len(yb)
    return {"acc": correct / total, "loss": loss_sum / total}


def train_one(train_set, val_set, n_classes: int, seed: int = 0, epochs: int = 30,
              lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 128,
              width: int = 32, p_conv: float = 0.2, p_fc: float = 0.5,
              patience: int = 8, verbose: bool = False, class_weight=None):
    """訓練單一模型，回傳 (model, history)。

    以 val accuracy 選最佳權重（early stopping）。這對本專案特別重要：
    不確定性估計的品質對過擬合很敏感 —— 過擬合的模型在訓練分佈上過度自信，
    epistemic 會被系統性低估，OOD 偵測跟著失效。
    """
    dev = device()
    set_seed(seed)
    model = DropoutCNN(n_classes=n_classes, width=width, p_conv=p_conv, p_fc=p_fc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_loader = make_loader(train_set, batch_size, shuffle=True, seed=seed)
    va_loader = make_loader(val_set, 512)
    cw = None if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32, device=dev)

    best = {"acc": -1.0, "state": None, "epoch": -1}
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=cw)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(yb)
            n += len(yb)
        sched.step()
        va = evaluate_plain(model, va_loader, dev)
        hist.append({"epoch": ep, "train_loss": run_loss / n, "val_acc": va["acc"],
                     "val_loss": va["loss"]})
        if va["acc"] > best["acc"]:
            best = {"acc": va["acc"], "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if verbose:
            print(f"  ep{ep:02d} train_loss={run_loss / n:.4f} val_acc={va['acc']:.4f}")
        if ep - best["epoch"] >= patience:
            break

    model.load_state_dict(best["state"])
    model.eval()
    return model, {"history": hist, "best_epoch": best["epoch"], "best_val_acc": best["acc"],
                   "seconds": round(time.time() - t0, 1), "n_params": count_params(model),
                   "epochs_run": len(hist)}


def train_ensemble(train_set, val_set, n_classes: int, m: int = 5, base_seed: int = 100,
                   verbose: bool = False, **kw):
    """訓練 M 個成員。每個成員拿到不同的初始化 seed **與**不同的資料順序。

    回傳 (models, infos)。第一個成員同時充當「單一模型」基準與 MC Dropout 的載體，
    這樣三種方法的比較是在**同一個訓練好的網路**上進行的（Ensemble 除外，
    它本質上就需要多個網路）—— 避免「MC Dropout 輸是因為它那個網路剛好比較差」。
    """
    models, infos = [], []
    for i in range(m):
        model, info = train_one(train_set, val_set, n_classes, seed=base_seed + i, **kw)
        info["member"] = i
        info["seed"] = base_seed + i
        models.append(model)
        infos.append(info)
        if verbose:
            print(f"  member {i}: val_acc={info['best_val_acc']:.4f} "
                  f"({info['seconds']}s, best epoch {info['best_epoch']})")
    return models, infos


def class_weights_from(train_set, n_classes: int) -> np.ndarray:
    """類別權重（inverse frequency，正規化成平均 1）。

    PneumoniaMNIST 的兩類不平衡（約 1:3）。不補償的話模型會偏向多數類，
    而那會扭曲不確定性：少數類樣本的 aleatoric 被高估，
    看起來像「資料本身模糊」，其實是先驗被類別頻率帶偏。
    """
    counts = np.bincount(train_set.y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return w / w.mean()
