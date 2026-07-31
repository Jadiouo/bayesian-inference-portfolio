"""
S1 · 資料載入 —— PneumoniaMNIST 的 normal / pneumonia 切分。

異常偵測的設定
--------------
只用 **normal 胸片**訓練 VAE，pneumonia 完全不給模型看。測試時：

    in-distribution : test split 的 normal    (234 張)
    anomaly         : test split 的 pneumonia (390 張)

這比「換一個資料集當 OOD」（A3 專案做的）更難也更貼近臨床：
兩者是**同一種影像、同一台機器、同一個解剖部位**，差別只在病灶。
低階像素統計幾乎一樣，所以模型必須真的學到「正常肺部長什麼樣」。

⚠️ 訓練集只有 **1214 張** normal。這對 VAE 偏少，是本專案最主要的限制
（見 README）。MedMNIST 的設計目的是輕量基準，不是高保真醫學影像。

資料重用 A3 已下載的快取（`data/A_medical/medmnist/`），
骨架也建議「可與 A3 共用 MedMNIST 載入程式」。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

NORMAL, PNEUMONIA = 0, 1


@dataclass
class ImageSet:
    """一組影像。x 形狀 (N,1,28,28)，值域 [0,1]。"""

    x: np.ndarray
    name: str
    y: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.x)

    def summary(self) -> str:
        return (f"{self.name}: n={self.n}, x{self.x.shape} "
                f"in [{self.x.min():.3f},{self.x.max():.3f}], "
                f"mean={self.x.mean():.3f}")


def _load_split(split: str, root: str):
    import medmnist

    ds = medmnist.PneumoniaMNIST(split=split, download=True, as_rgb=False, root=root)
    x = ds.imgs.astype(np.float32) / 255.0          # (N,28,28) in [0,1]
    y = np.asarray(ds.labels, dtype=np.int64).ravel()
    return x[:, None, :, :], y                       # (N,1,28,28)


def load(data_dir: str) -> dict:
    """回傳 train/val（只有 normal）與 test 的 normal / pneumonia。

    值域保持 **[0,1]** 而不是 [-1,1]：VAE 的重建似然用連續 Bernoulli /
    Gaussian 都以 [0,1] 為自然支撐，而 ELBO 的重建項數值可比性依賴這件事。
    """
    root = os.path.join(data_dir, "medmnist")
    os.makedirs(root, exist_ok=True)

    xtr, ytr = _load_split("train", root)
    xva, yva = _load_split("val", root)
    xte, yte = _load_split("test", root)

    return {
        "train_normal": ImageSet(xtr[ytr == NORMAL], "train[normal]"),
        "val_normal": ImageSet(xva[yva == NORMAL], "val[normal]"),
        "test_normal": ImageSet(xte[yte == NORMAL], "test[normal]"),
        "test_anomaly": ImageSet(xte[yte == PNEUMONIA], "test[pneumonia]"),
        # 保留完整 test 供需要標籤的評估用（1 = 異常）
        "test_all": ImageSet(xte, "test[all]", (yte == PNEUMONIA).astype(np.int64)),
    }


def complexity_scores(s: ImageSet) -> dict:
    """影像「複雜度」的低階統計 —— 誠實檢查的原料。

    VAE 異常偵測有一個已知的病：**重建誤差偏向簡單影像**。
    一張平坦、低對比的異常影像可能重建得很好（ELBO 高、被判為正常），
    而一張紋理豐富的正常影像 ELBO 反而低。若這些不看模型的統計量
    就能達到接近 VAE 的 AUROC，那「VAE 偵測到了病灶」這個說法
    必須大幅打折 —— 它偵測到的可能只是「這張圖比較複雜」。

    沿用 A3 專案的 trivial-baseline 精神：沒有這個對照，
    就無法區分「模型學到語意」與「資料集之間有低階差異」。
    """
    x = s.x[:, 0]                                    # (N,28,28)
    flat = x.reshape(len(x), -1)

    # 像素強度的直方圖熵（32 bins）—— 影像內容的豐富程度
    ent = np.empty(len(x))
    for i, row in enumerate(flat):
        h, _ = np.histogram(row, bins=32, range=(0.0, 1.0), density=False)
        p = h / max(h.sum(), 1)
        p = p[p > 0]
        ent[i] = -(p * np.log(p)).sum()

    gy = np.diff(x, axis=1)
    gx = np.diff(x, axis=2)
    grad_energy = (gy**2).sum(axis=(1, 2)) + (gx**2).sum(axis=(1, 2))

    return {
        "pixel_entropy": ent,
        "gradient_energy": grad_energy,
        "l2_norm": np.sqrt((flat**2).sum(axis=1)),
        "mean_intensity": flat.mean(axis=1),
        "std_intensity": flat.std(axis=1),
    }
