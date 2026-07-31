"""
A3 · 資料載入 —— MedMNIST（in-distribution）與三組難度遞增的 OOD 來源。

三組 OOD 實驗（計劃書 §2 專案 A3）
-----------------------------------
| 訓練                     | OOD 測試              | 難度 |
|--------------------------|-----------------------|------|
| PneumoniaMNIST (2 類)    | FashionMNIST          | 易   |
| PneumoniaMNIST (2 類)    | DermaMNIST            | 中   |
| PathMNIST 留出 2 類 (7 類)| 留出的那 2 類         | 難   |

**通道決策：全部統一成 3 通道（`as_rgb=True`）。**
另一個選項是把模型做成 1 通道、把 DermaMNIST 轉灰階，但那會丟掉顏色 ——
而顏色正是皮膚病灶影像最關鍵的特徵，等於人為地把「中等難度」那組閹掉。
統一 3 通道的代價是 PneumoniaMNIST 的灰階被複製成三份（冗餘但無害），
好處是 OOD 情境保持真實。

⚠️ 這個決策有一個誠實的副作用：DermaMNIST 的色彩統計與胸片差很多，
所以高 AUROC 可能來自**顏色**而不是模型「理解」了語意。
這正是 `uncertainty.trivial_ood_scores` 要檢查的事。

資料快取到 data/A_medical/（MedMNIST 到 medmnist/ 子夾、FashionMNIST 到 fashion/），
不進版控（見根目錄 .gitignore）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

MEDMNIST_FLAGS = ("pneumoniamnist", "dermamnist", "pathmnist", "bloodmnist")


@dataclass
class ImageSet:
    """一組影像：x 已正規化到 [-1,1]，形狀 (N,3,28,28)；y 是整數標籤（OOD 集為 -1）。"""

    x: np.ndarray
    y: np.ndarray
    name: str
    class_names: list[str] | None = None

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def n_classes(self) -> int:
        return 0 if self.y.min() < 0 else int(self.y.max()) + 1

    def subset(self, idx) -> "ImageSet":
        return ImageSet(self.x[idx], self.y[idx], self.name, self.class_names)

    def summary(self) -> str:
        cls = f", {self.n_classes} classes" if self.y.min() >= 0 else " (OOD, no labels)"
        return f"{self.name}: n={self.n}{cls}, x{self.x.shape} in [{self.x.min():.2f},{self.x.max():.2f}]"


def _normalize(x_uint8: np.ndarray) -> np.ndarray:
    """(N,H,W,C) uint8 → (N,C,H,W) float32 in [-1,1]。"""
    x = x_uint8.astype(np.float32) / 255.0
    if x.ndim == 3:  # (N,H,W) 灰階
        x = x[..., None]
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    x = np.transpose(x, (0, 3, 1, 2))
    return (x - 0.5) / 0.5


def load_medmnist(flag: str, split: str, data_dir: str) -> ImageSet:
    """載入一個 MedMNIST 資料集的某個 split，統一成 3 通道 [-1,1]。"""
    import medmnist
    from medmnist import INFO

    root = os.path.join(data_dir, "medmnist")
    os.makedirs(root, exist_ok=True)
    cls = getattr(medmnist, INFO[flag]["python_class"])
    ds = cls(split=split, download=True, as_rgb=True, root=root)
    x = _normalize(ds.imgs)
    y = np.asarray(ds.labels, dtype=np.int64).ravel()
    names = [INFO[flag]["label"][str(i)] for i in range(len(INFO[flag]["label"]))]
    return ImageSet(x, y, f"{flag}[{split}]", names)


def load_fashion_ood(data_dir: str, n: int | None = None, seed: int = 0) -> ImageSet:
    """FashionMNIST 當「簡單」OOD —— 與胸片完全不同類別的東西。

    28×28 灰階，複製成 3 通道以匹配模型輸入。
    """
    from torchvision.datasets import FashionMNIST

    root = os.path.join(data_dir, "fashion")
    os.makedirs(root, exist_ok=True)
    ds = FashionMNIST(root=root, train=False, download=True)
    x_uint8 = ds.data.numpy()  # (N,28,28) uint8
    if n is not None and n < len(x_uint8):
        idx = np.random.default_rng(seed).choice(len(x_uint8), n, replace=False)
        x_uint8 = x_uint8[idx]
    x = _normalize(x_uint8)
    return ImageSet(x, np.full(len(x), -1, dtype=np.int64), "fashionmnist[ood]")


def holdout_classes(train: ImageSet, val: ImageSet, test: ImageSet,
                    held: tuple[int, ...]) -> tuple[ImageSet, ImageSet, ImageSet, ImageSet]:
    """PathMNIST 的「困難」OOD：把 `held` 這幾類從訓練中完全移除，當成未見類別。

    剩餘類別的標籤重新映射成 0..K-1（否則 CrossEntropy 的輸出維度對不上）。
    回傳 (train_kept, val_kept, test_kept, test_held_as_ood)。

    這是最難的一組，因為未見類別與訓練類別來自**同一個影像分佈**
    （同樣的染色、同樣的顯微鏡、同樣的色彩統計）——低階統計幾乎一樣，
    模型必須真的在語意層面察覺「這個組織型態我沒見過」。
    """
    held = tuple(held)
    keep_mask = lambda s: ~np.isin(s.y, held)
    kept_labels = sorted(set(range(train.n_classes)) - set(held))
    remap = {old: new for new, old in enumerate(kept_labels)}

    def _keep(s: ImageSet, tag: str) -> ImageSet:
        m = keep_mask(s)
        y = np.array([remap[v] for v in s.y[m]], dtype=np.int64)
        names = [s.class_names[i] for i in kept_labels] if s.class_names else None
        return ImageSet(s.x[m], y, f"{s.name}|kept{tag}", names)

    m = np.isin(test.y, held)
    held_names = [test.class_names[i] for i in held] if test.class_names else None
    ood = ImageSet(test.x[m], np.full(int(m.sum()), -1, dtype=np.int64),
                   f"pathmnist[ood: classes {held}]", held_names)

    return _keep(train, ""), _keep(val, ""), _keep(test, ""), ood


def to_grayscale(s: ImageSet) -> ImageSet:
    """把彩色影像轉灰階後**複製回 3 通道** —— 移除「顏色捷徑」的受控版本。

    為什麼需要這個：in-distribution 的 PneumoniaMNIST 是灰階（三個通道完全相同，
    通道間標準差 = 0.0000），而 DermaMNIST 是彩色（0.2128）。
    這意味著「Pneumonia vs Derma」這組 OOD 實驗，模型只要學會偵測
    「這張圖有顏色」就能拿到接近完美的 AUROC —— 那不是語意理解，是捷徑。

    把 Derma 轉成灰階後，兩邊的通道統計一致，捷徑被拔掉。
    **彩色版與灰階版的 AUROC 差距，就是顏色捷徑的貢獻量**，
    這把一個混淆因子變成了可測量的效應。

    用 ITU-R BT.601 亮度權重（人眼感知加權），不是簡單平均 ——
    後者會讓綠色通道被低估，在皮膚病灶影像上不是中性的操作。
    """
    x = s.x  # (N,3,H,W) in [-1,1]
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 3, 1, 1)
    gray = (x * w).sum(axis=1, keepdims=True)
    return ImageSet(np.repeat(gray, 3, axis=1), s.y.copy(), f"{s.name}|gray", s.class_names)


def add_gaussian_noise(s: ImageSet, sigma: float, seed: int = 0) -> ImageSet:
    """對影像加高斯噪聲 —— 供「分解是否真的分離」的受控實驗用。

    加噪聲讓影像**本身**變模糊、類別界線變不清 → 理論上應該主要抬升
    **aleatoric**（資料固有的不確定性），而不是 epistemic（模型知識的不足）。
    這是檢驗熵分解有沒有真的在分離兩種不確定性的直接手段。

    注意噪聲加在已正規化的 [-1,1] 尺度上，之後 clip 回 [-1,1]
    （模擬真實的感測器飽和，也避免把分佈外的極端值當成另一種 OOD）。
    """
    rng = np.random.default_rng(seed)
    x = s.x + rng.normal(0, sigma, s.x.shape).astype(np.float32)
    return ImageSet(np.clip(x, -1.0, 1.0), s.y.copy(), f"{s.name}+noise{sigma:g}", s.class_names)


def subsample(s: ImageSet, n: int, seed: int = 0, stratified: bool = True) -> ImageSet:
    """子抽樣（PathMNIST 有 9 萬張，控制訓練時間用）。

    分層抽樣保持類別比例 —— 非分層會讓稀有類別在小樣本下消失，
    那會改變問題本身而不只是縮小它。
    """
    if n >= s.n:
        return s
    rng = np.random.default_rng(seed)
    if not stratified or s.y.min() < 0:
        idx = rng.choice(s.n, n, replace=False)
    else:
        idx_parts = []
        classes, counts = np.unique(s.y, return_counts=True)
        for c, cnt in zip(classes, counts):
            take = max(1, int(round(n * cnt / s.n)))
            pool = np.where(s.y == c)[0]
            idx_parts.append(rng.choice(pool, min(take, len(pool)), replace=False))
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
        idx = idx[:n]
    return s.subset(np.sort(idx))


def raw_pixel_stats(s: ImageSet) -> dict:
    """低階影像統計 —— trivial OOD baseline 的原料。

    刻意只用最笨的量：平均亮度、對比（標準差）、L2 norm、以及通道間差異
    （灰階影像的三個通道完全相同 → 通道差為 0，彩色影像不為 0）。
    如果這些就能分開 in-dist 與 OOD，那「模型的不確定性偵測到了 OOD」
    這個說法就需要大幅打折。
    """
    x = s.x
    return {
        "mean": x.mean(axis=(1, 2, 3)),
        "std": x.std(axis=(1, 2, 3)),
        "l2": np.sqrt((x**2).sum(axis=(1, 2, 3))),
        "chan_spread": x.std(axis=1).mean(axis=(1, 2)),
    }


def load_experiment_data(data_dir: str, path_subsample: int | None = 20000,
                         fashion_n: int | None = None, held: tuple[int, ...] = (7, 8),
                         seed: int = 0) -> dict:
    """一次載好三組實驗需要的全部資料。

    `path_subsample`：PathMNIST 訓練集有 ~9 萬張，全用會讓 Deep Ensemble
    的訓練時間主導整個專案。子抽樣到 2 萬張是**計算預算的選擇**，
    在 README 中明確標示；它縮小了問題但不改變結論的方向
    （困難組本來就預期 AUROC 偏低）。
    """
    out = {}

    # ── 實驗 1、2：PneumoniaMNIST 為 in-distribution ──────────────────
    out["pneu_train"] = load_medmnist("pneumoniamnist", "train", data_dir)
    out["pneu_val"] = load_medmnist("pneumoniamnist", "val", data_dir)
    out["pneu_test"] = load_medmnist("pneumoniamnist", "test", data_dir)
    out["ood_fashion"] = load_fashion_ood(data_dir, n=fashion_n, seed=seed)
    derma = load_medmnist("dermamnist", "test", data_dir)
    out["ood_derma"] = ImageSet(derma.x, np.full(derma.n, -1, dtype=np.int64),
                                "dermamnist[ood]", derma.class_names)
    # 顏色捷徑的受控版本：in-dist 的胸片是灰階，彩色的 Derma 可以靠「有顏色」被抓到。
    # 灰階版拔掉這個捷徑，兩版的 AUROC 差距就是捷徑的貢獻量。
    out["ood_derma_gray"] = to_grayscale(out["ood_derma"])

    # ── 實驗 3：PathMNIST 留出 2 類 ───────────────────────────────────
    p_tr = load_medmnist("pathmnist", "train", data_dir)
    p_va = load_medmnist("pathmnist", "val", data_dir)
    p_te = load_medmnist("pathmnist", "test", data_dir)
    tr, va, te, ood = holdout_classes(p_tr, p_va, p_te, held)
    if path_subsample:
        tr = subsample(tr, path_subsample, seed=seed)
        va = subsample(va, max(2000, path_subsample // 10), seed=seed)
    out["path_train"], out["path_val"], out["path_test"] = tr, va, te
    out["ood_path_held"] = ood
    out["path_held_classes"] = held
    out["path_held_names"] = ood.class_names

    return out
