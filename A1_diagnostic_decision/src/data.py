"""
A1 · 資料載入與預處理 —— Heart Disease (Cleveland)，A-D1。

303 筆、13 臨床特徵、二元診斷。首次執行自動下載並快取到統一資料夾 data/A_medical/。
預處理：去缺值 → 連續變數標準化 → 名目類別 one-hot → 分層 train/test 切分。

標準化很重要：弱資訊先驗 N(0, 2.5)（計劃書主題二）假設預測變數已標準化，
否則「先驗尺度」對不同單位的特徵就沒有一致意義。
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
NOMINAL = ["cp", "restecg", "slope", "thal"]


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    features: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def _download(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, "processed.cleveland.data")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(URL, dst)
    return dst


def load_raw(data_dir: str):
    """載入未標準化的特徵矩陣（含 one-hot），供交叉驗證做 per-fold 標準化。
    回傳 (X, y, cont_idx, features)。"""
    df = pd.read_csv(_download(data_dir), header=None, names=COLS, na_values="?").dropna().reset_index(drop=True)
    y = (df["num"] > 0).astype(int).to_numpy()
    dummies = pd.get_dummies(df[NOMINAL].astype(int), columns=NOMINAL, drop_first=True)
    X_df = pd.concat([df[CONTINUOUS + BINARY].reset_index(drop=True), dummies], axis=1)
    features = list(X_df.columns)
    cont_idx = [features.index(c) for c in CONTINUOUS]
    return X_df.to_numpy(dtype=float), y, cont_idx, features


def load_heart(data_dir: str, test_size: float = 0.25, seed: int = 0) -> Dataset:
    """載入 Heart Disease，回傳標準化、one-hot 後的 train/test 切分。"""
    X, y, cont_idx, features = load_raw(data_dir)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # 只用連續欄位做標準化（用 train 統計量，避免資料洩漏）
    scaler = StandardScaler().fit(Xtr[:, cont_idx])
    Xtr[:, cont_idx] = scaler.transform(Xtr[:, cont_idx])
    Xte[:, cont_idx] = scaler.transform(Xte[:, cont_idx])

    return Dataset(Xtr, Xte, ytr, yte, features)
