"""
A3 · 實驗編排：OOD 套組、分解驗證、成本效益、選擇性預測。

每個函式只回傳純資料（dict / numpy），不畫圖、不落盤 —— 沿用 A2/B2 的
「推論與出圖分離」原則，讓 run_all 負責落盤、replot 負責重畫。
"""
from __future__ import annotations

import numpy as np
import torch

import data as D
import evaluate as E
import train as T
import uncertainty as U
from model import device

# 三種方法的名稱與說明（圖表與 README 共用）
METHODS = ("single", "mcdrop", "ensemble")
METHOD_LABEL = {
    "single": "Single model (softmax)",
    "mcdrop": "MC Dropout (T=50)",
    "ensemble": "Deep Ensemble (M=5)",
}
# 可當 OOD 分數的量。single 模型的 epistemic 恆為 0，會被自動跳過。
SCORES = ("epistemic", "total_entropy", "neg_max_prob")
SCORE_LABEL = {
    "epistemic": "Epistemic (mutual information)",
    "total_entropy": "Total entropy",
    "neg_max_prob": "1 − max softmax",
    "aleatoric": "Aleatoric",
}


def make_predictors(models, T_mc: int = 50, seed: int = 0):
    """回傳 {method: callable(ImageSet) -> probs (T,N,C)}。

    三種方法共用**同一個**訓練好的網路（`models[0]`）當單模型與 MC Dropout 的載體，
    Ensemble 用全部 M 個。這樣「MC Dropout 輸給 Ensemble」不會是
    「它那個網路剛好比較差」造成的假象。
    """
    dev = device()
    return {
        "single": lambda s: U.predict_deterministic(models[0], s.x, dev),
        "mcdrop": lambda s: U.predict_mc_dropout(models[0], s.x, dev, T=T_mc, seed=seed),
        "ensemble": lambda s: U.predict_ensemble(models, s.x, dev),
    }


def run_ood_suite(models, in_set, ood_sets: dict, T_mc: int = 50, n_boot: int = 300,
                  seed: int = 0) -> dict:
    """核心 OOD 實驗：三方法 × 多分數 × 多個 OOD 來源的 AUROC。

    同時算 **trivial baseline**（只看像素統計，完全不看模型）——
    這是本專案最重要的對照組。如果一行 numpy 就打敗了 MC Dropout，
    那「貝葉斯不確定性偵測到了 OOD」這個說法必須大幅打折。
    """
    preds = make_predictors(models, T_mc=T_mc, seed=seed)
    dec_in = {m: U.decompose(f(in_set)) for m, f in preds.items()}
    scores_in = {m: U.ood_scores(dec_in[m]) for m in preds}
    trivial_in = U.trivial_ood_scores(in_set)

    out = {
        "in_set": in_set.name,
        "in_accuracy": {m: E.accuracy(dec_in[m], in_set.y) for m in preds},
        "in_ece": {m: E.ece(dec_in[m]["mean_prob"], in_set.y)["ece"] for m in preds},
        "in_uncertainty": {m: {k: dec_in[m][k] for k in ("total", "aleatoric", "epistemic",
                                                         "max_prob")} for m in preds},
        "ood": {},
        "trivial": {},
        "confidence": {"in": {m: dec_in[m]["max_prob"] for m in preds}},
    }

    for name, oset in ood_sets.items():
        dec_o = {m: U.decompose(f(oset)) for m, f in preds.items()}
        entry = {"n_ood": oset.n, "auroc": {}, "uncertainty": {}, "confidence": {}}
        for m in preds:
            so = U.ood_scores(dec_o[m])
            entry["uncertainty"][m] = {k: dec_o[m][k] for k in ("total", "aleatoric",
                                                                "epistemic", "max_prob")}
            entry["confidence"][m] = dec_o[m]["max_prob"]
            for sc in SCORES:
                if m == "single" and sc == "epistemic":
                    continue  # 單模型的 epistemic 恆為 0，AUROC 無意義
                a = E.auroc(scores_in[m][sc], so[sc])
                b = E.auroc_bootstrap(scores_in[m][sc], so[sc], n_boot=n_boot, seed=seed)
                entry["auroc"][f"{m}/{sc}"] = {**a, **{f"boot_{k}": v for k, v in b.items()}}
        out["ood"][name] = entry

        # trivial baseline：方向未知，允許事後翻轉（對 baseline 有利，讓對照更嚴格）
        to = U.trivial_ood_scores(oset)
        out["trivial"][name] = {
            k: {**E.auroc(trivial_in[k], to[k], allow_flip=True),
                **{f"boot_{kk}": vv for kk, vv in
                   E.auroc_bootstrap(trivial_in[k], to[k], n_boot=n_boot, seed=seed,
                                     allow_flip=True).items()}}
            for k in trivial_in
        }
        out["trivial"][name]["best"] = max(
            ((k, v["auroc"]) for k, v in out["trivial"][name].items() if k != "best"),
            key=lambda kv: kv[1])

    # 配對比較：Ensemble vs MC Dropout（通關標準 4），以及最佳方法 vs trivial 冠軍
    out["paired"] = {}
    for name, oset in ood_sets.items():
        dec_o = {m: U.decompose(preds[m](oset)) for m in ("mcdrop", "ensemble")}
        out["paired"][name] = E.paired_auroc_diff(
            scores_in["ensemble"]["epistemic"], U.ood_scores(dec_o["ensemble"])["epistemic"],
            scores_in["mcdrop"]["epistemic"], U.ood_scores(dec_o["mcdrop"])["epistemic"],
            n_boot=n_boot, seed=seed)
    return out


def decomposition_validation(models, in_set, ood_set, sigmas=(0.0, 0.15, 0.3, 0.5, 0.8),
                             T_mc: int = 50, seed: int = 0) -> dict:
    """**驗證熵分解是否真的分離兩種不確定性** —— 本專案的方法論核心。

    大部分作品算了 aleatoric/epistemic 就結束，從不檢查分解是否名符其實。
    這裡做受控對照：

      加噪聲（in-distribution + 高斯噪聲）→ 影像本身變模糊、類別界線變不清
          → 理論上應主要抬升 **aleatoric**
      換分佈（真正的 OOD 影像）           → 模型缺乏知識
          → 理論上應主要抬升 **epistemic**

    如果兩種操作把兩個分量抬得一樣多，那這個分解就沒有實際的區辨價值，
    只是一個恆等式的兩邊。回傳每個 sigma 下兩個分量的變化量，
    以及「操作的選擇性」——噪聲抬 aleatoric 的幅度相對於抬 epistemic 的幅度。
    """
    preds = make_predictors(models, T_mc=T_mc, seed=seed)
    rows = []
    for m, f in preds.items():
        base = U.decompose(f(in_set))
        base_ale, base_epi = base["aleatoric"].mean(), base["epistemic"].mean()
        for sg in sigmas:
            noisy = in_set if sg == 0 else D.add_gaussian_noise(in_set, sg, seed=seed)
            dec = U.decompose(f(noisy))
            rows.append({
                "method": m, "perturbation": "noise", "level": float(sg),
                "aleatoric": float(dec["aleatoric"].mean()),
                "epistemic": float(dec["epistemic"].mean()),
                "total": float(dec["total"].mean()),
                "d_aleatoric": float(dec["aleatoric"].mean() - base_ale),
                "d_epistemic": float(dec["epistemic"].mean() - base_epi),
                "accuracy": float(E.accuracy(dec, noisy.y)),
            })
        dec_o = U.decompose(f(ood_set))
        rows.append({
            "method": m, "perturbation": "distribution_shift", "level": 1.0,
            "aleatoric": float(dec_o["aleatoric"].mean()),
            "epistemic": float(dec_o["epistemic"].mean()),
            "total": float(dec_o["total"].mean()),
            "d_aleatoric": float(dec_o["aleatoric"].mean() - base_ale),
            "d_epistemic": float(dec_o["epistemic"].mean() - base_epi),
            "accuracy": float("nan"),
        })

    # 選擇性指標：噪聲應偏向 aleatoric、換分佈應偏向 epistemic
    #
    # 定義成比值而非差值，因為兩個分量的絕對尺度不同（epistemic 通常小一個量級），
    # 直接比較 Δ 大小會系統性偏向 aleatoric。比值 >1 表示該操作確實選擇性地
    # 抬升了「應該」被抬升的那個分量。
    sel = {}
    for m in preds:
        mr = [r for r in rows if r["method"] == m]
        noise_rows = sorted([r for r in mr if r["perturbation"] == "noise"],
                            key=lambda r: r["level"])
        shift = next(r for r in mr if r["perturbation"] == "distribution_shift")
        base_epi = noise_rows[0]["epistemic"]

        # ⚠️ 選擇性必須在 aleatoric **仍單調上升**的區間裡算。
        # 實測發現 σ 太大時 aleatoric 會反轉下降，而準確率繼續惡化 ——
        # 模型變成「自信地錯誤」。機制與 FashionMNIST 組同源：σ=0.8 的噪聲
        # 讓像素大量飽和到 ±1，把影像推離訓練分佈，觸發 ReLU 網路的外插過度自信。
        # 取最大 σ 來算選擇性會落在反轉之後，測到的是飽和假影而不是「加噪聲」的效果。
        peak = max(noise_rows, key=lambda r: r["aleatoric"])
        reversal = [r for r in noise_rows if r["level"] > peak["level"]]
        strongest_noise = peak

        # 單一模型的 epistemic 恆為 0（T=1，沒有分佈樣本可比較），
        # 任何以它為分母的比值都是除零產物、毫無意義 → 明確標成 nan。
        degenerate = m == "single"
        sel[m] = {
            "noise_d_ale": strongest_noise["d_aleatoric"],
            "noise_d_epi": strongest_noise["d_epistemic"],
            "shift_d_ale": shift["d_aleatoric"],
            "shift_d_epi": shift["d_epistemic"],
            "noise_selectivity": (float("nan") if degenerate else
                                  strongest_noise["d_aleatoric"] /
                                  max(abs(strongest_noise["d_epistemic"]), 1e-9)),
            "shift_selectivity": (float("nan") if degenerate else
                                  shift["d_epistemic"] / max(abs(shift["d_aleatoric"]), 1e-9)),
            "base_epistemic": base_epi,
            "degenerate": degenerate,
            # 選擇性是在哪個 σ 上算的（aleatoric 的峰值），不是最大 σ
            "noise_level_used": strongest_noise["level"],
            "noise_accuracy_at_used": strongest_noise["accuracy"],
            # 非單調反轉：σ 更大時 aleatoric 反而下降，而準確率繼續惡化
            "has_reversal": bool(reversal),
            "reversal_levels": [r["level"] for r in reversal],
            "reversal_d_aleatoric": [r["aleatoric"] - strongest_noise["aleatoric"]
                                     for r in reversal],
            "reversal_d_accuracy": [r["accuracy"] - strongest_noise["accuracy"]
                                    for r in reversal],
            "max_level_tested": noise_rows[-1]["level"],
            "accuracy_at_max_level": noise_rows[-1]["accuracy"],
            "aleatoric_at_max_level": noise_rows[-1]["aleatoric"],
        }
    return {"rows": rows, "selectivity": sel, "sigmas": list(sigmas),
            "ood_set": ood_set.name, "in_set": in_set.name}


def cost_benefit(models, in_set, ood_set, T_list=(1, 2, 5, 10, 20, 50, 100),
                 M_list=(1, 2, 3, 4, 5), seed: int = 0) -> dict:
    """T（MC Dropout 前向次數）與 M（Ensemble 大小）的成本效益。

    實務問題：MC Dropout 要跑幾次才夠？Ensemble 要訓練幾個模型才划算？
    兩者的成本結構完全不同 —— T 是**推論**成本（線性增加延遲），
    M 是**訓練**成本（線性增加訓練時間與儲存），推論時 M 也線性增加延遲。
    在臨床部署上這兩種成本的可接受度差很多。
    """
    dev = device()
    rows = []
    for t in T_list:
        di = U.decompose(U.predict_mc_dropout(models[0], in_set.x, dev, T=t, seed=seed))
        do = U.decompose(U.predict_mc_dropout(models[0], ood_set.x, dev, T=t, seed=seed))
        sc = "total_entropy" if t == 1 else "epistemic"
        rows.append({"kind": "mcdrop", "budget": t,
                     "auroc_epistemic": (float("nan") if t == 1 else
                                         E.auroc(di["epistemic"], do["epistemic"])["auroc"]),
                     "auroc_total": E.auroc(di["total"], do["total"])["auroc"],
                     "in_epistemic_mean": float(di["epistemic"].mean()),
                     "accuracy": float(E.accuracy(di, in_set.y))})
    for m in M_list:
        sub = models[:m]
        di = U.decompose(U.predict_ensemble(sub, in_set.x, dev))
        do = U.decompose(U.predict_ensemble(sub, ood_set.x, dev))
        rows.append({"kind": "ensemble", "budget": m,
                     "auroc_epistemic": (float("nan") if m == 1 else
                                         E.auroc(di["epistemic"], do["epistemic"])["auroc"]),
                     "auroc_total": E.auroc(di["total"], do["total"])["auroc"],
                     "in_epistemic_mean": float(di["epistemic"].mean()),
                     "accuracy": float(E.accuracy(di, in_set.y))})
    return {"rows": rows, "ood_set": ood_set.name}


def selective_prediction(models, in_set, T_mc: int = 50, seed: int = 0) -> dict:
    """risk–coverage：把不確定性接上「轉人工判讀」的決策（接 A1 的棄權選項）。

    對每個方法 × 每個不確定性量都算一條曲線，因為「用哪個量來排序病例」
    是實務上真正要決定的事，而它們的效果不同。
    同時報告 oracle AURC（完美排序的下界），否則無法判斷 AURC 的好壞。
    """
    preds = make_predictors(models, T_mc=T_mc, seed=seed)
    out = {"curves": {}, "oracle_aurc": {}, "accuracy": {}}
    for m, f in preds.items():
        dec = U.decompose(f(in_set))
        correct = dec["pred"] == in_set.y
        out["accuracy"][m] = float(correct.mean())
        out["oracle_aurc"][m] = E.oracle_aurc(correct)
        for sc, vals in U.ood_scores(dec).items():
            if m == "single" and sc == "epistemic":
                continue
            rc = E.risk_coverage(vals, correct)
            out["curves"][f"{m}/{sc}"] = rc
    return out
