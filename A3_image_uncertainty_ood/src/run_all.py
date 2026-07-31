"""
A3 · 一鍵重現：`python src/run_all.py`

兩條訓練軌道：
  軌道 1（PneumoniaMNIST，2 類）→ OOD: FashionMNIST、DermaMNIST、DermaMNIST 灰階
  軌道 2（PathMNIST 留出 2 類，7 類）→ OOD: 留出的那 2 類

每條軌道各訓練 M=5 個模型（Deep Ensemble），其中第一個同時充當
單模型 baseline 與 MC Dropout 的載體。

沿用 A2/B2 的設計原則：**推論結果與出圖資料落盤分離**。
所有繪圖資料寫進 data/A_medical/a3_plotdata.npz，純量結果寫進 figures/results.json，
之後調整圖表只要 `python src/replot.py`，不必重新訓練。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import evaluate as E
import experiments as X
import model as M
import plots as P
import train as T
import uncertainty as U

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "data", "A_medical"))
FIG_DIR = os.path.join(ROOT, "figures")
PLOTDATA = os.path.join(DATA_DIR, "a3_plotdata.npz")
RESULTS = os.path.join(FIG_DIR, "results.json")

M_ENSEMBLE = 5
T_MC = 50
EPOCHS = 30
N_BOOT = 300
PATH_SUBSAMPLE = 20000
HELD = (7, 8)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _to_arrays(prefix: str, suite: dict, arrays: dict):
    """把 suite 裡的逐樣本陣列攤平存進 npz（繪圖需要分佈，不只摘要）。"""
    for m, d in suite["in_uncertainty"].items():
        for k, v in d.items():
            arrays[f"{prefix}_in_{m}_{k}"] = v
    for oname, entry in suite["ood"].items():
        for m, d in entry["uncertainty"].items():
            for k, v in d.items():
                arrays[f"{prefix}_ood_{oname}_{m}_{k}"] = v


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    res: dict = {"config": {"M_ensemble": M_ENSEMBLE, "T_mc": T_MC, "epochs": EPOCHS,
                            "n_boot": N_BOOT, "path_subsample": PATH_SUBSAMPLE,
                            "held_classes": list(HELD)}}
    arrays: dict = {}

    # ── 資料 ─────────────────────────────────────────────────────────────
    d = D.load_experiment_data(DATA_DIR, path_subsample=PATH_SUBSAMPLE, held=HELD)
    res["data"] = {k: {"n": d[k].n, "name": d[k].name,
                       "n_classes": d[k].n_classes if d[k].y.min() >= 0 else None}
                   for k in d if hasattr(d[k], "n")}
    res["data_extra"] = {"path_held_classes": list(HELD),
                         "path_held_names": d["path_held_names"],
                         "path_kept_names": d["path_train"].class_names}
    _log("data: " + "; ".join(f"{k}={d[k].n}" for k in
                              ("pneu_train", "pneu_test", "ood_fashion", "ood_derma",
                               "path_train", "path_test", "ood_path_held")))

    # 低階像素統計（誠實檢查的原料，也寫進 results 供 README 引用）
    res["pixel_stats"] = {}
    for k in ("pneu_test", "ood_fashion", "ood_derma", "ood_derma_gray",
              "path_test", "ood_path_held"):
        st = D.raw_pixel_stats(d[k])
        res["pixel_stats"][k] = {kk: float(vv.mean()) for kk, vv in st.items()}
    _log("pixel stats: " + "; ".join(
        f"{k}(chan_spread={v['chan_spread']:.4f})" for k, v in res["pixel_stats"].items()))

    # ── 軌道 1：PneumoniaMNIST ───────────────────────────────────────────
    _log(f"track 1: training {M_ENSEMBLE} models on PneumoniaMNIST...")
    cw1 = T.class_weights_from(d["pneu_train"], 2)
    res["class_weights_pneumonia"] = cw1.tolist()
    models1, infos1 = T.train_ensemble(d["pneu_train"], d["pneu_val"], 2, m=M_ENSEMBLE,
                                       epochs=EPOCHS, class_weight=cw1, verbose=True)
    res["train_pneumonia"] = infos1

    chk = M.sanity_check_mode_invariance(
        models1[0], torch.from_numpy(d["pneu_test"].x[:64]).to(M.device()))
    res["mode_invariance_check"] = chk
    _log(f"mode-invariance check: passed={chk['passed']} "
         f"train/eval diff={chk['max_train_eval_diff']:.2e} "
         f"batch-composition diff (cpu)={chk['max_batch_composition_diff_cpu']:.2e} "
         f"has_batchnorm={chk['has_batchnorm']}")
    assert chk["passed"], "模型有 batch 依賴行為，MC Dropout 會被污染"

    ood1 = {"fashion": d["ood_fashion"], "derma": d["ood_derma"],
            "derma_gray": d["ood_derma_gray"]}
    suite1 = X.run_ood_suite(models1, d["pneu_test"], ood1, T_mc=T_MC, n_boot=N_BOOT)
    _to_arrays("t1", suite1, arrays)
    for m in X.METHODS:
        arrays[f"t1_conf_in_{m}"] = suite1["confidence"]["in"][m]
        for oname in ood1:
            arrays[f"t1_conf_ood_{oname}_{m}"] = suite1["ood"][oname]["confidence"][m]

    _log("track 1 accuracy/ECE: " + "; ".join(
        f"{m}: acc={suite1['in_accuracy'][m]:.4f} ece={suite1['in_ece'][m]:.4f}"
        for m in X.METHODS))
    for oname, entry in suite1["ood"].items():
        best = max(entry["auroc"].items(), key=lambda kv: kv[1]["auroc"])
        triv = suite1["trivial"][oname]["best"]
        _log(f"  OOD {oname:11s} best model={best[0]} {best[1]['auroc']:.3f} | "
             f"trivial {triv[0].replace('pixel_', '')}={triv[1]:.3f} | "
             f"{'TRIVIAL WINS' if triv[1] > best[1]['auroc'] else 'model wins'}")

    # ── 軌道 2：PathMNIST 留出類別（最難的一組）──────────────────────────
    _log(f"track 2: training {M_ENSEMBLE} models on PathMNIST "
         f"(7 kept classes, held out {HELD})...")
    models2, infos2 = T.train_ensemble(d["path_train"], d["path_val"], 7, m=M_ENSEMBLE,
                                       epochs=EPOCHS, base_seed=200, verbose=True)
    res["train_pathmnist"] = infos2
    ood2 = {"path_held": d["ood_path_held"]}
    suite2 = X.run_ood_suite(models2, d["path_test"], ood2, T_mc=T_MC, n_boot=N_BOOT)
    _to_arrays("t2", suite2, arrays)
    for m in X.METHODS:
        arrays[f"t2_conf_in_{m}"] = suite2["confidence"]["in"][m]
        arrays[f"t2_conf_ood_path_held_{m}"] = suite2["ood"]["path_held"]["confidence"][m]

    _log("track 2 accuracy/ECE: " + "; ".join(
        f"{m}: acc={suite2['in_accuracy'][m]:.4f} ece={suite2['in_ece'][m]:.4f}"
        for m in X.METHODS))
    best2 = max(suite2["ood"]["path_held"]["auroc"].items(), key=lambda kv: kv[1]["auroc"])
    triv2 = suite2["trivial"]["path_held"]["best"]
    _log(f"  OOD path_held best model={best2[0]} {best2[1]['auroc']:.3f} | "
         f"trivial {triv2[0].replace('pixel_', '')}={triv2[1]:.3f} | "
         f"{'TRIVIAL WINS' if triv2[1] > best2[1]['auroc'] else 'model wins'}")

    # 合併兩軌道的 AUROC 成一張矩陣（{method/score: {ood: auroc}}）
    aurocs: dict = {}
    for suite in (suite1, suite2):
        for oname, entry in suite["ood"].items():
            for combo, v in entry["auroc"].items():
                aurocs.setdefault(combo, {})[oname] = v["auroc"]
    res["auroc_matrix"] = aurocs
    res["auroc_detail"] = {
        "track1": {o: e["auroc"] for o, e in suite1["ood"].items()},
        "track2": {o: e["auroc"] for o, e in suite2["ood"].items()},
    }
    res["trivial_baseline"] = {
        "track1": {o: {k: (v if k == "best" else
                           {"auroc": v["auroc"], "flipped": v["flipped"],
                            "boot_lo95": v["boot_lo95"], "boot_hi95": v["boot_hi95"]})
                       for k, v in t.items()} for o, t in suite1["trivial"].items()},
        "track2": {o: {k: (v if k == "best" else
                           {"auroc": v["auroc"], "flipped": v["flipped"],
                            "boot_lo95": v["boot_lo95"], "boot_hi95": v["boot_hi95"]})
                       for k, v in t.items()} for o, t in suite2["trivial"].items()},
    }
    res["paired_ensemble_vs_mcdrop"] = {"track1": suite1["paired"], "track2": suite2["paired"]}
    res["accuracy"] = {"track1": suite1["in_accuracy"], "track2": suite2["in_accuracy"]}
    res["ece"] = {"track1": suite1["in_ece"], "track2": suite2["in_ece"]}

    # ── 分解驗證（方法論核心）────────────────────────────────────────────
    _log("decomposition validation (noise vs distribution shift)...")
    dv = X.decomposition_validation(models1, d["pneu_test"], d["ood_derma"], T_mc=T_MC)
    res["decomposition_validation"] = dv
    for m, s in dv["selectivity"].items():
        if s["degenerate"]:
            continue
        _log(f"  {m:9s} noise selectivity={s['noise_selectivity']:.2f} "
             f"(Δale={s['noise_d_ale']:+.4f} Δepi={s['noise_d_epi']:+.4f}) | "
             f"shift selectivity={s['shift_selectivity']:.2f} "
             f"(Δale={s['shift_d_ale']:+.4f} Δepi={s['shift_d_epi']:+.4f})")

    # ── 成本效益（T 與 M）────────────────────────────────────────────────
    _log("cost-benefit sweep over T and M...")
    cb = X.cost_benefit(models1, d["pneu_test"], d["ood_derma"],
                        M_list=tuple(range(1, M_ENSEMBLE + 1)))
    res["cost_benefit"] = cb
    for kind in ("mcdrop", "ensemble"):
        r = [x for x in cb["rows"] if x["kind"] == kind]
        _log(f"  {kind}: " + ", ".join(
            f"{x['budget']}→{x['auroc_epistemic']:.3f}" for x in r
            if not np.isnan(x["auroc_epistemic"])))

    # ── 選擇性預測（決策層）──────────────────────────────────────────────
    _log("selective prediction (risk-coverage)...")
    sel = X.selective_prediction(models1, d["pneu_test"], T_mc=T_MC)
    res["selective_prediction"] = {
        "accuracy": sel["accuracy"], "oracle_aurc": sel["oracle_aurc"],
        "summary": {k: {"aurc": v["aurc"], "acc_at_100": v["acc_at_100"],
                        "acc_at_80": v["acc_at_80"], "acc_at_50": v["acc_at_50"],
                        "gain_at_80": v["acc_at_80"] - v["acc_at_100"]}
                    for k, v in sel["curves"].items()},
    }
    for k, v in sel["curves"].items():
        arrays[f"rc_{k.replace('/', '_')}_coverage"] = v["coverage"]
        arrays[f"rc_{k.replace('/', '_')}_accuracy"] = v["accuracy"]
    for k, v in res["selective_prediction"]["summary"].items():
        _log(f"  {k:26s} acc {v['acc_at_100']:.4f} → {v['acc_at_80']:.4f} "
             f"@80% coverage ({v['gain_at_80']:+.4f}), AURC {v['aurc']:.4f}")

    # 校準（軌道 1）
    preds1 = X.make_predictors(models1, T_mc=T_MC)
    ece_res = {}
    for m, f in preds1.items():
        dec = U.decompose(f(d["pneu_test"]))
        ece_res[m] = E.ece(dec["mean_prob"], d["pneu_test"].y)
    res["calibration"] = ece_res

    # 範例影像（給圖 7；只存少量）
    rng = np.random.default_rng(0)
    for tag, key in (("in_pneu", "pneu_test"), ("fashion", "ood_fashion"),
                     ("derma", "ood_derma"), ("derma_gray", "ood_derma_gray"),
                     ("in_path", "path_test"), ("path_held", "ood_path_held")):
        s = d[key]
        arrays[f"sample_{tag}"] = s.x[rng.choice(s.n, 8, replace=False)]

    # ── 落盤 ─────────────────────────────────────────────────────────────
    np.savez_compressed(PLOTDATA, **arrays)
    res["_meta"] = {"runtime_sec": round(time.time() - t0, 1),
                    "plotdata": os.path.relpath(PLOTDATA, ROOT),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=_jsonify)
    _log(f"saved {PLOTDATA} ({os.path.getsize(PLOTDATA) / 1e6:.1f} MB) and {RESULTS}")

    make_figures(arrays, res, suite1, suite2, dv, cb, sel, ece_res, d)
    _log(f"done in {res['_meta']['runtime_sec']:.0f}s")


def _jsonify(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return float(o)


def make_figures(arrays, res, suite1, suite2, dv, cb, sel, ece_res, d):
    """出圖。replot.py 用落盤資料重建同樣的呼叫。"""
    # 合併兩軌道的不確定性分佈（軌道 2 的 in-dist 不同，但圖上分欄呈現）
    in_unc = suite1["in_uncertainty"]
    ood_unc = {k: v["uncertainty"] for k, v in suite1["ood"].items()}
    ood_unc["path_held"] = suite2["ood"]["path_held"]["uncertainty"]
    conf_in = suite1["confidence"]["in"]
    conf_ood = {k: v["confidence"] for k, v in suite1["ood"].items()}
    conf_ood["path_held"] = suite2["ood"]["path_held"]["confidence"]

    # 圖 1 的 path_held 欄要用軌道 2 的 in-dist 當對照
    in_unc_by_ood = {k: in_unc for k in suite1["ood"]}
    in_unc_by_ood["path_held"] = suite2["in_uncertainty"]
    conf_in_by_ood = {k: conf_in for k in suite1["ood"]}
    conf_in_by_ood["path_held"] = suite2["confidence"]["in"]

    auroc_by_score = {}
    for combo, per_ood in res["auroc_matrix"].items():
        auroc_by_score[combo] = per_ood

    P.uncertainty_histograms_multi(
        in_unc_by_ood, ood_unc, conf_in_by_ood, conf_ood, auroc_by_score,
        os.path.join(FIG_DIR, "01_uncertainty_histograms.png"), method="ensemble")

    trivial_all = {**res["trivial_baseline"]["track1"], **res["trivial_baseline"]["track2"]}
    P.auroc_matrix(res["auroc_matrix"], trivial_all,
                   os.path.join(FIG_DIR, "02_auroc_matrix.png"))
    P.decomposition_validation(dv, os.path.join(FIG_DIR, "03_decomposition_validation.png"))
    P.risk_coverage(sel, os.path.join(FIG_DIR, "04_risk_coverage.png"))
    P.cost_benefit(cb, os.path.join(FIG_DIR, "05_cost_benefit.png"))
    P.calibration(ece_res, res["accuracy"]["track1"],
                  os.path.join(FIG_DIR, "06_calibration.png"))
    P.sample_grid({"in_pneu": d["pneu_test"], "fashion": d["ood_fashion"],
                   "derma": d["ood_derma"], "derma_gray": d["ood_derma_gray"],
                   "in_path": d["path_test"], "path_held": d["ood_path_held"]},
                  os.path.join(FIG_DIR, "07_samples.png"))
    _log("figures written")


if __name__ == "__main__":
    main()
