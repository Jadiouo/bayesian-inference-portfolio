"""
S1 · 一鍵重現：`python src/run_all.py`

流程
----
1. 正確性驗證（KL 公式、重參數化 vs REINFORCE）—— 先證明工具是對的
2. 主模型：只在 normal 胸片上訓練 VAE
3. 通關 1/2：ELBO 當異常分數，分解成重建項與 KL 項
4. 誠實對照：trivial 像素統計、複雜度混淆、複雜度校正分數
5. 掃描：IWAE 的 K、β（posterior collapse）、latent 容量、訓練 seed

沿用 A2/A3/B2/C2 的設計原則：**推論結果與出圖資料落盤分離**。
繪圖資料寫進 data/A_medical/s1_plotdata.npz，純量結果寫進 figures/results.json，
之後調圖只要 `python src/replot.py`。

⚠️ 可重現性：A3 的教訓 —— 只設 seed 不夠。本檔在 import torch 前設
`CUBLAS_WORKSPACE_CONFIG`，並在 set_seed 裡關掉 cuDNN 的非決定性路徑。
"""
from __future__ import annotations

import os
import sys

# 必須在 import torch 之前（A3 踩坑 c）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import evaluate as E
import experiments as X
import plots as P
import train as T
import vae as V

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "data", "A_medical"))
FIG_DIR = os.path.join(ROOT, "figures")
PLOTDATA = os.path.join(DATA_DIR, "s1_plotdata.npz")
RESULTS = os.path.join(FIG_DIR, "results.json")

EPOCHS = 300
LATENT_DIM = 16
IWAE_K = 20
N_BOOT = 2000
N_BINS = 8              # 複雜度分層的箱數（見 results.json 的 bin_sensitivity）
BETAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DIMS = (2, 4, 8, 16, 32, 64)
SEEDS = (0, 1, 2, 3, 4)
MAIN_SEED = 0

# `S1_QUICK=1 python src/run_all.py` 跑一遍縮小版：驗證整條管線與全部圖表
# 能產出，約 20 秒。數字沒有意義，**不要**拿它填 README。
if os.environ.get("S1_QUICK") == "1":
    EPOCHS, N_BOOT = 40, 120
    BETAS, DIMS, SEEDS = (0.5, 1.0, 8.0), (2, 16, 64), (0, 1, 2)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsonify(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return float(o)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    dev = V.device()
    res: dict = {"config": {"epochs": EPOCHS, "latent_dim": LATENT_DIM, "iwae_K": IWAE_K,
                            "n_boot": N_BOOT, "n_bins": N_BINS, "betas": list(BETAS),
                            "dims": list(DIMS), "seeds": list(SEEDS),
                            "main_seed": MAIN_SEED}}
    arrays: dict = {}

    # ── 資料 ─────────────────────────────────────────────────────────────
    d = D.load(DATA_DIR)
    res["data"] = {k: {"n": v.n, "name": v.name} for k, v in d.items()}
    _log("data: " + "; ".join(f"{k}={v.n}" for k, v in d.items()))

    # ── 正確性驗證（在任何結果之前）──────────────────────────────────────
    _log("verifying the KL formula against Monte Carlo...")
    kv = V.verify_kl_formula(latent_dim=LATENT_DIM, seed=MAIN_SEED)
    for k in ("analytic", "mc", "mc_se"):
        arrays[f"klverify_{k}"] = kv[k]
    res["kl_verify"] = {k: v for k, v in kv.items()
                        if k not in ("analytic", "mc", "mc_se")}
    _log(f"  KL range {kv['kl_range'][0]:.1f}–{kv['kl_range'][1]:.1f}, "
         f"max deviation {kv['max_diff_in_se_units']:.2f} MC SE, passed={kv['passed']}")
    assert kv["passed"], "解析 KL 與 Monte Carlo 不一致 —— ELBO 的 KL 項有錯"

    # ── 主模型 ───────────────────────────────────────────────────────────
    _log(f"training the main VAE on {d['train_normal'].n} normal chest X-rays...")
    model, info = T.train(d["train_normal"], d["val_normal"], latent_dim=LATENT_DIM,
                          beta=1.0, epochs=EPOCHS, seed=MAIN_SEED, verbose=True)
    res["train"] = {k: v for k, v in info.items() if k != "history"}
    arrays["hist_val_elbo"] = np.array([h["val_elbo"] for h in info["history"]])
    arrays["hist_train_recon"] = np.array([h["train_recon"] for h in info["history"]])
    arrays["hist_train_kl"] = np.array([h["train_kl"] for h in info["history"]])
    _log(f"  {info['epochs_run']} epochs in {info['seconds']}s, "
         f"best val ELBO {info['best_val_elbo']:.2f} @ epoch {info['best_epoch']}, "
         f"{info['n_params']:,} params")

    xb = torch.from_numpy(d["train_normal"].x[:128]).to(dev)
    res["kl_verify_trained"] = V.verify_kl_analytic(model, xb, seed=MAIN_SEED)
    _log(f"  KL check on the trained model: "
         f"max |Δ| / mean KL = {res['kl_verify_trained']['max_diff_over_mean_kl']:.4f}")

    _log("measuring gradient variance: reparameterisation vs REINFORCE...")
    gv = V.grad_variance_comparison(model, xb, n_trials=60, seed=MAIN_SEED)
    res["grad_variance"] = gv
    _log(f"  reparam {gv['reparam']['mean_var']:.3e} vs "
         f"REINFORCE {gv['reinforce']['mean_var']:.3e} "
         f"→ {gv['variance_ratio_reinforce_over_reparam']:,.0f}× larger")

    # ── 通關 1/2：ELBO 當異常分數 ────────────────────────────────────────
    _log("scoring the test set (ELBO, reconstruction, KL, IWAE)...")
    sc_n = X.score_all(model, d["test_normal"], dev, iwae_K=IWAE_K, seed=MAIN_SEED)
    sc_a = X.score_all(model, d["test_anomaly"], dev, iwae_K=IWAE_K, seed=MAIN_SEED)
    for tag, sc in (("norm", sc_n), ("anom", sc_a)):
        for k in ("elbo", "recon", "kl", "iwae"):
            arrays[f"score_{tag}_{k}"] = sc[k]
    table = X.auroc_table(sc_n, sc_a, n_boot=N_BOOT, seed=MAIN_SEED)
    res["auroc_table"] = table
    for k, v in table.items():
        _log(f"  {P.SCORE_LAB[k]:>18s}  AUROC {v['auroc']:.4f} "
             f"[{v['lo95']:.3f},{v['hi95']:.3f}]  d={v['separation']['cohens_d']:+.2f}  "
             f"overlap={v['separation']['overlap_coefficient']:.2f}")

    au = T.active_units(model, d["train_normal"], dev)
    res["active_units"] = {k: v for k, v in au.items()
                           if k not in ("kl_per_dim", "mu_variance_per_dim")}
    arrays["kl_per_dim_beta1"] = au["kl_per_dim"]
    arrays["mu_var_per_dim_beta1"] = au["mu_variance_per_dim"]
    _log(f"  active latent units: {au['n_active_by_kl']}/{au['latent_dim']} by KL, "
         f"{au['n_active_by_mu_var']} by μ-variance, total KL {au['total_kl']:.2f} nats")

    # ── 誠實對照 1：不看模型的像素統計 ───────────────────────────────────
    _log("trivial baselines (pixel statistics, no model)...")
    triv = X.trivial_baselines(d["test_normal"], d["test_anomaly"], D,
                               n_boot=N_BOOT, seed=MAIN_SEED)
    res["trivial"] = {"scores": triv["scores"], "best": triv["best"]}
    for k, v in triv["scores"].items():
        flag = "  ← BEATS THE VAE" if v["auroc"] > table["neg_elbo"]["auroc"] else ""
        _log(f"  {k:18s} AUROC {v['auroc']:.4f} (flipped={v['flipped']}){flag}")
    arrays["bits_norm"] = triv["raw_normal"]["png_bits"]
    arrays["bits_anom"] = triv["raw_anomaly"]["png_bits"]

    # ── 誠實對照 2：ELBO 是不是只在測複雜度 ──────────────────────────────
    _log("complexity confound: correlation, stratified AUROC, bin sensitivity...")
    comp = X.complexity_analysis(sc_n, sc_a, triv, n_bins=N_BINS,
                                 n_boot=N_BOOT // 2, seed=MAIN_SEED)
    res["complexity"] = comp
    sp = comp["spearman_elbo_vs_bits"]
    _log(f"  Spearman(ELBO, PNG bits) = {sp['normal_only']:+.3f} (normal only), "
         f"{sp['pooled']:+.3f} (pooled)")
    floor = comp["stratified"]["png_bits_control"]["auroc"]
    for k in X.SCORE_KEYS:
        st = comp["stratified"][k]
        _log(f"  {P.SCORE_LAB[k]:>18s}  raw {comp['raw_auroc'][k]:.4f} → "
             f"stratified {st['auroc']:.4f} [{st['lo95']:.3f},{st['hi95']:.3f}]")
    _log(f"  residual floor (complexity stratified against itself) = {floor:.4f}")
    _log("  bin sensitivity: " + ", ".join(
        f"k={r['n_bins']}:{r['net']:+.3f}" for r in comp["bin_sensitivity"]))

    # ── 誠實對照 3：扣掉複雜度成分 ───────────────────────────────────────
    _log("complexity-corrected score  S = −ELBO − L(x)...")
    corr = X.complexity_corrected(sc_n, sc_a, triv, n_boot=N_BOOT, seed=MAIN_SEED)
    res["complexity_corrected"] = corr
    for k, v in corr.items():
        pd = v["paired_delta"]
        _log(f"  {P.SCORE_LAB[k]:>18s}  {v['raw']['auroc']:.4f} → "
             f"{v['corrected']['auroc']:.4f}  (Δ={pd['delta']:+.4f} "
             f"[{pd['lo95']:+.4f},{pd['hi95']:+.4f}], win {pd['win_rate']:.0%})")

    # ── 掃描 ─────────────────────────────────────────────────────────────
    _log("sweep: IWAE K (does a tighter bound help?)...")
    iwae = X.iwae_sweep(model, d["test_normal"], d["test_anomaly"], dev,
                        seed=MAIN_SEED, n_boot=N_BOOT // 2)
    res["iwae_sweep"] = iwae
    for r in iwae["rows"]:
        _log(f"  K={r['K']:<4d} AUROC {r['auroc']:.4f}  bound +{r['gap_to_elbo_normal']:.3f} "
             f"nats over ELBO")

    _log("sweep: seed stability (the resolution floor for every sweep below)...")
    stab = X.seed_stability(d["train_normal"], d["val_normal"], d["test_normal"],
                            d["test_anomaly"], dev, seeds=SEEDS, epochs=EPOCHS,
                            latent_dim=LATENT_DIM, n_boot=N_BOOT // 4)
    res["seed_stability"] = stab
    _log(f"  AUROC {stab['mean']:.4f} ± {stab['sd']:.4f} "
         f"(range {stab['range']:.4f} over {len(SEEDS)} seeds)")

    _log("sweep: β (posterior collapse)...")
    beta = X.beta_sweep(d["train_normal"], d["val_normal"], d["test_normal"],
                        d["test_anomaly"], dev, betas=BETAS, epochs=EPOCHS,
                        latent_dim=LATENT_DIM, seed=MAIN_SEED, n_boot=N_BOOT // 4)
    res["beta_sweep"] = beta

    # 取最高 β 的 per-dim KL 給圖 6（collapse 的直接證據）
    m_hi, _ = T.train(d["train_normal"], d["val_normal"], latent_dim=LATENT_DIM,
                      beta=max(BETAS), epochs=EPOCHS, seed=MAIN_SEED)
    arrays["kl_per_dim_beta_high"] = T.active_units(m_hi, d["train_normal"],
                                                    dev)["kl_per_dim"]
    res["beta_high_used"] = max(BETAS)

    _log("sweep: latent capacity...")
    latent = X.latent_sweep(d["train_normal"], d["val_normal"], d["test_normal"],
                            d["test_anomaly"], dev, dims=DIMS, epochs=EPOCHS,
                            seed=MAIN_SEED, n_boot=N_BOOT // 4)
    res["latent_sweep"] = latent

    # 掃描的效應對得上訓練噪聲嗎？
    res["sweep_vs_noise"] = {
        "seed_sd": stab["sd"],
        "iwae_range": max(r["auroc"] for r in iwae["rows"]) -
                      min(r["auroc"] for r in iwae["rows"]),
        "beta_range": max(r["auroc_elbo"] for r in beta["rows"]) -
                      min(r["auroc_elbo"] for r in beta["rows"]),
        "latent_range": max(r["auroc_elbo"] for r in latent["rows"]) -
                        min(r["auroc_elbo"] for r in latent["rows"]),
    }
    sv = res["sweep_vs_noise"]
    _log(f"  effect sizes vs seed SD ({sv['seed_sd']:.4f}): "
         f"IWAE {sv['iwae_range']:.4f} ({sv['iwae_range'] / sv['seed_sd']:.1f}σ), "
         f"β {sv['beta_range']:.4f} ({sv['beta_range'] / sv['seed_sd']:.1f}σ), "
         f"latent {sv['latent_range']:.4f} ({sv['latent_range'] / sv['seed_sd']:.1f}σ)")

    # ── 定性材料 ─────────────────────────────────────────────────────────
    rec = X.reconstructions(model, {"test_normal": d["test_normal"],
                                    "test_anomaly": d["test_anomaly"]}, dev,
                            n=8, seed=MAIN_SEED)
    for name, r in rec.items():
        for k in ("x", "xhat", "elbo", "recon", "kl"):
            arrays[f"rec_{name}_{k}"] = r[k]

    emb = X.latent_embedding(model, {"test_normal": d["test_normal"],
                                     "test_anomaly": d["test_anomaly"]}, dev)
    for k, v in emb["proj"].items():
        arrays[f"proj_{k}"] = v
    res["latent_pca_evr"] = emb["explained_variance_ratio"]

    rng = np.random.default_rng(MAIN_SEED)
    for key in ("test_normal", "test_anomaly"):
        arrays[f"sample_{key}"] = d[key].x[rng.choice(d[key].n, 8, replace=False)]

    # ── 落盤 ─────────────────────────────────────────────────────────────
    np.savez_compressed(PLOTDATA, **arrays)
    res["_meta"] = {"runtime_sec": round(time.time() - t0, 1),
                    "plotdata": os.path.relpath(PLOTDATA, ROOT),
                    "device": str(dev),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                    else "cpu"}
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, default=_jsonify)
    _log(f"saved {PLOTDATA} ({os.path.getsize(PLOTDATA) / 1e6:.1f} MB) and {RESULTS}")

    make_figures(arrays, res, sc_n, sc_a, triv, emb)
    _log(f"done in {res['_meta']['runtime_sec']:.0f}s")


def make_figures(arrays, res, sc_n, sc_a, triv, emb):
    """出圖。replot.py 用落盤資料重建同樣的呼叫。"""
    samples = {k: arrays[f"sample_{k}"] for k in ("test_normal", "test_anomaly")}
    kv = dict(res["kl_verify"])
    kv.update({k: arrays[f"klverify_{k}"] for k in ("analytic", "mc", "mc_se")})

    P.data_and_verification(samples, kv, res["kl_verify_trained"], res["grad_variance"],
                            os.path.join(FIG_DIR, "01_data_and_verification.png"))
    P.elbo_distributions(sc_n, sc_a, res["auroc_table"],
                         os.path.join(FIG_DIR, "02_elbo_distributions.png"))
    rec = {name: {k: arrays[f"rec_{name}_{k}"] for k in ("x", "xhat", "elbo", "recon", "kl")}
           for name in ("test_normal", "test_anomaly")}
    P.reconstructions(rec, os.path.join(FIG_DIR, "03_reconstructions.png"))
    P.trivial_comparison(res["auroc_table"], triv, res["complexity_corrected"],
                         os.path.join(FIG_DIR, "04_trivial_baseline.png"))
    P.complexity_confound(sc_n, sc_a, triv, res["complexity"],
                          res["complexity_corrected"],
                          os.path.join(FIG_DIR, "05_complexity_confound.png"))
    P.bound_and_collapse(res["iwae_sweep"], res["beta_sweep"],
                         {"beta1": arrays["kl_per_dim_beta1"],
                          "beta_high": arrays["kl_per_dim_beta_high"]},
                         res["seed_stability"],
                         os.path.join(FIG_DIR, "06_bound_and_collapse.png"),
                         beta_high=res["beta_high_used"])
    P.latent_capacity_seed(emb, res["latent_sweep"], res["seed_stability"],
                           os.path.join(FIG_DIR, "07_latent_capacity_seed.png"))
    _log("figures written")


if __name__ == "__main__":
    main()
