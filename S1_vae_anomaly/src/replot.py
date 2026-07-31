"""
S1 · 從落盤資料重畫全部圖表：`python src/replot.py`

只讀 `data/A_medical/s1_plotdata.npz` + `figures/results.json`，
不訓練任何模型、不跑任何前向傳播 —— 秒級完成。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plots as P
import run_all as R


def main():
    z = np.load(R.PLOTDATA)
    res = json.load(open(R.RESULTS))
    F = R.FIG_DIR

    # 分數：把 −ELBO 等衍生欄位重建出來（落盤只存原始方向，省空間）
    def scores(tag):
        d = {k: z[f"score_{tag}_{k}"] for k in ("elbo", "recon", "kl", "iwae")}
        d["neg_elbo"], d["neg_recon"], d["neg_iwae"] = -d["elbo"], -d["recon"], -d["iwae"]
        return d

    sc_n, sc_a = scores("norm"), scores("anom")

    # trivial baseline：純量在 results.json，複雜度原始值在 npz
    triv = {"scores": res["trivial"]["scores"], "best": res["trivial"]["best"],
            "raw_normal": {"png_bits": z["bits_norm"]},
            "raw_anomaly": {"png_bits": z["bits_anom"]}}

    kv = dict(res["kl_verify"])
    kv.update({k: z[f"klverify_{k}"] for k in ("analytic", "mc", "mc_se")})

    P.data_and_verification(
        {k: z[f"sample_{k}"] for k in ("test_normal", "test_anomaly")},
        kv, res["kl_verify_trained"], res["grad_variance"],
        os.path.join(F, "01_data_and_verification.png"))

    P.elbo_distributions(sc_n, sc_a, res["auroc_table"],
                         os.path.join(F, "02_elbo_distributions.png"))

    rec = {name: {k: z[f"rec_{name}_{k}"] for k in ("x", "xhat", "elbo", "recon", "kl")}
           for name in ("test_normal", "test_anomaly")}
    P.reconstructions(rec, os.path.join(F, "03_reconstructions.png"))

    P.trivial_comparison(res["auroc_table"], triv, res["complexity_corrected"],
                         os.path.join(F, "04_trivial_baseline.png"))
    P.complexity_confound(sc_n, sc_a, triv, res["complexity"],
                          res["complexity_corrected"],
                          os.path.join(F, "05_complexity_confound.png"))
    P.bound_and_collapse(res["iwae_sweep"], res["beta_sweep"],
                         {"beta1": z["kl_per_dim_beta1"],
                          "beta_high": z["kl_per_dim_beta_high"]},
                         res["seed_stability"],
                         os.path.join(F, "06_bound_and_collapse.png"),
                         beta_high=res["beta_high_used"])

    emb = {"proj": {k: z[f"proj_{k}"] for k in ("test_normal", "test_anomaly")},
           "explained_variance_ratio": res["latent_pca_evr"]}
    P.latent_capacity_seed(emb, res["latent_sweep"], res["seed_stability"],
                           os.path.join(F, "07_latent_capacity_seed.png"))

    print(f"replotted 7 figures into {F}")


if __name__ == "__main__":
    main()
