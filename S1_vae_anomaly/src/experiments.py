"""
S1 · 實驗 —— 通關標準與它們的誠實對照。

實驗地圖
--------
通關 1(異常 ELBO 明顯較低)  → `score_all` + `auroc_table`
通關 2(ELBO 分解 / 重參數化 / OOD) → `auroc_table` 的分項 + vae.grad_variance_comparison

誠實對照(本專案的重點,沿用 A3 的精神):
  `trivial_baselines`      不看模型的像素統計能達到多少 AUROC?
  `complexity_analysis`    ELBO 是不是只在測量「這張圖多難壓縮」?
  `complexity_corrected`   扣掉複雜度成分後還剩多少判別力?
  `iwae_sweep`             把下界收緊會讓偵測變好嗎?(不是自明的)
  `beta_sweep`             posterior collapse 與偵測能力的 trade-off
  `latent_sweep`           latent 容量:太大會不會把異常也重建出來?
  `seed_stability`         這些差異有沒有大過訓練隨機性?
"""
from __future__ import annotations

import time

import numpy as np
import torch

import evaluate as E
import train as T
import vae as V

# 分數方向統一為「越大越異常」。ELBO/recon/IWAE 越大越正常,所以取負;
# KL 越大表示 latent 被推離先驗越遠,本身就是「越大越異常」。
SCORE_KEYS = ("neg_elbo", "neg_recon", "kl", "neg_iwae")


@torch.no_grad()
def score_all(model, s, dev, batch_size: int = 256, iwae_K: int = 20,
              n_samples: int = 8, seed: int = 0) -> dict:
    """對一組影像算出全部逐樣本分數。

    `n_samples>1`:ELBO 的重建項是期望值,單樣本估計的噪聲會直接變成
    分數噪聲、稀釋 AUROC。用 8 個樣本平均把它壓下來(成本很低)。
    """
    model.eval()
    out = {k: [] for k in ("elbo", "recon", "kl", "iwae")}
    out["kl_per_dim"] = []
    for i in range(0, s.n, batch_size):
        xb = torch.from_numpy(s.x[i:i + batch_size]).to(dev)
        torch.manual_seed(seed + i)
        t = V.elbo_terms(model, xb, n_samples=n_samples)
        for k in ("elbo", "recon", "kl"):
            out[k].append(t[k].cpu().numpy())
        out["kl_per_dim"].append(t["kl_per_dim"].cpu().numpy())
        out["iwae"].append(V.iwae_bound(model, xb, K=iwae_K,
                                        seed=seed + i).cpu().numpy())
    d = {k: np.concatenate(v) for k, v in out.items() if k != "kl_per_dim"}
    d["kl_per_dim"] = np.concatenate(out["kl_per_dim"], axis=0)
    d["neg_elbo"], d["neg_recon"], d["neg_iwae"] = -d["elbo"], -d["recon"], -d["iwae"]
    return d


def _labels_and(sc_norm, sc_anom, key):
    y = np.concatenate([np.zeros(len(sc_norm[key])), np.ones(len(sc_anom[key]))])
    s = np.concatenate([sc_norm[key], sc_anom[key]])
    return s, y


def auroc_table(sc_norm: dict, sc_anom: dict, n_boot: int = 2000,
                seed: int = 0) -> dict:
    """每種分數的 AUROC + bootstrap 區間 + 分離度。"""
    out = {}
    for k in SCORE_KEYS:
        s, y = _labels_and(sc_norm, sc_anom, k)
        ci = E.auroc_ci(s, y, n_boot=n_boot, seed=seed)
        base = k.replace("neg_", "")
        ci["separation"] = E.separation(sc_anom[base], sc_norm[base])
        out[k] = ci
    return out


def trivial_baselines(set_norm, set_anom, data_mod, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """**不看模型**的像素統計當異常分數 —— A3 的教訓在這裡重演一次。

    每個統計量都試兩個方向(原值與取負),取較好的那個並記下 `flipped`。
    這對 trivial baseline 是**有利**的設定(等於讓它看標籤選方向),
    刻意如此:我們要的是「一個懶惰的方法最好能做到多少」的上界,
    低估它會讓模型看起來比實際更強。
    """
    cn = data_mod.complexity_scores(set_norm)
    ca = data_mod.complexity_scores(set_anom)
    cn["png_bits"] = E.png_bits(set_norm.x)
    ca["png_bits"] = E.png_bits(set_anom.x)

    out = {}
    for k in cn:
        s = np.concatenate([cn[k], ca[k]])
        y = np.concatenate([np.zeros(len(cn[k])), np.ones(len(ca[k]))])
        a = E.auroc(s, y)
        flipped = a < 0.5
        r = E.auroc_ci(-s if flipped else s, y, n_boot=n_boot, seed=seed)
        r["flipped"] = bool(flipped)
        out[k] = r
    best = max(out.items(), key=lambda kv: kv[1]["auroc"])
    return {"scores": out, "best": [best[0], best[1]["auroc"]],
            "raw_normal": cn, "raw_anomaly": ca}


def complexity_analysis(sc_norm, sc_anom, triv, n_bins: int = 5,
                        n_boot: int = 1000, seed: int = 0) -> dict:
    """ELBO 有多少是在測量複雜度?

    三個角度:
      1. **相關**:ELBO 與 PNG 位元數的 Spearman(在正常樣本內算,
         避免組間差異假造出相關)
      2. **分層 AUROC**:控制複雜度後剩下的判別力
      3. **複雜度校正分數**:S = −ELBO − L(x)(Serrà et al. 2020)
    """
    bits_n, bits_a = triv["raw_normal"]["png_bits"], triv["raw_anomaly"]["png_bits"]
    bits = np.concatenate([bits_n, bits_a])

    out = {"spearman_elbo_vs_bits": {
        "normal_only": E.spearman(sc_norm["elbo"], bits_n),
        "anomaly_only": E.spearman(sc_anom["elbo"], bits_a),
        "pooled": E.spearman(np.concatenate([sc_norm["elbo"], sc_anom["elbo"]]), bits),
    }, "bits_mean": {"normal": float(bits_n.mean()), "anomaly": float(bits_a.mean())}}

    out["n_bins"] = n_bins
    out["stratified"], out["raw_auroc"] = {}, {}
    for k in SCORE_KEYS:
        s, y = _labels_and(sc_norm, sc_anom, k)
        out["raw_auroc"][k] = E.auroc(s, y)
        out["stratified"][k] = E.stratified_auroc(s, y, bits, n_bins=n_bins,
                                                  n_boot=n_boot, seed=seed)
    # 複雜度也對自己分層 —— 它「應該」掉到 0.5,而實測掉不到。
    # 那個殘餘就是分層法本身的解析度極限(箱內仍有複雜度變異),
    # 所以它是模型分層 AUROC 的**對照底線**,不是可以忽略的瑕疵。
    y_all = np.concatenate([np.zeros(len(bits_n)), np.ones(len(bits_a))])
    ctrl = bits if not triv["scores"]["png_bits"]["flipped"] else -bits
    out["stratified"]["png_bits_control"] = E.stratified_auroc(
        ctrl, y_all, bits, n_bins=n_bins, n_boot=n_boot, seed=seed)

    # 箱數敏感度:結論不該取決於「切幾箱」這個任意選擇
    s_elbo, _ = _labels_and(sc_norm, sc_anom, "neg_elbo")
    out["bin_sensitivity"] = [
        {"n_bins": nb,
         "neg_elbo": E.stratified_auroc(s_elbo, y_all, bits, n_bins=nb, n_boot=0)["auroc"],
         "floor": E.stratified_auroc(ctrl, y_all, bits, n_bins=nb, n_boot=0)["auroc"]}
        for nb in (3, 5, 8, 10, 15, 20, 30)]
    for r in out["bin_sensitivity"]:
        r["net"] = r["neg_elbo"] - r["floor"]
    return out


def complexity_corrected(sc_norm, sc_anom, triv, n_boot: int = 2000,
                         seed: int = 0) -> dict:
    """S = −ELBO − L(x):把通用壓縮器估到的複雜度成分扣掉。

    直覺:−ELBO 大約是「用我的模型編碼這張圖要幾個 nat」。
    如果一張圖對**任何**編碼器都很貴(L(x) 大),那它的 −ELBO 大
    就不代表異常。相減後剩下的是「我的模型比通用壓縮器差多少」。

    單位要對齊:ELBO 是 nats,PNG 是 bits → 乘 ln2。
    """
    LN2 = float(np.log(2.0))
    y = np.concatenate([np.zeros(sc_norm["elbo"].shape[0]),
                        np.ones(sc_anom["elbo"].shape[0])])
    bits = np.concatenate([triv["raw_normal"]["png_bits"],
                           triv["raw_anomaly"]["png_bits"]]) * LN2
    out = {}
    for k in ("neg_elbo", "neg_iwae"):
        raw = np.concatenate([sc_norm[k], sc_anom[k]])
        out[k] = {
            "raw": E.auroc_ci(raw, y, n_boot=n_boot, seed=seed),
            "corrected": E.auroc_ci(raw - bits, y, n_boot=n_boot, seed=seed),
            "paired_delta": E.paired_auroc_ci(raw - bits, raw, y,
                                              n_boot=n_boot, seed=seed),
        }
    return out


def iwae_sweep(model, set_norm, set_anom, dev, K_list=(1, 5, 20, 50, 100),
               seed: int = 0, n_boot: int = 1000) -> dict:
    """把下界收緊(K↑)會讓異常偵測變好嗎?

    L_K 隨 K 單調上升(更接近 log p(x)),所以**平均 ELBO 一定改善**。
    但 AUROC 是另一回事:ELBO 的鬆弛量 KL(q‖p(z|x)) 本身帶有
    「這筆資料的後驗好不好近似」的資訊,而那對異常偵測可能有用。
    收緊下界會移除這個訊號 —— 所以 K↑ 讓偵測變差是完全可能的。
    """
    rows = []
    for K in K_list:
        t0 = time.time()
        sn = score_all(model, set_norm, dev, iwae_K=K, seed=seed)
        sa = score_all(model, set_anom, dev, iwae_K=K, seed=seed)
        s, y = _labels_and(sn, sa, "neg_iwae")
        ci = E.auroc_ci(s, y, n_boot=n_boot, seed=seed)
        rows.append({"K": K, "auroc": ci["auroc"], "lo95": ci["lo95"], "hi95": ci["hi95"],
                     "mean_bound_normal": float(sn["iwae"].mean()),
                     "mean_bound_anomaly": float(sa["iwae"].mean()),
                     "mean_elbo_normal": float(sn["elbo"].mean()),
                     "gap_to_elbo_normal": float(sn["iwae"].mean() - sn["elbo"].mean()),
                     "seconds": round(time.time() - t0, 1)})
    return {"rows": rows, "K_list": list(K_list)}


def beta_sweep(train_set, val_set, test_norm, test_anom, dev,
               betas=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0), epochs: int = 200,
               latent_dim: int = 16, seed: int = 0, n_boot: int = 1000,
               verbose: bool = True) -> dict:
    """β-VAE 的 β 掃描:posterior collapse 與偵測能力。

    預期:β↑ → KL 被壓 → 活躍維度掉 → 分數失去「latent 有多不尋常」的成分。
    但重建項的權重相對變小也會讓重建變糊,兩個效應方向未必一致 ——
    所以這裡同時記錄活躍維度數與 AUROC,讓資料自己說話。

    ⚠️ β≠1 時目標不是 log p(x) 的下界。表格裡仍用 β=1 的 ELBO 公式算分數
    (那是「拿這個模型當異常偵測器」時實際會用的分數),但不稱它為該模型的
    訓練目標,也不宣稱它是下界。
    """
    rows = []
    for b in betas:
        m, info = T.train(train_set, val_set, latent_dim=latent_dim, beta=b,
                          epochs=epochs, seed=seed)
        au = active_units_of(m, train_set, dev)
        sn = score_all(m, test_norm, dev, seed=seed)
        sa = score_all(m, test_anom, dev, seed=seed)
        tab = auroc_table(sn, sa, n_boot=n_boot, seed=seed)
        rows.append({"beta": b, "auroc_elbo": tab["neg_elbo"]["auroc"],
                     "lo95": tab["neg_elbo"]["lo95"], "hi95": tab["neg_elbo"]["hi95"],
                     "auroc_recon": tab["neg_recon"]["auroc"],
                     "auroc_kl": tab["kl"]["auroc"],
                     "n_active": au["n_active_by_kl"],
                     "n_active_mu": au["n_active_by_mu_var"],
                     "total_kl": au["total_kl"],
                     "val_elbo": info["best_val_elbo"],
                     "mean_recon_normal": float(sn["recon"].mean()),
                     "epochs_run": info["epochs_run"], "seconds": info["seconds"]})
        if verbose:
            r = rows[-1]
            print(f"  β={b:<5g} AUROC(elbo)={r['auroc_elbo']:.3f} "
                  f"recon={r['auroc_recon']:.3f} kl={r['auroc_kl']:.3f} | "
                  f"active {r['n_active']}/{latent_dim} KL={r['total_kl']:.2f}", flush=True)
    return {"rows": rows}


def latent_sweep(train_set, val_set, test_norm, test_anom, dev,
                 dims=(2, 4, 8, 16, 32, 64), epochs: int = 200, seed: int = 0,
                 n_boot: int = 1000, verbose: bool = True) -> dict:
    """latent 容量掃描。

    異常偵測對容量有一個張力:容量太小 → 連正常影像都重建不好,
    分數被「模型不夠力」主導;容量太大 → 模型連沒見過的異常也能重建
    (VAE 的 decoder 泛化性太好),分數失去判別力。
    中間有沒有最佳點?這是可以直接測的,而不是靠直覺挑 latent_dim。
    """
    rows = []
    for d in dims:
        m, info = T.train(train_set, val_set, latent_dim=d, beta=1.0,
                          epochs=epochs, seed=seed)
        au = active_units_of(m, train_set, dev)
        sn = score_all(m, test_norm, dev, seed=seed)
        sa = score_all(m, test_anom, dev, seed=seed)
        tab = auroc_table(sn, sa, n_boot=n_boot, seed=seed)
        rows.append({"latent_dim": d, "auroc_elbo": tab["neg_elbo"]["auroc"],
                     "lo95": tab["neg_elbo"]["lo95"], "hi95": tab["neg_elbo"]["hi95"],
                     "auroc_recon": tab["neg_recon"]["auroc"],
                     "auroc_kl": tab["kl"]["auroc"],
                     "n_active": au["n_active_by_kl"],
                     "frac_active": au["frac_active_by_kl"],
                     "val_elbo": info["best_val_elbo"],
                     "recon_normal": float(sn["recon"].mean()),
                     "recon_anomaly": float(sa["recon"].mean()),
                     "n_params": info["n_params"], "seconds": info["seconds"]})
        if verbose:
            r = rows[-1]
            print(f"  d={d:<3d} AUROC(elbo)={r['auroc_elbo']:.3f} "
                  f"active={r['n_active']}/{d} val_elbo={r['val_elbo']:.2f} "
                  f"recon N/A={r['recon_normal']:.1f}/{r['recon_anomaly']:.1f}", flush=True)
    return {"rows": rows}


def seed_stability(train_set, val_set, test_norm, test_anom, dev,
                   seeds=(0, 1, 2, 3, 4), epochs: int = 200, latent_dim: int = 16,
                   n_boot: int = 500, verbose: bool = True) -> dict:
    """同一設定訓練多顆模型 —— 上面所有掃描的差異都要跟這個比。

    A3 的教訓:兩次「相同」執行的準確率差了 1.6 個百分點。
    如果 β 掃描的 AUROC 差距小於 seed 之間的散佈,那個「趨勢」就是噪聲。
    """
    rows = []
    for sd in seeds:
        m, info = T.train(train_set, val_set, latent_dim=latent_dim, beta=1.0,
                          epochs=epochs, seed=sd)
        sn = score_all(m, test_norm, dev, seed=0)
        sa = score_all(m, test_anom, dev, seed=0)
        tab = auroc_table(sn, sa, n_boot=n_boot, seed=0)
        au = active_units_of(m, train_set, dev)
        rows.append({"seed": sd, "auroc_elbo": tab["neg_elbo"]["auroc"],
                     "auroc_recon": tab["neg_recon"]["auroc"],
                     "auroc_kl": tab["kl"]["auroc"],
                     "n_active": au["n_active_by_kl"],
                     "val_elbo": info["best_val_elbo"], "best_epoch": info["best_epoch"]})
        if verbose:
            print(f"  seed={sd} AUROC(elbo)={rows[-1]['auroc_elbo']:.4f} "
                  f"active={rows[-1]['n_active']}", flush=True)
    a = np.array([r["auroc_elbo"] for r in rows])
    return {"rows": rows, "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max()), "range": float(a.max() - a.min())}


def active_units_of(model, s, dev):
    return T.active_units(model, s, dev)


@torch.no_grad()
def reconstructions(model, sets: dict, dev, n: int = 8, seed: int = 0) -> dict:
    """取幾張圖的重建結果 —— 「模型看不看得懂這張圖」的直接證據。

    重建用 **後驗均值 μ**(不抽樣),因為要展示的是模型的最佳猜測,
    抽樣的噪聲會讓圖看起來比實際更糟。逐樣本 ELBO 另外標在圖上。
    """
    model.eval()
    rng = np.random.default_rng(seed)
    out = {}
    for name, s in sets.items():
        idx = rng.choice(s.n, min(n, s.n), replace=False)
        xb = torch.from_numpy(s.x[idx]).to(dev)
        mu, logvar = model.encode(xb)
        xhat = torch.sigmoid(model.decode(mu))
        t = V.elbo_terms(model, xb, n_samples=8, seed=seed)
        out[name] = {"x": s.x[idx], "xhat": xhat.cpu().numpy(),
                     "elbo": t["elbo"].cpu().numpy(),
                     "recon": t["recon"].cpu().numpy(),
                     "kl": t["kl"].cpu().numpy()}
    return out


@torch.no_grad()
def latent_embedding(model, sets: dict, dev, batch_size: int = 256) -> dict:
    """把 μ(x) 投到 2D(對正常樣本的 μ 做 PCA)看異常落在哪。

    PCA 只用**正常**樣本擬合 —— 用全部樣本擬合會讓投影方向去遷就異常,
    那就看不出「異常有沒有離開正常的流形」。
    """
    model.eval()
    mus = {}
    for name, s in sets.items():
        acc = []
        for i in range(0, s.n, batch_size):
            xb = torch.from_numpy(s.x[i:i + batch_size]).to(dev)
            acc.append(model.encode(xb)[0].cpu().numpy())
        mus[name] = np.concatenate(acc)

    ref = mus["test_normal"]
    center = ref.mean(0)
    _, _, Vt = np.linalg.svd(ref - center, full_matrices=False)
    W = Vt[:2].T
    evr = None
    sv = np.linalg.svd(ref - center, compute_uv=False)
    evr = (sv**2 / (sv**2).sum())[:2]
    return {"proj": {k: (v - center) @ W for k, v in mus.items()},
            "mu": mus, "explained_variance_ratio": evr.tolist()}
