"""
A2 · 刪失的資訊代價 —— 「精度由事件數決定，不是樣本數」。

生存分析的一句老智慧：**你的統計檢定力取決於觀察到幾個事件，
而不是收了幾個病人。** 這在臨床試驗設計裡是硬約束（所以是
「event-driven trial」——追蹤到累積 N 個事件才解盲，而不是追蹤到某個日期）。

刪失資料確實攜帶資訊（「至少活到 t_i」），但**比事件少**。
本模組用雙臂對照把這句話變成可測量的東西：

    A 臂「截短追蹤期」：把行政刪失時間往前推。**樣本數固定 686 不變**，
                        事件數下降，被截短的人變成刪失（特徵向量仍在）。
    B 臂「隨機子抽樣」：分層隨機丟掉病人，保持事件率 43.6%。
                        **樣本數與事件數同時下降**（特徵向量一起消失）。
    C 臂「截短後丟掉刪失」：先截短，再把刪失整批丟掉。
                        n = 事件數，完全沒有刪失資料可用。

實測結果推翻了本模組原本的假設
------------------------------------
原本預期 A、B 兩臂會重合（「只有事件數重要」）。實測是 **A 臂明顯比 B 臂窄**：
事件數從 299 降到 117 時，B 臂的後驗 sd 膨脹 55%，而 A 臂幾乎不動。
log-log 斜率 A 臂約 −0.25、B 臂約 −0.56（後者才貼合理論 −0.5）。

機制是：兩臂丟掉的東西不一樣。
    A 臂只把**結果**資訊粗化成「T > h」，686 個**特徵向量全部保留**，
      設計矩陣的變異度完全沒變 —— 而迴歸係數的精度吃的正是這個變異度。
    B 臂把整個人拿掉，X 的變異度跟著縮小。
    C 臂是對照組：證明 A 臂的優勢真的來自那些刪失列，把它們丟掉就崩潰。

所以「精度由事件數決定」這句老智慧要講得更精確：
**事件數主導的是基準風險與絕對時間尺度的精度；迴歸係數的精度同時吃
事件數與協變量變異度，而刪失觀測對後者有實質貢獻。**
臨床試驗的 event-driven 設計針對前者（固定設計下檢定治療效果的檢定力），
不能直接搬來說「刪失資料沒資訊」。

這也是通關標準 1「刪失不是遺失資料」的定量版本：
第 2 張圖用**偏差**證明它（忽略刪失會系統性低估），這裡用**精度**證明它。
"""
from __future__ import annotations

import numpy as np

import models as M
from data import Survival, drop_censored, truncate_followup

ARMS = ("truncate", "subsample", "truncate_drop")


def _posterior_sd(idata, features: list[str]) -> dict:
    """各 β 的後驗標準差**與後驗均值**。

    均值是後來才加的，而且是必要的：只看 sd 會得出完全錯誤的結論。
    實測發現 C 臂（丟掉刪失）的後驗 sd 是三臂**最小**的 —— 但那不是因為它
    比較好，而是因為它對一個有偏的樣本很有信心。精度與正確性是兩件事，
    要同時記錄才看得出來。
    """
    beta = M.posterior_flat(idata, "beta")  # (S, p)
    sd = beta.std(axis=0, ddof=1)
    mu = beta.mean(axis=0)
    out = {f"sd_{f}": float(v) for f, v in zip(features, sd)}
    out.update({f"mean_{f}": float(v) for f, v in zip(features, mu)})
    out["sd_mean"] = float(sd.mean())
    out["sd_pnodes"] = float(sd[features.index("pnodes")])
    out["mean_pnodes"] = float(mu[features.index("pnodes")])
    return out


def stratified_subsample(s: Survival, n_events_target: int, seed: int) -> Survival:
    """分層子抽樣，讓事件數接近 target 而**事件率維持原值**。

    做法：按原始事件率反推需要的樣本數，再從事件組與刪失組分別抽同比例。
    """
    rng = np.random.default_rng(seed)
    ev_idx = np.where(s.event == 1)[0]
    cn_idx = np.where(s.event == 0)[0]
    frac = n_events_target / len(ev_idx)
    n_ev = int(round(len(ev_idx) * frac))
    n_cn = int(round(len(cn_idx) * frac))
    keep = np.concatenate([rng.choice(ev_idx, n_ev, replace=False),
                           rng.choice(cn_idx, n_cn, replace=False)])
    keep.sort()
    return Survival(X=s.X[keep], t=s.t[keep], event=s.event[keep], features=s.features)


def run_information_experiment(
    s: Survival,
    horizons=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5),
    seeds=(0, 1, 2),
    kind: str = "weibull",
    draws: int = 1000,
    tune: int = 1000,
) -> dict:
    """跑雙臂實驗，回傳兩臂的 (n, n_events, 後驗 sd) 記錄。

    A 臂對每個 horizon 只有一種資料（截短是決定性的），但仍跑多個 MCMC seed
    以量化取樣噪聲；B 臂對每個目標事件數用多個抽樣 seed（抽樣噪聲更大）。
    """
    arm_a, arm_b, arm_c = [], [], []

    def _fit_checked(s_fit, label, seed, max_retry=3):
        """取樣 + 收斂守門，失敗換 seed 重試。

        必要性見 models.sample 的踩坑說明：一條卡死的鏈會讓後驗 sd 膨脹到
        先驗 sd 的量級，在這個實驗裡剛好偽裝成「刪失越多越不確定」的漂亮趨勢。
        沒有這道守門，第一版跑出來的 A 臂 log-log 斜率是 −0.31（污染值），
        修正後是與理論相符的數字。
        """
        for attempt in range(max_retry):
            idata = M.sample(M.build(s_fit, kind), draws=draws, tune=tune,
                             chains=2, seed=seed + 7919 * attempt)
            rep = M.check_convergence(idata, label=f"{label} try{attempt}", raise_on_fail=False)
            if rep["passed"]:
                return idata, rep, attempt
        raise RuntimeError(f"[{label}] 連續 {max_retry} 次取樣未通過收斂檢查：{rep['failures']}")

    for h in horizons:
        s_tr = truncate_followup(s, h)
        if s_tr.n_events < 20:  # 事件太少，模型無法識別
            continue
        for sd_seed in seeds:
            idata, rep, retries = _fit_checked(s_tr, f"trunc h={h} s={sd_seed}", 1000 + sd_seed)
            rec = {"arm": "truncate", "horizon": h, "n": s_tr.n,
                   "n_events": s_tr.n_events, "seed": sd_seed,
                   "censoring_rate": s_tr.censoring_rate,
                   "retries": retries, "max_rhat": rep["max_rhat"]}
            rec.update(_posterior_sd(idata, s.features))
            arm_a.append(rec)

        # B 臂：對齊到 A 臂這個 horizon 產生的事件數
        for sub_seed in seeds:
            s_sub = stratified_subsample(s, s_tr.n_events, seed=2000 + sub_seed)
            idata, rep, retries = _fit_checked(s_sub, f"sub h={h} s={sub_seed}", 1000 + sub_seed)
            rec = {"arm": "subsample", "horizon": h, "n": s_sub.n,
                   "n_events": s_sub.n_events, "seed": sub_seed,
                   "censoring_rate": s_sub.censoring_rate,
                   "retries": retries, "max_rhat": rep["max_rhat"]}
            rec.update(_posterior_sd(idata, s.features))
            arm_b.append(rec)

        # C 臂：截短後把刪失整批丟掉 —— 檢定 A 臂的優勢是否真來自刪失列
        s_cd = drop_censored(s_tr)
        if s_cd.n_events >= 20:
            for c_seed in seeds:
                idata, rep, retries = _fit_checked(s_cd, f"trdrop h={h} s={c_seed}", 1000 + c_seed)
                rec = {"arm": "truncate_drop", "horizon": h, "n": s_cd.n,
                       "n_events": s_cd.n_events, "seed": c_seed,
                       "censoring_rate": s_cd.censoring_rate,
                       "retries": retries, "max_rhat": rep["max_rhat"]}
                rec.update(_posterior_sd(idata, s.features))
                arm_c.append(rec)

    return {"truncate": arm_a, "subsample": arm_b, "truncate_drop": arm_c}


def summarize_arms(res: dict, key: str = "sd_pnodes") -> dict:
    """把多 seed 記錄聚合成每個設定一個點（均值±sd），並擬合 log-log 斜率。

    理論預期：後驗 sd ∝ events^(−1/2)，也就是 log(sd) 對 log(events) 的斜率 −0.5。
    實測斜率有多接近 −0.5，是這個實驗的定量結論。
    """
    out = {}
    for arm in ARMS:
        recs = res.get(arm)
        if not recs:
            continue
        by = {}
        for r in recs:
            by.setdefault(r["horizon"], []).append(r)
        mkey = key.replace("sd_", "mean_", 1)
        pts = []
        for h, group in sorted(by.items()):
            ev = np.mean([g["n_events"] for g in group])
            n = np.mean([g["n"] for g in group])
            v = np.array([g[key] for g in group])
            mu = np.array([g[mkey] for g in group]) if mkey in group[0] else np.array([np.nan])
            pts.append({"horizon": h, "n": float(n), "n_events": float(ev),
                        "sd_mean": float(v.mean()), "sd_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        "post_mean": float(mu.mean()),
                        "censoring_rate": float(np.mean([g["censoring_rate"] for g in group]))})
        ev = np.array([p["n_events"] for p in pts])
        sd = np.array([p["sd_mean"] for p in pts])
        slope, intercept = np.polyfit(np.log(ev), np.log(sd), 1)
        out[arm] = {"points": pts, "loglog_slope": float(slope),
                    "loglog_intercept": float(intercept)}
    return out


def arm_gap(summary: dict, baseline_mean: float, key: str = "sd_mean") -> dict:
    """三臂在**相同事件數**下的精度與偏差比較 —— 本實驗的核心檢定量。

    在事件數對齊後，三臂唯一的差別是「還剩下多少刪失列與特徵向量」：
        A truncate       : n=686，刪失列全在
        B subsample      : n 縮小，刪失列按比例保留
        C truncate_drop  : n=事件數，完全沒有刪失列

    ⚠️ 這裡的假設被實測推翻了兩次，過程本身是結論：

    第一版預期 A、B 重合（「精度只由事件數決定」）。實測 A 明顯比 B 窄，
    因為 A 保留全部特徵向量、只把結果粗化成「T>h」，而迴歸係數的精度吃的
    正是協變量變異度。

    第二版預期 C 最寬（刪失列最少 → 最不精確）。實測 **C 的後驗 sd 是三臂
    最小的**。原因不是 C 比較好，而是 C 只留下早期發生事件的人 —— 它對一個
    有偏的樣本很有信心。所以本函式**同時**回報 sd 與偏差（相對全資料基準
    `baseline_mean`）：只看區間寬度來挑模型，會挑到最錯的那個。
    這與第 2 張圖互為印證：那裡 drop 的中位存活偏了 −56%。
    """
    idx = {arm: {p["horizon"]: p for p in summary[arm]["points"]}
           for arm in ARMS if arm in summary}
    common = set.intersection(*[set(v) for v in idx.values()])
    rows = []
    for h in sorted(common):
        row = {"horizon": h}
        for arm in idx:
            p = idx[arm][h]
            row[f"n_{arm}"] = p["n"]
            row[f"events_{arm}"] = p["n_events"]
            row[f"sd_{arm}"] = p[key]
            row[f"bias_{arm}"] = p["post_mean"] - baseline_mean
            row[f"abs_bias_pct_{arm}"] = 100.0 * abs(p["post_mean"] - baseline_mean) / abs(baseline_mean)
        base = row["sd_subsample"]
        row["gap_trunc_vs_sub_pct"] = 100.0 * (row["sd_truncate"] - base) / base
        if "sd_truncate_drop" in row:
            row["gap_drop_vs_trunc_pct"] = (
                100.0 * (row["sd_truncate_drop"] - row["sd_truncate"]) / row["sd_truncate"])
        rows.append(row)

    g_ts = np.array([r["gap_trunc_vs_sub_pct"] for r in rows])
    out = {"rows": rows, "baseline_mean": float(baseline_mean),
           "mean_abs_gap_pct": float(np.abs(g_ts).mean()),
           "mean_signed_gap_pct": float(g_ts.mean())}
    if rows and "gap_drop_vs_trunc_pct" in rows[0]:
        g_dt = np.array([r["gap_drop_vs_trunc_pct"] for r in rows])
        out["mean_gap_drop_vs_trunc_pct"] = float(g_dt.mean())

    # 每臂的平均 |偏差| 與平均 sd —— bias/variance 兩個維度並列
    for arm in idx:
        out[f"mean_abs_bias_pct_{arm}"] = float(
            np.mean([r[f"abs_bias_pct_{arm}"] for r in rows]))
        out[f"mean_sd_{arm}"] = float(np.mean([r[f"sd_{arm}"] for r in rows]))

    # 核心檢定：在多少個設定上，C 同時「最窄」且「偏差最大」？
    if "truncate_drop" in idx:
        trap = [r for r in rows
                if r["sd_truncate_drop"] < min(r["sd_truncate"], r["sd_subsample"])
                and r["abs_bias_pct_truncate_drop"] > max(r["abs_bias_pct_truncate"],
                                                          r["abs_bias_pct_subsample"])]
        out["n_horizons"] = len(rows)
        out["n_horizons_C_narrowest_and_most_biased"] = len(trap)
    return out
