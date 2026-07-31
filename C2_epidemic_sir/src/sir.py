"""
C2 · 離散時間 SIR 模型。

連續 SIR 是 ODE：
    dS/dt = −β S I / N,   dI/dt = β S I / N − γ I,   dR/dt = γ I

本專案用**每日離散版**，而不是丟給 ODE solver：

    new_infections_t = β_t · S_{t-1} · I_{t-1} / N
    new_recoveries_t = γ · I_{t-1}
    S_t = S_{t-1} − new_infections_t
    I_t = I_{t-1} + new_infections_t − new_recoveries_t
    R_t = R_{t-1} + new_recoveries_t

三個理由：
1. **觀測本身就是日資料。** 通報以「天」為單位彙總，日步長的離散化與資料
   結構對齊，不需要在 solver 的自適應步長與日彙總之間再做一次插值。
2. **NUTS 需要梯度。** `pymc.ode.DifferentialEquation` 每次評估都要解 ODE 並
   反向傳梯度，比一個 `pytensor.scan` 慢好幾個量級；在 122 天 × 122 個時變參數
   的模型上根本跑不動。
3. **時變 β 天然合適。** β_t 逐日改變在遞迴式裡就是把純量換成向量，
   在 ODE 形式裡則要處理係數的時間插值。

代價是離散化誤差：日步長對 R₀≈3、病程 10 天的疫情是可接受的
（每日感染比例遠小於 1），但在爆炸性成長階段會低估峰值。
README 的限制章節記錄這點。

**本模組提供 numpy 與 pytensor 兩份實作，並逐點驗證等價。**
理由：`scan` 寫錯（初始值、輸出順序、taps）不會拋錯，只會安靜給出
一條錯的軌跡，而那會被後續的 NUTS 擬合「吸收」成一組奇怪但收斂的參數。
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# numpy 版：可讀、可獨立檢查、用於模擬與資料生成
# ---------------------------------------------------------------------------


def simulate_numpy(beta, gamma: float, S0: float, I0: float, N: float,
                   n_days: int | None = None) -> dict:
    """離散 SIR 前向模擬（numpy）。

    `beta` 可以是純量（固定）或長度 n_days 的陣列（時變）。
    回傳每日的 S/I/R 與**新感染數**（後者才是觀測模型要用的量 ——
    通報的是新病例，不是感染者存量）。
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if n_days is None:
        n_days = len(beta) if len(beta) > 1 else 1
    if len(beta) == 1:
        beta = np.repeat(beta, n_days)
    assert len(beta) == n_days, f"beta 長度 {len(beta)} != n_days {n_days}"

    S = np.empty(n_days)
    I = np.empty(n_days)
    R = np.empty(n_days)
    new_inf = np.empty(n_days)

    s, i, r = float(S0), float(I0), float(N - S0 - I0)
    for t in range(n_days):
        ni = beta[t] * s * i / N
        ni = min(ni, s)                      # 不能感染比剩餘易感者更多的人
        nr = gamma * i
        s = s - ni
        i = i + ni - nr
        r = r + nr
        S[t], I[t], R[t], new_inf[t] = s, i, r, ni
    return {"S": S, "I": I, "R": R, "new_infections": new_inf}


def reproduction_number(beta, gamma: float, S, N: float) -> np.ndarray:
    """有效再生數 R_t = (β_t / γ) · (S_t / N)。

    注意兩個不同的量：
      **基本**再生數 R₀ = β/γ —— 全體易感時，一個感染者平均傳給幾個人
      **有效**再生數 R_t = R₀ · S_t/N —— 扣掉已免疫人口後的實際值

    疫情早期 S/N ≈ 1，兩者幾乎相同；本專案的時段（德國 2020 第一波，
    累計感染佔人口 0.2%）也是如此，所以 R_t 的變化幾乎全部來自 β_t，
    也就是**行為與政策**，不是群體免疫。這個區別在解讀 R_t 下降的原因時很關鍵。
    """
    beta = np.asarray(beta, dtype=float)
    S = np.asarray(S, dtype=float)
    return (beta / gamma) * (S / N)


# ---------------------------------------------------------------------------
# pytensor 版：供 PyMC 推論（需要梯度）
# ---------------------------------------------------------------------------


def simulate_pytensor(beta, gamma, S0, I0, N, n_days: int):
    """離散 SIR 前向模擬（pytensor scan），回傳 (S, I, R, new_infections)。

    `beta` 必須是長度 n_days 的 pytensor 向量（固定 β 就傳重複值）。
    """
    import pytensor
    import pytensor.tensor as pt

    # ⚠️ 初始狀態必須與 `beta` 同 dtype。否則 scan 會用 pytensor 的 floatX
    # （可能是 float32）建立 outputs_info，而 inner function 的運算結果是
    # float64，於是報 "initial state has dtype float32, while the result of
    # the inner function has dtype float64"。這個錯誤只在編譯期出現，
    # 而且訊息指向 scan 內部，不容易看出根因是 dtype 而不是邏輯。
    dt = beta.dtype
    S0_t = pt.cast(pt.as_tensor_variable(S0), dt)
    I0_t = pt.cast(pt.as_tensor_variable(I0), dt)
    N_t = pt.cast(pt.as_tensor_variable(N), dt)
    R0_t = N_t - S0_t - I0_t
    gamma_t = pt.cast(pt.as_tensor_variable(gamma), dt)

    def step(beta_t, s_prev, i_prev, r_prev, N_, gam):
        ni = beta_t * s_prev * i_prev / N_
        ni = pt.minimum(ni, s_prev)
        nr = gam * i_prev
        s = s_prev - ni
        i = i_prev + ni - nr
        r = r_prev + nr
        return s, i, r, ni

    (S, I, R, NI), _ = pytensor.scan(
        fn=step,
        sequences=[beta],
        outputs_info=[
            {"initial": S0_t},
            {"initial": I0_t},
            {"initial": R0_t},
            None,                      # new_infections 不需要遞迴狀態
        ],
        non_sequences=[N_t, gamma_t],
        n_steps=n_days,
        strict=True,
    )
    return S, I, R, NI


def verify_scan_matches_numpy(n_days: int = 120, seed: int = 0, tol: float = 1e-9) -> dict:
    """逐點比對 pytensor scan 與 numpy 實作。

    測兩種情形：固定 β 與時變 β（後者才是本專案用的，也才會暴露
    sequences/taps 接錯的錯誤）。

    這個檢查是必要的而非保險：scan 的 `outputs_info` 順序、`sequences`
    對齊、以及 `None` 佔位（表示該輸出不回饋）都很容易寫錯，
    而錯了之後模型仍能跑、仍能收斂，只是在擬合一個不是 SIR 的動態系統。
    """
    import pytensor
    import pytensor.tensor as pt

    rng = np.random.default_rng(seed)
    N = 84.1e6
    I0, gamma = 200.0, 0.1
    S0 = N - I0

    out = {}
    for label, beta_np in (
        ("constant_beta", np.repeat(0.28, n_days)),
        ("time_varying_beta", 0.28 * np.exp(np.cumsum(rng.normal(0, 0.05, n_days)))),
    ):
        ref = simulate_numpy(beta_np, gamma, S0, I0, N, n_days)

        beta_t = pt.dvector("beta")
        S, I, R, NI = simulate_pytensor(beta_t, gamma, S0, I0, N, n_days)
        f = pytensor.function([beta_t], [S, I, R, NI], on_unused_input="ignore")
        s2, i2, r2, ni2 = f(beta_np)

        errs = {}
        for k, a, b in (("S", ref["S"], s2), ("I", ref["I"], i2), ("R", ref["R"], r2),
                        ("new_infections", ref["new_infections"], ni2)):
            scale = max(np.abs(a).max(), 1.0)
            errs[k] = float(np.abs(a - b).max() / scale)
        out[label] = {"max_rel_err": errs, "worst": float(max(errs.values())),
                      "passed": bool(max(errs.values()) < tol)}
    out["passed"] = all(v["passed"] for k, v in out.items() if k != "passed")
    return out


# ---------------------------------------------------------------------------
# 供 R_t 與觀測模型使用的小工具
# ---------------------------------------------------------------------------


def expected_cases(new_infections, reporting_rate: float, dow_effect=None, dow=None):
    """把「新感染數」轉成「預期通報病例數」。

        E[cases_t] = reporting_rate · new_infections_t · dow_multiplier(t)

    兩件事在這裡合流：
    - **通報率** < 1：多數感染從未被檢驗到（無症狀、輕症、檢驗量能）。
      它與 I0 高度相關（都能解釋「觀測到的量級」），所以無法只靠病例數
      分開估計 —— README 的限制章節記錄這個識別問題。
    - **week-day 效應**：實測德國 2020 第一波週一 0.78×、週五 1.25×
      （差 60%）。那是通報流程的節奏，不是傳染動態；不建模的話
      殘差會留下 7 天週期，並被過度分散參數吸收成「疫情很吵」。
    """
    exp = reporting_rate * np.asarray(new_infections, dtype=float)
    if dow_effect is not None and dow is not None:
        exp = exp * np.asarray(dow_effect)[np.asarray(dow)]
    return exp
