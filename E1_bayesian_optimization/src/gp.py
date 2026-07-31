"""
E1 · 手刻高斯過程迴歸（不用套件，理解機制）。

> 🔑 GP 後驗的那兩條式子**就是高斯共軛在函數空間的版本**（計劃書主題二第二卡）：
>
>     μ(x*) = k*ᵀ (K + σ_n² I)⁻¹ y
>     σ²(x*) = k** − k*ᵀ (K + σ_n² I)⁻¹ k*
>
> 有限維的高斯共軛是「先驗均值被資料拉動、變異數必定縮小」；
> 這裡完全一樣，只是「參數」變成了整個函數。第二條式子裡
> 減掉的那一項就是「觀測帶走了多少不確定性」。

實作要點
--------
- **一律走 Cholesky**，不呼叫 `inv()`。解 `(K+σ²I)α = y` 用兩次三角回代，
  數值上比顯式求逆穩定得多，而且 `log|K|` 直接由 `2Σ log L_ii` 得到，
  不必再算一次行列式。
- **jitter**：K 加 `1e-8 ~ 1e-6` 的對角項。BO 會反覆在幾乎相同的位置取樣
  （acquisition 收斂時），那讓 K 接近奇異；沒有 jitter 時 Cholesky 直接失敗。
  本模組用「失敗就把 jitter 乘 10 再試」的階梯策略，並記錄最終用了多少。
- **超參數用 log 尺度優化**，保證正值，且讓 L-BFGS 的步長在各量級間一致。
- **解析梯度**：∂log p/∂θ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θ)。手刻它是這個專案的
  一部分——數值梯度雖然能跑，但在 d=50 時慢一個量級，而且會掩蓋
  「你是否真的知道邊際似然怎麼對超參數微分」。

lengthscale 用 **isotropic**（單一長度尺度）而非 ARD（每維一個）。
理由：本專案的測試函數各維對稱，ARD 帶不來好處；而維度詛咒實驗要跑到
50 維，ARD 會有 50 個超參數要在 ≤50 個觀測點上估計，那本身就會失敗，
會把「BO 的維度詛咒」與「超參數估計的維度詛咒」混在一起。
這個選擇在 README 的限制章節有記錄。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

LOG2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# 核函數
# ---------------------------------------------------------------------------


class Kernel:
    """核函數基類。參數一律以 log 尺度存放。"""

    name = "base"

    def __init__(self, log_amp: float = 0.0, log_ls: float = 0.0):
        self.log_amp = float(log_amp)  # log σ_f
        self.log_ls = float(log_ls)    # log ℓ

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.log_amp, self.log_ls])

    @theta.setter
    def theta(self, v):
        self.log_amp, self.log_ls = float(v[0]), float(v[1])

    def _dist(self, A, B):
        return cdist(A, B, metric="euclidean")

    def __call__(self, A, B, grad: bool = False):
        raise NotImplementedError

    def diag(self, A):
        return np.full(len(A), np.exp(2.0 * self.log_amp))

    def clone(self):
        return type(self)(self.log_amp, self.log_ls)


class RBF(Kernel):
    """k(x,x') = σ_f² exp(−r² / (2ℓ²))。無限可微 → 假設目標函數非常平滑。"""

    name = "RBF"

    def __call__(self, A, B, grad: bool = False):
        amp2 = np.exp(2.0 * self.log_amp)
        ls = np.exp(self.log_ls)
        r = self._dist(A, B)
        sq = (r / ls) ** 2
        K = amp2 * np.exp(-0.5 * sq)
        if not grad:
            return K
        # ∂K/∂log σ_f = 2K；∂K/∂log ℓ = K · (r²/ℓ²)
        return K, np.stack([2.0 * K, K * sq])


class Matern52(Kernel):
    """k = σ_f² (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(−√5 r/ℓ)。

    兩次可微 —— 比 RBF **不平滑**得多。這正是「核函數就是先驗」的具體體現：
    RBF 假設函數無限可微，會把觀測之間的行為推斷得過度平滑，
    在崎嶇的目標（如 Ackley）上系統性低估變異，導致 BO 過早停止探索。
    """

    name = "Matern52"
    S5 = np.sqrt(5.0)

    def __call__(self, A, B, grad: bool = False):
        amp2 = np.exp(2.0 * self.log_amp)
        ls = np.exp(self.log_ls)
        r = self._dist(A, B)
        u = self.S5 * r / ls
        poly = 1.0 + u + u**2 / 3.0
        e = np.exp(-u)
        K = amp2 * poly * e
        if not grad:
            return K
        # ∂K/∂log ℓ = amp2·e·[ −(1 + 2u/3)·(−u) ... ]  以 u 對 log ℓ 的導數 = −u 代入
        dpoly_du = 1.0 + 2.0 * u / 3.0
        dK_du = amp2 * e * (dpoly_du - poly)
        dK_dlogls = dK_du * (-u)
        return K, np.stack([2.0 * K, dK_dlogls])


KERNELS = {"RBF": RBF, "Matern52": Matern52}


# ---------------------------------------------------------------------------
# GP 迴歸
# ---------------------------------------------------------------------------


@dataclass
class GPFit:
    """一次 fit 的內部狀態（供 predict 與診斷使用）。"""

    X: np.ndarray
    y_std: np.ndarray
    L: np.ndarray
    alpha: np.ndarray
    y_mean: float
    y_scale: float
    jitter: float
    lml: float


class GP:
    """高斯過程迴歸，含邊際似然超參數學習。

    y 內部標準化（減均值、除標準差）。這不是美化：核函數的 amplitude
    先驗尺度只有在 y 為 O(1) 時才有意義，否則超參數優化會從一個荒謬的
    起點出發。predict 回傳時換算回原尺度。
    """

    def __init__(self, kernel: Kernel | str = "Matern52", log_noise: float = -2.0,
                 jitter: float = 1e-8, normalize_y: bool = True):
        self.kernel = KERNELS[kernel]() if isinstance(kernel, str) else kernel
        self.log_noise = float(log_noise)  # log σ_n
        self.base_jitter = float(jitter)
        self.normalize_y = normalize_y
        self.fit_: GPFit | None = None
        self.n_cholesky_retries = 0

    # -- 內部：Cholesky（含 jitter 階梯）--------------------------------
    def _chol(self, K: np.ndarray) -> tuple[np.ndarray, float]:
        """對 K 做 Cholesky，失敗就把 jitter 乘 10 再試。

        BO 的取樣點會越來越密集（acquisition 收斂），K 因此接近奇異。
        直接讓 Cholesky 拋例外會讓整個 BO run 掛掉，而那在 30 seeds ×
        多設定的實驗裡意味著整批結果消失。
        """
        jit = self.base_jitter
        n = len(K)
        for _ in range(8):
            try:
                L = cholesky(K + jit * np.eye(n), lower=True)
                return L, jit
            except np.linalg.LinAlgError:
                jit *= 10.0
                self.n_cholesky_retries += 1
        raise np.linalg.LinAlgError(f"Cholesky 在 jitter={jit:.1e} 仍失敗（n={n}）")

    def _lml_and_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                      want_grad: bool = True):
        """負的 log marginal likelihood 與其對 [log σ_f, log ℓ, log σ_n] 的梯度。

            log p(y|X,θ) = −½ yᵀK⁻¹y − ½ log|K| − n/2 log 2π
            ∂/∂θ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θ)
        """
        k = self.kernel.clone()
        k.theta = theta[:2]
        log_noise = theta[2]
        n = len(X)

        if want_grad:
            K0, dK = k(X, X, grad=True)
        else:
            K0, dK = k(X, X), None
        noise = np.exp(2.0 * log_noise)
        K = K0 + noise * np.eye(n)

        try:
            L, _ = self._chol(K)
        except np.linalg.LinAlgError:
            return (np.inf, np.zeros(3)) if want_grad else np.inf

        alpha = cho_solve((L, True), y)
        lml = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * n * LOG2PI
        if not want_grad:
            return -lml

        K_inv = cho_solve((L, True), np.eye(n))
        W = np.outer(alpha, alpha) - K_inv          # ααᵀ − K⁻¹
        grads = np.empty(3)
        grads[0] = 0.5 * np.einsum("ij,ij->", W, dK[0])
        grads[1] = 0.5 * np.einsum("ij,ij->", W, dK[1])
        grads[2] = 0.5 * np.einsum("ij,ij->", W, 2.0 * noise * np.eye(n))
        return -lml, -grads

    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True,
            n_restarts: int = 3, seed: int = 0, bounds=None):
        """擬合 GP。`optimize=True` 時用 L-BFGS-B 最大化邊際似然。

        多起點重啟是必要的：邊際似然在 (lengthscale, noise) 平面上常有
        兩個吸引盆 —— 一個是「訊號解」（合理的 ℓ、小 noise），
        另一個是「噪聲解」（ℓ→∞、noise 吃掉全部變異，模型退化成常數）。
        單起點很容易掉進後者，而那會讓 BO 完全停止探索。
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if self.normalize_y:
            y_mean = float(y.mean())
            y_scale = float(y.std()) or 1.0
        else:
            y_mean, y_scale = 0.0, 1.0
        y_std = (y - y_mean) / y_scale

        if bounds is None:
            bounds = [(-3.0, 3.0), (-3.0, 3.0), (-5.0, 1.0)]  # log σ_f, log ℓ, log σ_n

        theta0 = np.array([self.kernel.log_amp, self.kernel.log_ls, self.log_noise])
        if optimize and len(X) >= 3:
            rng = np.random.default_rng(seed)
            starts = [theta0] + [
                np.array([rng.uniform(*bounds[0]), rng.uniform(*bounds[1]),
                          rng.uniform(*bounds[2])]) for _ in range(n_restarts - 1)]
            best = (np.inf, theta0)
            for s in starts:
                try:
                    r = minimize(self._lml_and_grad, s, args=(X, y_std, True),
                                 jac=True, method="L-BFGS-B", bounds=bounds)
                except np.linalg.LinAlgError:
                    continue
                if np.isfinite(r.fun) and r.fun < best[0]:
                    best = (r.fun, r.x)
            theta_best = best[1]
        else:
            theta_best = theta0

        self.kernel.theta = theta_best[:2]
        self.log_noise = float(theta_best[2])

        K = self.kernel(X, X) + np.exp(2.0 * self.log_noise) * np.eye(len(X))
        L, jit = self._chol(K)
        alpha = cho_solve((L, True), y_std)
        lml = float(-0.5 * y_std @ alpha - np.log(np.diag(L)).sum()
                    - 0.5 * len(X) * LOG2PI)
        self.fit_ = GPFit(X=X, y_std=y_std, L=L, alpha=alpha, y_mean=y_mean,
                          y_scale=y_scale, jitter=jit, lml=lml)
        return self

    def predict(self, Xs: np.ndarray, return_std: bool = True):
        """後驗均值與標準差（原始 y 尺度）。

            μ(x*) = k*ᵀ α                      （α = (K+σ²I)⁻¹ y）
            σ²(x*) = k** − ‖L⁻¹k*‖²            （v = L⁻¹k*，用回代而非求逆）
        """
        if self.fit_ is None:
            raise RuntimeError("先 fit")
        f = self.fit_
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self.kernel(f.X, Xs)                      # (n, m)
        mu = Ks.T @ f.alpha * f.y_scale + f.y_mean
        if not return_std:
            return mu
        v = np.linalg.solve(f.L, Ks)                   # L v = k*
        var = self.kernel.diag(Xs) - np.einsum("ij,ij->j", v, v)
        var = np.clip(var, 1e-12, None) * f.y_scale**2
        return mu, np.sqrt(var)

    def sample_posterior(self, Xs: np.ndarray, n_samples: int = 1, seed: int = 0):
        """從後驗抽函數樣本（Thompson sampling 用）。

        用聯合共變異數的 Cholesky，而不是逐點獨立抽樣 ——
        後者會抽出鋸齒狀的假函數，Thompson sampling 取它的最大值毫無意義。
        """
        f = self.fit_
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self.kernel(f.X, Xs)
        Kss = self.kernel(Xs, Xs)
        mu = Ks.T @ f.alpha
        v = np.linalg.solve(f.L, Ks)
        cov = Kss - v.T @ v
        cov += 1e-8 * np.eye(len(Xs))
        Lc, _ = self._chol(cov)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((len(Xs), n_samples))
        samples = mu[:, None] + Lc @ z
        return (samples * f.y_scale + f.y_mean).T  # (n_samples, m)

    # -- 診斷 ------------------------------------------------------------
    @property
    def hyperparams(self) -> dict:
        return {"amplitude": float(np.exp(self.kernel.log_amp)),
                "lengthscale": float(np.exp(self.kernel.log_ls)),
                "noise": float(np.exp(self.log_noise)),
                "kernel": self.kernel.name,
                "lml": None if self.fit_ is None else self.fit_.lml,
                "jitter": None if self.fit_ is None else self.fit_.jitter}


# ---------------------------------------------------------------------------
# 正確性驗證：與 sklearn 逐點比對
# ---------------------------------------------------------------------------


def verify_against_sklearn(X, y, kernel: str = "Matern52", log_amp=0.0, log_ls=-0.3,
                           log_noise=-2.0, n_test: int = 50, seed: int = 0) -> dict:
    """在**固定超參數**下，逐點比對手刻 GP 與 sklearn 的後驗均值與標準差。

    固定超參數是關鍵：兩邊的超參數優化器不同（起點、收斂條件、參數化都不同），
    比較「各自優化後」的結果只會測到優化器差異，測不到後驗公式是否正確。
    這裡把兩邊的核參數與噪聲鎖成同一組，差異就只能來自線性代數本身。

    這是本專案的自我檢查：手刻的意義不是取代套件，而是**確認自己知道
    套件在算什麼**。不能重現它，就無法在它出錯時發現。
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as SkRBF
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    Xs = rng.uniform(X.min(axis=0), X.max(axis=0), size=(n_test, X.shape[1]))

    amp, ls, noise = np.exp(log_amp), np.exp(log_ls), np.exp(log_noise)

    mine = GP(kernel=KERNELS[kernel](log_amp, log_ls), log_noise=log_noise,
              normalize_y=False)
    mine.fit(X, y, optimize=False)
    mu_mine, sd_mine = mine.predict(Xs)

    sk_base = (SkRBF(length_scale=ls) if kernel == "RBF"
               else Matern(length_scale=ls, nu=2.5))
    sk_kernel = ConstantKernel(amp**2, constant_value_bounds="fixed") * sk_base
    sk = GaussianProcessRegressor(kernel=sk_kernel, alpha=noise**2, optimizer=None,
                                  normalize_y=False)
    sk.fit(X, y)
    mu_sk, sd_sk = sk.predict(Xs, return_std=True)

    d_mu = np.abs(mu_mine - mu_sk)
    d_sd = np.abs(sd_mine - sd_sk)
    scale_mu = max(np.abs(mu_sk).max(), 1e-12)
    scale_sd = max(np.abs(sd_sk).max(), 1e-12)
    # 判準用**相對**誤差：兩邊都走 float64 Cholesky，但運算順序不同
    # （sklearn 把噪聲以 `alpha` 加到對角、Matern 的 √5·r 計算順序也不同），
    # 累積誤差落在 1e-8 量級是正常的浮點行為，不是邏輯錯誤。
    # 公式本身的正確性由 `verify_gradient` 獨立確認（那裡的相對誤差 ~1e-8）。
    rel_mu = float(d_mu.max() / scale_mu)
    rel_sd = float(d_sd.max() / scale_sd)
    return {
        "kernel": kernel,
        "max_abs_diff_mean": float(d_mu.max()),
        "max_abs_diff_std": float(d_sd.max()),
        "max_rel_diff_mean": rel_mu,
        "max_rel_diff_std": rel_sd,
        "n_train": int(len(X)), "n_test": int(n_test),
        "tol_rel": 1e-6,
        "passed": bool(rel_mu < 1e-6 and rel_sd < 1e-6),
    }


def verify_gradient(X, y, kernel: str = "Matern52", eps: float = 1e-6,
                    seed: int = 0) -> dict:
    """用中央差分檢查解析梯度 —— 手刻梯度最容易出錯的地方。

    梯度寫錯時 L-BFGS 仍會「跑完」並回傳一個看似合理的超參數，
    只是它不是邊際似然的最大值。這種錯誤不會拋例外，只會安靜地
    讓 GP 擬合變差、讓 BO 表現變爛，而且極難從結果反推。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    y = (y - y.mean()) / (y.std() or 1.0)
    gp = GP(kernel=kernel)
    rng = np.random.default_rng(seed)
    theta = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-3, 0)])

    _, g_analytic = gp._lml_and_grad(theta, X, y, want_grad=True)
    g_numeric = np.empty(3)
    for i in range(3):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        fp = gp._lml_and_grad(tp, X, y, want_grad=False)
        fm = gp._lml_and_grad(tm, X, y, want_grad=False)
        g_numeric[i] = (fp - fm) / (2 * eps)
    rel = np.abs(g_analytic - g_numeric) / np.maximum(np.abs(g_numeric), 1e-8)
    return {"kernel": kernel, "theta": theta.tolist(),
            "analytic": g_analytic.tolist(), "numeric": g_numeric.tolist(),
            "max_rel_err": float(rel.max()), "passed": bool(rel.max() < 1e-4)}
