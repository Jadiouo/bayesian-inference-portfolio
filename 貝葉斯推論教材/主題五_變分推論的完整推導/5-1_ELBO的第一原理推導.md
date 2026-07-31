---
title: 5-1 · ELBO 的第一原理推導
theme: 主題五 · 變分推論的完整推導
tags:
  - 貝葉斯推論
  - 主題五
  - ELBO
prev: "[[4-3_MCMC的核心直覺]]"
next: "[[5-2_MeanField與CAVI]]"
type: course-card
created: 2026-07-28
card_no: "5-1"
theme_no: 五
read: false
projects: [S1]
---

# 5-1 · ELBO 的第一原理推導

> [!abstract] ELBO 是現代深度學習的幕後英雄
> VAE、Diffusion、貝葉斯神經網路、半監督學習 —— **它們所有的訓練 loss,都是 ELBO 的某種化身。**
>
> 你會看到一個幾乎是魔法的技巧:讓「算不出來的後驗」和「算不出來的 $p(D)$」,在同一條式子裡**互相消掉,留下一個可以計算的目標函數**。

> [!question] 本卡核心問題
> - VI 的目標 $\text{KL}(q\,\|\,p(\theta\mid D))$ 裡有後驗 —— **而後驗就是我們算不出來的東西**。怎麼辦?
> - ELBO 怎麼漂亮地解決這個悖論?
> - 為什麼它叫「Evidence **Lower Bound**」?

---

## ① 悖論:用魔法找魔法書

VI 的目標:

$$\phi^* = \arg\min_\phi\text{KL}\big(q_\phi(\theta)\,\|\,p(\theta\mid D)\big)$$

展開 KL:

$$\text{KL}(q_\phi\|p(\theta|D)) = \mathbb{E}_{q_\phi}\big[\log q_\phi(\theta) - \log p(\theta\mid D)\big]$$

> 🚨 **問題**:這個式子裡有 $\log p(\theta\mid D)$ —— **這就是我們算不出來的真實後驗本身!**
>
> **我們想用 VI 近似後驗,但 VI 的目標式子裡已經出現了後驗。**

---

## ② 天才解法:把後驗拆開

用貝葉斯定理打開 $p(\theta\mid D) = \dfrac{p(D\mid\theta)p(\theta)}{p(D)}$,代入 KL:

$$\text{KL}(q\|p(\theta|D)) = \mathbb{E}_q\big[\log q(\theta) - \log p(D\mid\theta) - \log p(\theta) + \log p(D)\big]$$

**注意第四項 $\log p(D)$:它不依賴 $\theta$**,對 $q$ 取期望時就是常數,可以提出來:

$$\text{KL}(q\|p(\theta|D)) = \underbrace{-\mathbb{E}_q[\log p(D\mid\theta) + \log p(\theta) - \log q(\theta)]}_{= -\text{ELBO}} + \log p(D)$$

整理成最終形式:

$$\boxed{\;\log p(D) = \underbrace{\mathbb{E}_q[\log p(D\mid\theta) + \log p(\theta) - \log q(\theta)]}_{\text{ELBO}(q)} + \text{KL}\big(q\,\|\,p(\theta\mid D)\big)\;}$$

### 🔊 念出來

> 「log p(D),等於,ELBO,加上,q 和真實後驗的 KL 散度」

### 三個項各是什麼?

| 項 | 是什麼 | 我們的關係 |
|---|---|---|
| $\log p(D)$ | 邊際似然的對數(evidence) | **常數** —— 不依賴 $q$(數據已固定) |
| **ELBO**$(q)$ | **可計算**(不需要 $p(D)$!) | 我們可以算、可以對 $q$ 優化 |
| **KL**$(q\|p(\theta\mid D))$ | $q$ 和真實後驗的差距 | **算不出來**(含 $p(\theta\mid D)$) |

---

## ③ 🪄 魔法:最大化 ELBO = 最小化 KL

由於 $\log p(D)$ 是**常數**:

$$\text{ELBO}(q) + \text{KL}(q\|p(\theta|D)) = \text{常數}$$

**兩者是互補的 —— 一個變大,另一個就變小。**

> [!important] 🔑 核心魔法
> **最大化 ELBO 等價於最小化 KL(q ‖ 真後驗)。**
>
> **而 ELBO 可以算,KL 不能算。**
>
> **所以:我們最大化可以算的 ELBO,就間接最小化了不能算的 KL。**

### 🖼️ 畫面

```
log p(D) ────────────────  天花板(固定不動)
              ↑
            KL(差距)      ← 我們想讓它小
              ↓
           ELBO           ← 我們最大化這個

   ELBO 往上頂 → KL 被擠小
```

### 為什麼叫「Lower Bound」?

因為 KL $\ge 0$,所以:

$$\text{ELBO}(q) \le \log p(D)$$

**ELBO 永遠不會超過 $\log p(D)$ —— 它是 evidence 的下界。**

> [!note] 一個容易被忽略的哲學點
> 這條等式**不是「巧合」**,而是我們**刻意**把 $\log p(D)$ 這個常數,拆成「ELBO + KL」這個有用的分解。
>
> **常數總是可以等於「東西 A + 東西 B」,但拆成這個形狀,讓我們能透過「最大化可算的 A」來「間接最小化不可算的 B」** —— 這是 ELBO 的核心工程價值。

---

## ④ ELBO 的內部結構:擬合 − 正則

把 ELBO 重新組合成 VAE 教科書的常見形式:

$$\boxed{\;\text{ELBO}(q) = \underbrace{\mathbb{E}_q[\log p(D\mid\theta)]}_{\text{重建項}} - \underbrace{\text{KL}\big(q(\theta)\,\|\,p(\theta)\big)}_{\text{正則項}}\;}$$

> 🔊 **念**:「ELBO,等於,重建項,**減**,正則項」
>
> ⚠️ 注意是**相減**,不是相加。

### 🧱 重建項:$\mathbb{E}_q[\log p(D\mid\theta)]$

**「在 q 採樣的 θ 下,數據的對數似然有多大?」**

最大化它 → 讓 $q$ 選擇能**解釋數據**的 $\theta$ → **MLE 的精神**

### 🛡️ 正則項:$\text{KL}(q(\theta)\,\|\,p(\theta))$

**「q 偏離先驗 $p(\theta)$ 有多遠?」**

最小化它(因為前面有負號)→ 讓 $q$ 接近**先驗** → **MAP / L2 正則的精神**

### 🔑 一句話

> **ELBO = 擬合 − 正則**
>
> **這正是貝葉斯推論的核心精神(先驗 + 似然 → 後驗),只是用 $q$ 同時實現這兩件事。**

### 和 MAP / L2 的關係

回想 [[1-1_四個項的認識論意義]]:

$$\hat\theta_{\text{MAP}} = \arg\max_\theta\big[\log p(D\mid\theta) + \log p(\theta)\big]$$

如果先驗是 $\mathcal{N}(0,\sigma_0^2 I)$,則 $\log p(\theta) = -\frac{1}{2\sigma_0^2}\|\theta\|^2 + \text{const}$ —— **這就是 L2 正則化**。

| 方法 | 重建項 | 正則項 |
|---|---|---|
| **MAP + L2** | $\log p(D\mid\theta)$(似然) | $-\lambda\|\theta\|^2$(高斯先驗) |
| **VI / ELBO** | $\mathbb{E}_q[\log p(D\mid\theta)]$(**期望**似然) | $-\text{KL}(q\|p(\theta))$(**分佈級**先驗) |

> 🔑 **ELBO 是 MAP 的升級版:從「點估計 + 點正則」升級到「分佈估計 + 分佈正則」。**
>
> 這正是貝葉斯比 MAP 強的地方 —— **不只給你一個點,還給你一整個分佈的不確定性。**

### 🚨 如果拿掉 KL 項會怎樣?

> **$q$ 會 collapse 到一個點。**
>
> 只剩重建項時,$q$ 為了最大化它,會把所有機率質量**塌縮到 $\theta_{\text{MLE}}$** —— 一個 delta 函數!
>
> **這時 $q$ 不再是「分佈」,只是一個「點」** —— 你失去了所有不確定性資訊,退化成 MLE。
>
> **KL 正則的角色:把 $q$「撐開」,不讓它塌縮。**

**這在 VAE 文獻裡叫 posterior collapse,是有名的失敗模式。** → [[5-3_重參數化技巧]]

---

## ⑤ 反直覺:你做的所有貝葉斯訓練都在最大化 ELBO

### 🎨 VAE 的訓練 = 最大化 ELBO

$$\mathcal{L}_{\text{VAE}} = -\underbrace{\mathbb{E}_q[\log p(x\mid z)]}_{\text{重建損失}} + \underbrace{\text{KL}(q(z\mid x)\,\|\,p(z))}_{\text{KL 正則}}$$

**這就是 $-$ELBO!**

- 重建損失:解碼器要重建出輸入
- KL 正則:讓編碼器輸出的 $q(z\mid x)$ 接近 $\mathcal{N}(0,I)$

> 🔑 **「VAE 的 KL 項是哪裡來的」這個困擾初學者的問題,答案就是:它是 ELBO 推導出來的正則項。**

### 🔄 EM 算法 = 交替最大化 ELBO

- **E-step**:固定 $\theta$,優化 $q$ → 最小化 KL
- **M-step**:固定 $q$,優化 $\theta$ → 最大化重建項

→ [[5-2_MeanField與CAVI]]

### 🌊 Diffusion Model = ELBO 在時間軸的展開

把單步分解成多步,每步做一個小的 KL 最小化。**訓練 diffusion 本質上就是在優化 ELBO。**

> 🔑 **整個現代深度生成模型(VAE、Diffusion、Normalizing Flow)的訓練,都建立在最大化 ELBO 上。**

---

## ⑥ 電機工程類比:擬合 + 正則化

你用有限階數的模型 $\hat s_\phi(t)$ 擬合複雜訊號 $s(t)$。目標:

1. **擬合好**(讓 $\hat s$ 接近 $s$)
2. **不要過擬合**(模型參數不要太離譜)

訊號處理裡你會寫:

$$\mathcal{L} = \underbrace{\|s - \hat s_\phi\|^2}_{\text{擬合誤差}} + \lambda\underbrace{\|\phi\|^2}_{\text{正則化}}$$

**這就是 $-$ELBO 的訊號處理版本!**

> 🔑 **ELBO 本質上是「擬合 + 正則化」的貝葉斯版本。差別只在**:
> - 訊號處理的正則化是**手動加的懲罰項**
> - **ELBO 的正則化是從貝葉斯框架自然導出的 $\text{KL}(q\|p)$**

---

## 🔗 回頭連結

| 你已學過的 | 在 ELBO 的位置 |
|---|---|
| **KL 散度** | ELBO 的核心構件 |
| **交叉熵 / MLE** | 重建項 = 對數似然 = MLE 精神 |
| **MAP / L2 正則** | KL 正則項 = 先驗約束的分佈版本 |
| **貝葉斯定理** | ELBO 的推導從拆開 $p(\theta\mid D)$ 開始 |
| **$p(D)$ 的困境** | ELBO 是「不算 $p(D)$ 也能做貝葉斯」的解答 |
| **VAE / Diffusion** | 它們的 loss 就是 $-$ELBO |

---

## 🎓 費曼檢驗

### 問題 1
為什麼「最大化 ELBO」等價於「最小化 KL(q ‖ 真後驗)」?關鍵是哪一項是常數?

> [!success]- 參考答案
> 因為 $\log p(D) = \text{ELBO} + \text{KL}$,而 **$\log p(D)$ 是常數**(數據已固定,不依賴 $q$)。
>
> 常數 = A + B,所以 A 變大 → B 必定變小。
>
> **而 ELBO 可算、KL 不可算 —— 所以我們最大化可算的那個,間接最小化不可算的那個。**
>
> ⚠️ 小修正:常數是 $\log p(D)$,不是 $D$。$D$ 是數據(當然固定),但在等式裡當常數的是它的邊際似然對數。

### 問題 2
ELBO 兩個項各在做什麼?只有重建項會怎樣?

> [!success]- 參考答案
> - **重建項** $\mathbb{E}_q[\log p(D\mid\theta)]$:「q 採樣的模型能不能解釋數據?」→ **MLE 的精神**
> - **KL 正則項** $\text{KL}(q\|p(\theta))$:「q 有沒有離先驗太遠?」→ **MAP / L2 正則的精神**
>   (回憶:MAP + 高斯先驗 = L2 正則化)
>
> **只有重建項** → **$q$ 會 collapse 到一個 delta 函數**,塌縮到 $\theta_{\text{MLE}}$。這時 $q$ 不再是分佈,只是一個點 —— 你失去了所有不確定性,退化成 MLE。
>
> **KL 的角色就是「把 q 撐開,不讓它塌縮」。** 在 VAE 裡這叫 posterior collapse。

### 問題 3
「Evidence」指什麼?為什麼 ELBO 是它的「下界」?ELBO 上升代表什麼?

> [!success]- 參考答案
> - **Evidence = $p(D)$**,邊際似然。更精準地說,是「數據本身在這個模型下發生的機率」(不只是「數據」)。
> - **為什麼是下界**:$\log p(D) = \text{ELBO} + \text{KL}$,而 **KL ≥ 0**,所以 $\text{ELBO}\le\log p(D)$。ELBO 是「從下面頂上去」的天花板下最高樓。
> - **ELBO 上升 = 三件事同時發生**:
>   1. KL(q ‖ 真後驗)縮小 → q 越來越接近真實後驗
>   2. 間接逼近 $\log p(D)$
>   3. 訓練 loss 下降(VAE 的 loss = $-$ELBO)
>
> ⚠️ 注意:**你不能從 ELBO 的值推斷 KL 的絕對值** —— 因為 $\log p(D)$ 是未知常數,你不知道天花板在哪。但你知道 ELBO 越大,KL 越小。

---

## 🧪 我的實作對應

> [!example] 這張卡片的理論，在作品集裡的實測
>
> - [[S1 VAE 異常偵測]] —— ELBO 拆成重建項與 KL 各自獨立驗證；並實測「收緊下界（IWAE）對下游任務完全無效」（0.2σ）
>   （[程式碼與完整敘事](https://github.com/Jadiouo/bayesian-inference-portfolio/tree/main/S1_vae_anomaly)）

---

## 導覽

⬅️ [[4-3_MCMC的核心直覺]] | ➡️ [[5-2_MeanField與CAVI]]
