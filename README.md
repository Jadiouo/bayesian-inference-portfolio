# 貝葉斯推論 · 跨領域實作作品集

> **一套數學，五個領域，十個完整專案。**
> 骨架都是同一條式子：**p(θ | D) ∝ p(D | θ) · p(θ)**
>
> 核心能力敘事：**「我能量化模型的不確定性，並把它轉換成可執行的決策。」**
> —— 在醫療是「該不該轉診」，在天文是「這是不是行星」，在商業是「該不該上線」。

每個專案都是完整的：`src/` 模組化程式 + 已執行的 notebook + 圖表 + 帶真實數字的 README，
`python src/run_all.py` 一鍵重現。**十個專案全部完成**。

---

## 這個作品集想證明的一件事

> **每個專案都用一個「模型之外」的獨立檢查，推翻了自己的第一版結論。**

這是刻意的方法論，不是意外：

| 專案 | 第一版看起來 | 獨立檢查之後 |
|---|---|---|
| **A2** 生存分析 | 「後驗窄 = 估得準」 | 最窄的那個**偏差 83%** —— 對有偏樣本很有信心 |
| **A3** 影像 OOD | 「模型偵測到 OOD」 | 不看模型的像素統計 **4 組贏 3 組** |
| **S1** VAE 異常偵測 | 「ELBO 偵測到病灶」 | 兩行 numpy 的梯度能量**打敗 VAE 16 個百分點** |
| **C2** SIR 傳播模型 | 0 divergences、r̂=1.000，收斂完美 | 該解需要 **86.8% 人口被感染** —— 模型失配偽裝成參數估計 |
| **E1** 貝葉斯最優化 | 「高維失敗是內層優化解不動」 | 實測 shortfall≈0，真因是 **GP 退化成先驗** |
| **B2** 貝氏因子 | 「先驗範圍是個可調的旋鈕」 | 放寬 3 倍就翻盤 —— **先驗範圍就是模型定義** |

完整的方法論總結見 **[`學習筆記.md`](學習筆記.md)** —— 十個專案抽出的六條共同主線、
每個專案的觀察與踩坑、七個理論主題 × 專案對照。

---

## 三張代表圖

**理論與實作對得上**：階層模型的收縮量 ω 隨樣本數變化，緊貼理論的精確度加權曲線 σ²/(σ²+nσ_a²)

![shrinkage](C1_hierarchical_pooling/figures/02_shrinkage.png)

**奧卡姆剃刀是可測量的**：食雙星模型的最大似然高 +2.7，logZ 卻低 2.98 —— 多出來的參數體積被懲罰掉

![evidence](B2_bayes_factor/figures/03_evidence.png)

**誠實的對照基準**：VAE 的 ELBO 拿 0.756，一個不看模型的梯度能量統計拿 0.915

![trivial](S1_vae_anomaly/figures/04_trivial_baseline.png)

---

## 十個專案

| 專案 | 領域 | 做了什麼 | 一句話結果 |
|---|---|---|---|
| [**D1**](D1_ab_testing/) | 商業 A/B | Beta–Binomial 封閉解 + 偷看模擬 | P(B>A)=88% vs 頻率派 p=0.24；誠實展示偷看假陽性 5%→26% **貝葉斯也一樣** |
| [**A1**](A1_diagnostic_decision/) | 醫療決策 | PyMC 邏輯迴歸 + 損失矩陣 | 漏診:誤診 = 100:1 時最優門檻是 **0.01**；用 0.5 的期望損失高 86% |
| [**C1**](C1_hierarchical_pooling/) | 公衛階層 | 三模型對照 + 非中心化 | 收縮量 ω 從 0.64→0.14 **緊貼理論曲線**；funnel 從 113 divergences 降到 **0** |
| [**A2**](A2_survival_analysis/) | 生存分析 | 手刻刪失似然 + AFT/Cox 互驗 | 正確處理刪失中位存活 4.62 年 vs 丟掉刪失 2.17 年（**−56%**）|
| [**B1**](B1_transit_fitting/) | 天文參數估計 | emcee + batman 凌日模型 | Rp = **1.50 R⊕** 涵蓋已發表值；週期對上文獻到 **4 ppm** |
| [**B2**](B2_bayes_factor/) | 模型比較 | ultranest nested sampling | 判定假行星 log₁₀B = **+9.39**；ΔBIC **判到反邊**證明近似不可靠 |
| [**C2**](C2_epidemic_sir/) | 疾病傳播 | `pytensor.scan` 手刻 SIR | 政策後 R_t **3.20→2.15**，P(下降)=1.000；2020-04-11 首次 P(R_t<1)>0.95 |
| [**A3**](A3_image_uncertainty_ood/) | 影像不確定性 | Deep Ensemble + MC Dropout | 分解**驗證失敗**：OOD 主要抬升 aleatoric 而非 epistemic |
| [**E1**](E1_bayesian_optimization/) | 實驗設計 | **手刻 GP**（解析梯度）+ 三種 acquisition | 混凝土配比只用 14 次達到隨機搜尋 40 次的水準（**省 65%**）|
| [**S1**](S1_vae_anomaly/) | 變分推論 | ConvVAE + IWAE + 重參數化 | 重參數化 vs REINFORCE 梯度變異數差 **898×** |

六種推論工具與各自的失效邊界（NUTS / emcee / nested sampling / GP / VI / Ensemble）
整理在 [`學習筆記.md` §4](學習筆記.md)。

---

## 快速開始

```bash
# 環境（conda + libmamba，Python 3.12）
conda env create -f environment.yml
conda activate bayes

# torch —— CUDA 12.8 版，對應 Blackwell GPU（標準 cu124 版在此卡會報 no kernel image）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 僅 PyPI 提供的領域套件
pip install medmnist batman-package
pip install scikit-optimize          # 單獨裝，避免相依衝突污染核心環境

python -m ipykernel install --user --name bayes --display-name "Python (bayes)"
python 00_warmup/warmup_pymc.py      # 熱身驗證
```

> **為什麼分步驟裝？** torch 走專用 CUDA 索引、`medmnist` / `batman-package` 只在 PyPI、
> `scikit-optimize` 容易與新版 numpy 相依衝突 —— 分開裝可隔離問題，避免拖垮核心環境的相依求解。

跑任一專案：

```bash
cd A1_diagnostic_decision
python src/run_all.py     # 完整管線（下載資料 → 推論 → 出圖）
python src/replot.py      # 只重畫圖表（秒級，讀落盤資料）
```

**設計原則**：推論結果與出圖資料分離落盤（`*_plotdata.npz` + `figures/results.json`），
調圖表不必重跑動輒 40 分鐘的推論。

開發環境：Linux · Python 3.12 · NVIDIA RTX 5070 Ti（Blackwell / CUDA 12.8）。
GPU 驗證：`python -c "import torch; print(torch.cuda.get_device_name(0))"`

---

## 資料夾結構

```
Bayes/
├── README.md                    ← 本檔
├── 學習筆記.md                   ← 十個專案的方法論總結 + 教材對照
├── 貝葉斯推論_實作訓練計劃.md      ← 完整計劃書（理論 + 每個專案詳規）
├── 貝葉斯推論教材/                ← 七主題 21 張卡片的學習筆記（Obsidian 格式）
├── environment.yml
├── data/                        ← 資料集（不進版控，見 data/README.md 的取得方式與授權）
├── common/                      ← 跨專案共用工具
├── 00_warmup/
└── <每個專案一個資料夾>/
    ├── README.md   notebooks/   src/   figures/   requirements.txt
```

每個專案的 README 走同一個結構：
一句話（問題→方法→結果）· 為什麼需要貝葉斯 · 關鍵結果 · 方法 · **限制與誠實的部分** · 如何重現。

---

## 資料與授權

**本作品集不重新散布任何資料集** —— `data/` 完全不進版控，
所有資料由程式自動下載或套件內建。取得方式與各資料集的授權條款見
[`data/README.md`](data/README.md)。

⚠️ 其中 **JHU CSSE COVID-19 資料僅限教育與學術用途，禁止商業使用**（C2 專案）。

程式碼採 [MIT License](LICENSE)。`貝葉斯推論教材/` 是個人學習筆記。
