# 資料集統一目錄

所有資料集集中在此資料夾，依領域分子夾。

> [!important] 這個資料夾**完全不進版控**
> `.gitignore` 對 `data/` 採**白名單制** —— 只有本檔與各子夾的 `.gitkeep` 會進 git，
> 其餘一切（原始資料、快取、`*_plotdata.npz` 推論結果）一律排除。
>
> 用白名單而非副檔名黑名單是刻意的：黑名單漏過 UCI 的 `.data` 檔，
> 而且每加一種新資料格式都要記得補規則。
>
> **所有資料都能重新取得** —— 執行各專案的 `python src/run_all.py` 會自動下載或生成。

**取得方式圖例**
- 🟢 **程式自動下載**：第一次執行程式碼時自動抓（需網路），無需手動操作
- 🔵 **套件內建 / 程式生成**：完全無需下載
- 🟡 **手動下載**：點連結下載後，放進對應子夾

---

## 資料來源與授權

本專案**不重新散布任何資料集**，只提供取得方式。使用時請遵守各資料集自身的授權條款：

| 資料集 | 用於 | 授權 / 條款 | 需要署名 |
|---|---|:-:|:-:|
| Heart Disease (Cleveland) | A1 | UCI ML Repository · CC BY 4.0 | ✅ |
| GBSG2 | A2 | lifelines 內建（原始資料出自 Schumacher et al. 1994）| ✅ |
| MedMNIST v2 | A3, S1 | CC BY 4.0 | ✅ |
| FashionMNIST | A3 | MIT License | — |
| Kepler 光曲線 | B1, B2 | NASA / MAST · **公共領域** | 建議 |
| KOI 累積表 | B2 | NASA Exoplanet Archive · **公共領域** | 建議 |
| Radon | C1 | pymc 內建（Gelman & Hill）| — |
| **JHU CSSE COVID-19** | C2 | ⚠️ **僅限教育與學術研究用途，禁止商業使用** | ✅ |
| Concrete Compressive Strength | E1 | UCI ML Repository · CC BY 4.0 | ✅ |

⚠️ **JHU CSSE 的條款最嚴格** —— 它明確禁止商業用途。C2 專案是學術性質的重現，
符合該條款；若要把 C2 的程式碼用於商業情境，必須換掉資料來源。

**引用**：各資料集的正式引用格式見其官方頁面（連結在下方各領域表格內）。
MedMNIST 請引用 Yang et al., *Scientific Data* (2023)；
Kepler 資料請依 MAST 的規範致謝 NASA。

---

## A · 醫療 → `data/A_medical/`

| 代號 | 名稱 | 內容 | 取得 |
|---|---|---|---|
| A-D1 | Heart Disease (Cleveland) | 303 筆 · 13 臨床特徵 · 二元診斷 | 🟡 <https://archive.ics.uci.edu/dataset/45/heart+disease> |
| A-D2 | Pima Indians Diabetes | 768 筆 · 8 特徵 | 🟢 `sklearn.datasets.fetch_openml('diabetes', version=1, as_frame=True)` |
| A-D3 | Breast Cancer Wisconsin | 569 筆 · 30 特徵 | 🔵 `sklearn.datasets.load_breast_cancer()` |
| A-D4 | lifelines 生存資料 | Rossi / Stanford heart transplant… | 🔵 `from lifelines.datasets import load_rossi` |
| A-D5 | GBSG · METABRIC | 乳癌生存（DeepSurv 論文標準版）| 🔵 `from lifelines.datasets import load_gbsg2`（**A2 實際採用**，免下載）/ 🟡 <https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data> |
| A-D6 | MedMNIST v2 | 12 個 28×28 醫學影像資料集 | 🟢 `medmnist`（**A3 指定 `root=data/A_medical/medmnist/`**，約 219 MB）|
| A-D8 | FashionMNIST | A3 的「簡單」OOD 對照組（非醫學影像） | 🟢 `torchvision.datasets.FashionMNIST`（自動下載至 `data/A_medical/fashion/`，約 82 MB）|
| A-D7 | NIH ChestX-ray14 | 112,120 張胸腔 X 光 · 14 病徵 | 🟡 <https://nihcc.app.box.com/v/ChestXray-NIHCC>（大檔，選用）|

> A1 用 A-D1 / A-D2；A2 用 A-D4 → A-D5；A3 與 S1 用 A-D6（MedMNIST，強烈建議起手式）。

**A2 實際採用 GBSG2**（German Breast Cancer Study Group 2）：686 位淋巴結陽性乳癌病人、
8 個臨床特徵、追蹤至多 7.3 年，**299 事件 / 387 右刪失（56.4%）**。這個高刪失率正是
A2 的主題（刪失似然），所以選它而非低刪失率的 Rossi。
lifelines 內建、免下載；`A2_survival_analysis/src/data.py` 會另外快取一份
`data/A_medical/gbsg2.csv` 供離線重現與檢查。
A2 的推論結果落盤成 `data/A_medical/a2_plotdata.npz`（已被 `.gitignore` 排除，
執行 `python src/run_all.py` 即自動重建；只調圖表則跑 `python src/replot.py`）。

**A3 用 A-D6 + A-D8**：in-distribution 是 `PneumoniaMNIST`（2 類）與
`PathMNIST`（留出 2 類 → 7 類）；OOD 是 `FashionMNIST`、`DermaMNIST`（彩色）、
`DermaMNIST` 灰階（拔掉顏色捷徑的受控組）、以及 PathMNIST 留出的那 2 類。
全部統一成 3 通道 28×28。推論結果落盤成 `data/A_medical/a3_plotdata.npz`。
⚠️ FashionMNIST 解壓後是無副檔名的 IDX 檔，`.gitignore` 已加
`*-idx[0-9]-ubyte` 與 `data/**/raw/` 兩條規則排除它們（共約 82 MB）。

**S1 用 A-D6，且只用 `PneumoniaMNIST`**，直接重用 A3 已下載的
`data/A_medical/medmnist/`（不會重複下載）。與 A3 的差別在**切法**：
A3 是「兩個資料集之間」的 OOD，S1 是**同一個資料集內**只拿 normal 訓練
（1214 張），把 pneumonia 當異常。同機器、同部位、同解剖結構，
只差病灶 —— 比 A3 難，也更接近臨床部署的情境。
S1 載入時 `as_rgb=False`（單通道），因為 VAE 的重建似然逐像素定義，
複製成 3 通道只會把同一個像素重複算三次、放大重建項而不增加資訊。
推論結果落盤成 `data/A_medical/s1_plotdata.npz`。

## B · 天文 → `data/B_astro/`

| 代號 | 名稱 | 取得 |
|---|---|---|
| B-D1 | Kepler / K2 / TESS 光曲線 | 🟢 `lightkurve.search_lightcurve("Kepler-10", author="Kepler")`（下載至 lightkurve 快取）|
| B-D2 | NASA Exoplanet Archive（已確認參數，拿來對答案）| 🟢 `astroquery` / 🟡 <https://exoplanetarchive.ipac.caltech.edu> |
| B-D3 | Kepler Objects of Interest（含 CONFIRMED / FALSE POSITIVE 標籤）| 🟢 `astroquery` / 🟡 同上 |

> 建議第一個目標 **Kepler-10b**：軌道週期約 20 小時（極短），4 年資料含數千次凌日，訊號清楚。

**B2 用到的第二個目標（FALSE POSITIVE 對照組）**：`KIC 5961350` / **KOI-6017.01**。
從 KOI 累積表以 `koi_disposition='FALSE POSITIVE'` 且 `koi_fpflag_ss=1`（Stellar Eclipse）
篩出，官方註記同時有 `DEEP_V_SHAPED` 與 `HAS_SEC_TCE`——即「V 字形」加「偵測到次食」，
正是食雙星的兩個標誌。查詢語法見 `B2_bayes_factor/README.md`。
B1/B2 的處理結果會快取成 `data/B_astro/*.npz`（已被 `.gitignore` 排除，執行程式即自動重建）。

## C · 公衛 → `data/C_epi/`

| 代號 | 名稱 | 取得 |
|---|---|---|
| C-D1 | Radon（919 棟房 · 85 郡，階層模型教科書範例）| 🟢 `pymc.get_data("radon.csv")` / PyMC 範例庫 |
| C-D2 | Eight Schools（最小階層模型，8 列）| 🔵 `arviz.load_arviz_data("eight_schools")` |
| C-D3 | Our World in Data · COVID（各國每日確診 / 死亡 / 疫苗）| 🟢 <https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv>　⚠️ **已改為週報格式**（見下） |
| C-D6 | **JHU CSSE COVID-19**（累積確診的**真每日**時間序列）| 🟢 <https://github.com/CSSEGISandData/COVID-19>（**C2 實際採用**，1.8 MB）|
| C-D4 | WHO Global Health Observatory | 🟡 <https://www.who.int/data/gho> |
| C-D5 | Efron–Morris 棒球打擊率（收縮估計原始資料，18 列）| 🔵 手動輸入（計劃書 §4.1）|

> C1 用 C-D1（快速看現象可先用 C-D2 / C-D5）；**C2 改用 C-D6（JHU），原因見下**。

⚠️ **C-D3（OWID）已無法用於每日建模。** OWID 把 COVID 資料集回溯性地改成
**週報格式**：整週的總和堆在一天，其餘六天是 0。實測德國 2020-03～06 有
**85% 的天是 0**（例如 2020-03-29 一天記 33,981 例）。這使得
(a) 每日 SIR 觀測模型無法用、(b) **週末效應完全消失**，而 C2 的骨架明確要求處理它。
`new_cases_smoothed` 雖是每日值，但在週內為常數（階梯狀），會人為地讓觀測太平滑。

所以 **C2 改用 C-D6（JHU CSSE）**：累積確診的每日時間序列，差分得每日新增，
保留真正的週期性（德國第一波週一 0.78× vs 週五 1.25×，差 60%）。
換來源不是沒有檢查的決定 —— `src/data.py:cross_validate_sources()` 比對兩者的
**週總和**：總量相對差 0.04%、週序列相關 0.979，確認是同一場疫情、只差時間解析度。
C2 的推論結果落盤成 `data/C_epi/c2_plotdata.npz`（已被 `.gitignore` 排除）。

## D · 商業 → `data/D_business/`

| 代號 | 名稱 | 取得 |
|---|---|---|
| D-D1 | Criteo Uplift（2500 萬筆廣告曝光）| 🟡 <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>（大檔）|
| D-D2 | Online Shoppers Purchasing Intention（12,330 筆）| 🟡 <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset> |
| D-D3 | Bank Marketing（45,211 筆）| 🟡 <https://archive.ics.uci.edu/dataset/222/bank+marketing> |
| D-D4 | 模擬資料（方法論驗證必用——因為你要知道真相）| 🔵 自己生成 |

> D1 的方法論驗證（型一錯誤、偷看資料）**必用 D-D4 模擬**；真實資料（D-D2 / D-D3）用來展示「在髒資料上也能跑」。

## E · 實驗設計 → `data/E_experiment/`

| 代號 | 名稱 | 取得 |
|---|---|---|
| E-D1 | 標準測試函數（Branin / Hartmann6 / Ackley，有已知最優解）| 🔵 手寫 / `scipy` |
| E-D2 | UCI Concrete Compressive Strength（配方 → 強度）| 🟢 `fetch_openml(data_id=4353)`（**E1 實際採用**，快取成 `data/E_experiment/concrete.csv`）/ 🟡 <https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength> |
| E-D3 | Superconductivity（21,263 筆材料 → 臨界溫度）| 🟡 <https://archive.ics.uci.edu/dataset/464/superconductivty+data> |
| E-D4 | HPOBench / LCBench（真實超參數最優化基準，可離線）| 🟡 <https://github.com/automl/HPOBench> |

> E1 用 E-D1 驗證方法 → E-D2 做「材料配方最優化」的真實應用。

**E1 實際使用**：E-D1 全部手寫（Branin 2D、Hartmann6 6D、Ackley 任意維度，
都有已知全域最優可驗證，程式在 `E1_bayesian_optimization/src/objectives.py`）；
E-D2 的 Concrete 從 OpenML 自動下載（1030 筆、8 個配方變數），
用梯度提升迴歸訓練成代理模型當「可查詢的真實世界」，
搜尋範圍取各變數的 [1%, 99%] 分位數以免代理模型在稀疏角落外插。
推論結果落盤成 `data/E_experiment/e1_plotdata.npz`（已被 `.gitignore` 排除）。

---

## 醫療資料的取得門檻（計劃書 §2.1）

MIMIC-IV / MIMIC-CXR / VinDr-CXR **需要 PhysioNet 認證**（要完成 CITI 人類受試者訓練）。
**本計劃刻意避開它們**——上表全部免申請。
