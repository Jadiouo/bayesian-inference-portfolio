# common · 跨專案共用工具

放**多個專案會重複用到**的小工具，避免同一段程式碼在 notebook 之間複製貼上
（呼應計劃書 §8.1：「可重用的程式碼，不要全部塞在 notebook」）。

## 抽象原則

> 某個工具「第二次」被不同專案需要時，才抽到這裡。
> 只被單一專案用到的東西，留在該專案自己的 `src/`。

先不預先實作，等第一個消費者（多半是 D1 / A1）出現再填，避免抽錯介面。

## 規劃中的模組

| 模組 | 內容 | 主要使用者 |
|---|---|---|
| `calibration.py` | reliability diagram、ECE | A1、A3、D1 |
| `decision.py` | 損失矩陣、期望損失、最優門檻、棄權區 | A1、D1 |
| `plotting.py` | HDI / 後驗圖、risk-coverage 曲線 | A1、A3、C1 |
| `mcmc_diag.py` | r_hat / ESS / divergences 收斂檢查 | 多數 PyMC 專案 |

## 在專案中引用

```python
import sys; sys.path.append("..")     # 專案資料夾在 Bayes/ 底下一層
from common.decision import optimal_threshold
```
