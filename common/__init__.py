"""
common —— 跨專案共用工具。

原則（避免過早抽象）：某個工具「第二次」被不同專案需要時，才抽到這裡；
只被單一專案用到的東西，留在該專案的 src/。

規劃中的內容（多個專案會重複用到）：
  - calibration.py  reliability diagram、ECE            （A1、A3、D1）
  - decision.py     損失矩陣、期望損失、最優門檻、棄權區   （A1、D1）
  - plotting.py     HDI / 後驗圖、risk-coverage 曲線       （A1、A3、C1）
  - mcmc_diag.py    r_hat / ESS / divergences 收斂檢查      （多數 PyMC 專案）

使用方式（在專案 notebook / src 內）：
  import sys; sys.path.append("..")   # 讓 common 可被 import
  from common.decision import optimal_threshold
"""

__all__: list[str] = []
