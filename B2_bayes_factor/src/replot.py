"""
B2 · 只重新出圖，不重跑推論。

nested sampling 跑一輪要 ~1.5 小時，但調整圖表是高頻動作。`run_all.py` 會把出圖需要的
一切（分箱資料、後驗中位數參數、J 的後驗樣本）存成 `data/B_astro/b2_plotdata.npz`，
數字部分存成 `figures/results.json`。這支程式從那兩個檔案重建全部六張圖。

執行：python src/replot.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_all

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    R = json.load(open(os.path.join(HERE, "..", "figures", "results.json")))
    targets = run_all.load_plotdata()
    summaries = [dict(R["evidence"][t["key"]]) for t in targets]
    run_all.make_figures(targets, summaries, R["prior_scan"], R, run_all.FIG)


if __name__ == "__main__":
    main()
