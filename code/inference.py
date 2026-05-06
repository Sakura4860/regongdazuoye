"""
故障诊断推理脚本
================

加载已训练的 XGBoost 模型，对一条/多条观测数据做分类。

用法:
    python code/inference.py --input data/dataset.csv --n 5
    python code/inference.py --input my_obs.csv

输出: 每条样本的预测类别 + 置信度。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from code.train import CLASS_NAMES, FEATURE_NAMES, MERGE_MAP


def load_model(model_path: Path) -> XGBClassifier:
    m = XGBClassifier()
    m.load_model(model_path)
    return m


def predict(model: XGBClassifier, X: np.ndarray) -> list[dict]:
    """返回每条样本的预测字典：{class, class_name, confidence, top2}。"""
    probs = model.predict_proba(X)
    preds = probs.argmax(axis=1)

    out = []
    for i, p in enumerate(preds):
        top2_idx = np.argsort(probs[i])[::-1][:2]
        out.append({
            "index": int(i),
            "class": int(p),
            "class_name": CLASS_NAMES[int(p)],
            "confidence": float(probs[i, p]),
            "top2": [
                {"class": int(k), "class_name": CLASS_NAMES[int(k)],
                 "prob": float(probs[i, int(k)])}
                for k in top2_idx
            ],
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default=str(ROOT / "models" / "xgb_fdd.json"))
    parser.add_argument("--input", type=str,
                        default=str(ROOT / "data" / "dataset.csv"),
                        help="观测 CSV，至少包含 20 个特征列")
    parser.add_argument("--n", type=int, default=5,
                        help="演示时只预测前 n 条；=-1 全量")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    model = load_model(Path(args.model))
    df = pd.read_csv(args.input)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    if args.n > 0:
        X = X[:args.n]

    preds = predict(model, X)

    for p in preds:
        print(f"样本 {p['index']:>4d}: "
              f"预测={p['class_name']:<12s} "
              f"置信度={p['confidence']:.3f} "
              f"次优={p['top2'][1]['class_name']} "
              f"({p['top2'][1]['prob']:.3f})")

    # 如果原始 CSV 含 class 标签，顺便打印一致性
    if "class" in df.columns:
        y_true = np.array([MERGE_MAP[int(c)] for c in df["class"].iloc[:len(preds)]])
        y_pred = np.array([p["class"] for p in preds])
        acc = (y_true == y_pred).mean()
        print(f"\n与输入 class 标签一致率: {acc * 100:.1f}% ({int((y_true==y_pred).sum())}/{len(preds)})")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[OK] 预测结果已保存: {out_path}")


if __name__ == "__main__":
    main()
