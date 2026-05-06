"""
故障分类模型训练
================

输入：data/dataset.csv （20 特征 + class + branch）
流程：
1. 80/20 分层划分
2. RandomForest baseline (n=200, depth=12)
3. XGBoost 主力 (max_depth=6, lr=0.1, n=300)
4. 5-fold CV 评估（F1 macro）
5. 输出混淆矩阵 + classification report
6. 导出 XGBoost 到 models/xgb_fdd.json
7. SHAP summary plot（可选：--skip-shap 跳过）

用法：
    python code/train.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK，强制 stdout 用 UTF-8，避免中文/Unicode 符号报错
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


FEATURE_NAMES = [
    "Q0", "dP_header", "dP_main_supply", "H_pump",
    "Q1", "Q2", "Q3", "Q4", "Q5",
    "dP1", "dP2", "dP3", "dP4", "dP5",
    "dT1", "dT2", "dT3", "dT4", "dT5",
    "Q_sum_rel",
]

CLASS_NAMES = [
    "正常",
    "阻力↑",
    "阻力↓",
    "双支路",
    "泵频偏移",
]

# 类别合并映射：faults.py 生成 7 类，但 5=负荷波动 / 6=传感器漂移 本质上
# 是"正常运行 + 大扰动"（水力工况正常），应作为正常类的鲁棒性样本。
# 这一点与 solution-plan §5.2 "测量噪声 300 条用于鲁棒性"的原始定位一致。
MERGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 0}

SEED = 42


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """载入特征矩阵与类别标签。按 MERGE_MAP 将鲁棒性类合并进正常。"""
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_raw = df["class"].to_numpy(dtype=np.int64)
    y = np.array([MERGE_MAP[c] for c in y_raw], dtype=np.int64)
    return X, y


def cross_val_f1(model, X, y, n_splits: int = 5) -> tuple[float, float]:
    """5-fold CV，返回 F1 macro 的均值与标准差。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    f1_scores = []
    for fold_idx, (tr, va) in enumerate(skf.split(X, y), 1):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        y_hat = m.predict(X[va])
        f1 = f1_score(y[va], y_hat, average="macro")
        f1_scores.append(f1)
        print(f"  fold {fold_idx}: F1 macro = {f1:.4f}")
    return float(np.mean(f1_scores)), float(np.std(f1_scores))


def plot_confusion(y_true, y_pred, title: str, save_path: Path) -> None:
    """绘制混淆矩阵（按行归一化百分比）。"""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title, fontsize=12)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            cnt = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if pct > 55 else "black"
            ax.text(j, i, f"{cnt}\n{pct:.1f}%",
                    ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="行归一化百分比 (%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ 混淆矩阵: {save_path}")


def shap_summary(xgb_model: XGBClassifier, X_test: np.ndarray,
                 save_path: Path, sample: int = 500) -> None:
    """用 TreeExplainer 生成 SHAP summary plot（跨类别平均）。"""
    import shap
    if X_test.shape[0] > sample:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X_test.shape[0], size=sample, replace=False)
        Xs = X_test[idx]
    else:
        Xs = X_test

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(Xs)

    # 新版 shap (≥0.43) 多分类输出 (n_samples, n_features, n_class)
    # 旧版可能返回 list[n_class] 或 (n_class, n_samples, n_features)
    if isinstance(shap_vals, list):
        arr = np.stack([np.abs(v) for v in shap_vals], axis=0)  # (C, N, F)
        shap_abs = arr.mean(axis=0)                              # (N, F)
    elif shap_vals.ndim == 3:
        if shap_vals.shape[-1] == len(CLASS_NAMES):             # (N, F, C)
            shap_abs = np.mean(np.abs(shap_vals), axis=-1)
        else:                                                    # (C, N, F)
            shap_abs = np.mean(np.abs(shap_vals), axis=0)
    else:
        shap_abs = np.abs(shap_vals)

    assert shap_abs.shape == Xs.shape, \
        f"SHAP shape {shap_abs.shape} != X shape {Xs.shape}"

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_abs, Xs,
        feature_names=FEATURE_NAMES,
        plot_type="bar",
        show=False,
        color="#e67e22",
    )
    plt.title("SHAP 特征重要性 (跨 5 类平均 |SHAP|)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] SHAP summary: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-cv", action="store_true",
                        help="跳过 5-fold CV（仅做一次 80/20）")
    args = parser.parse_args()

    data_path = ROOT / "data" / "dataset.csv"
    fig_dir = ROOT / "fig"
    model_dir = ROOT / "models"
    results_dir = ROOT / "results"
    fig_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print(f"载入数据: {data_path}")
    X, y = load_dataset(data_path)
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")

    # 分层 80/20
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # ----------------------------------------------------------
    # Baseline: RandomForest
    # ----------------------------------------------------------
    print("\n[1/2] RandomForest baseline (n_estimators=200, max_depth=12)")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        n_jobs=-1, random_state=SEED,
        class_weight="balanced",
    )
    if not args.skip_cv:
        print("  5-fold CV:")
        rf_cv_mean, rf_cv_std = cross_val_f1(rf, X_tr, y_tr)
        print(f"  → 平均 F1 macro = {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")

    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    f1_rf = f1_score(y_te, y_pred_rf, average="macro")
    print(f"  测试集 F1 macro = {f1_rf:.4f}")
    print(classification_report(y_te, y_pred_rf, target_names=CLASS_NAMES, digits=3))
    plot_confusion(y_te, y_pred_rf,
                   f"RandomForest 混淆矩阵 (F1={f1_rf:.3f})",
                   fig_dir / "confusion_matrix_rf.png")

    # ----------------------------------------------------------
    # Main: XGBoost
    # ----------------------------------------------------------
    print("\n[2/2] XGBoost (max_depth=6, lr=0.1, n_estimators=300)")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    if not args.skip_cv:
        print("  5-fold CV:")
        xgb_cv_mean, xgb_cv_std = cross_val_f1(xgb, X_tr, y_tr)
        print(f"  → 平均 F1 macro = {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")

    xgb.fit(X_tr, y_tr)
    y_pred_xgb = xgb.predict(X_te)
    f1_xgb = f1_score(y_te, y_pred_xgb, average="macro")
    print(f"  测试集 F1 macro = {f1_xgb:.4f}")
    report = classification_report(y_te, y_pred_xgb, target_names=CLASS_NAMES,
                                   digits=3, output_dict=True)
    print(classification_report(y_te, y_pred_xgb, target_names=CLASS_NAMES, digits=3))
    plot_confusion(y_te, y_pred_xgb,
                   f"XGBoost 混淆矩阵 (F1={f1_xgb:.3f})",
                   fig_dir / "confusion_matrix.png")

    # 保存模型与报告
    xgb_path = model_dir / "xgb_fdd.json"
    xgb.save_model(xgb_path)
    print(f"✓ XGBoost 模型已保存: {xgb_path}")

    summary = {
        "seed": SEED,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "rf_test_f1_macro": float(f1_rf),
        "xgb_test_f1_macro": float(f1_xgb),
        "xgb_classification_report": report,
    }
    if not args.skip_cv:
        summary["rf_cv_f1_macro_mean"] = rf_cv_mean
        summary["rf_cv_f1_macro_std"] = rf_cv_std
        summary["xgb_cv_f1_macro_mean"] = xgb_cv_mean
        summary["xgb_cv_f1_macro_std"] = xgb_cv_std
    (results_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✓ 训练摘要: {results_dir / 'train_summary.json'}")

    # ----------------------------------------------------------
    # SHAP 解释
    # ----------------------------------------------------------
    if not args.skip_shap:
        print("\n[SHAP] 生成特征重要性摘要...")
        shap_summary(xgb, X_te, fig_dir / "shap_summary.png")


if __name__ == "__main__":
    main()
