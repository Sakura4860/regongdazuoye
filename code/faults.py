"""
故障数据生成器
==============

6 类故障 + 正常 = 7 个类别：
0: 正常
1: 支路阻力增大（堵塞/阀门过关）
2: 支路阻力减小（阀门过开/设备短路）
3: 双支路同时失调
4: 泵频偏移
5: 管网总负荷波动（温度/总流量大幅变化）——非失调但扰动
6: 传感器漂移（在正常基础上叠加偏移，用于鲁棒性）

对每条样本输出 20 维特征 + (类别, 失调支路) 标签。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

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

from code.networks.station_5br import build_station_5br
from code.solver import solve

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(42)

FEATURE_NAMES = [
    "Q0", "dP_header", "dP_main_supply", "H_pump",
    "Q1", "Q2", "Q3", "Q4", "Q5",
    "dP1", "dP2", "dP3", "dP4", "dP5",
    "dT1", "dT2", "dT3", "dT4", "dT5",
    "Q_sum_rel",  # 支路流量和 / 主管流量，平衡性指标
]

CLASS_NAMES = [
    "正常",
    "支路阻力↑(堵塞/阀关)",
    "支路阻力↓(阀过开)",
    "双支路失调",
    "泵频偏移",
    "负荷工况变化",
    "传感器漂移",
]


def extract_features(sol, noise_scale=0.0, dT_bias=None) -> np.ndarray:
    """从稳态解抽取 20 维特征。可选加测量噪声。"""
    q = [sol.Q_branches[i] for i in range(1, 6)]
    dp = [sol.dP_branches[i] for i in range(1, 6)]
    # 温差 dT_i = T_supply - T_return_i，基准 15℃ 加随机用户负荷扰动
    if dT_bias is None:
        dT_bias = np.zeros(5)
    dT = [sol.T_supply_branches[i] - sol.T_return_branches[i] + dT_bias[i - 1]
          for i in range(1, 6)]

    feat = np.array([
        sol.Q_total,
        sol.dP_header,
        sol.dP_main_supply,
        sol.H_pump,
        q[0], q[1], q[2], q[3], q[4],
        dp[0], dp[1], dp[2], dp[3], dp[4],
        dT[0], dT[1], dT[2], dT[3], dT[4],
        sum(q) / max(sol.Q_total, 1e-6),
    ])

    # 按传感器精度加噪声
    if noise_scale > 0:
        # 不同测点误差等级不同
        sigmas_rel = np.array([
            0.005,  # Q0: 0.5% 电磁
            0.001,  # dP_h: 0.1%
            0.001,  # dP_main
            0.005,  # H_pump: 由差压推算
            0.02, 0.02, 0.02, 0.02, 0.02,  # Q_i: 2% 超声
            0.002, 0.002, 0.002, 0.002, 0.002,  # dP_i: 0.2%
            0.01, 0.01, 0.01, 0.01, 0.01,  # dT_i: ~0.15℃ / 15℃ ≈ 1%
            0.01,
        ])
        feat = feat * (1.0 + RNG.normal(0, sigmas_rel * noise_scale))
    return feat


def gen_normal(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 n 个正常工况样本（变总流量 + 温度扰动）。"""
    X, y_class, y_branch = [], [], []
    for _ in range(n):
        net = build_station_5br()
        # 总流量扰动：50 ±30% 覆盖题目范围 30-80
        net.pump_speed = float(RNG.uniform(0.85, 1.10))
        # 温度扰动
        net.T_supply = float(RNG.uniform(58, 70))
        # 用户负荷扰动：各支路 ΔT 按设计 15℃ ±2℃
        dT_bias = RNG.normal(0, 1.0, size=5)

        sol = solve(net)
        feat = extract_features(sol, noise_scale=1.0, dT_bias=dT_bias)
        X.append(feat)
        y_class.append(0)
        y_branch.append(0)
    return np.array(X), np.array(y_class), np.array(y_branch)


def gen_branch_resistance(n: int, direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单支路阻力偏移。direction ∈ {'up', 'down'}。"""
    X, y_class, y_branch = [], [], []
    cls = 1 if direction == "up" else 2
    for _ in range(n):
        net = build_station_5br()
        net.pump_speed = float(RNG.uniform(0.85, 1.10))
        net.T_supply = float(RNG.uniform(58, 70))
        dT_bias = RNG.normal(0, 1.0, size=5)

        bid = int(RNG.integers(1, 6))
        if direction == "up":
            factor = float(RNG.uniform(1.5, 5.0))
        else:
            factor = float(RNG.uniform(0.2, 0.7))
        net.branch_resistance[bid] *= factor

        sol = solve(net)
        feat = extract_features(sol, noise_scale=1.0, dT_bias=dT_bias)
        X.append(feat)
        y_class.append(cls)
        y_branch.append(bid)
    return np.array(X), np.array(y_class), np.array(y_branch)


def gen_double(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """双支路同时失调。扰动强度避开温和区，保证可辨识。"""
    X, y_class, y_branch = [], [], []
    for _ in range(n):
        net = build_station_5br()
        net.pump_speed = float(RNG.uniform(0.85, 1.10))
        net.T_supply = float(RNG.uniform(58, 70))
        dT_bias = RNG.normal(0, 1.0, size=5)

        b1, b2 = RNG.choice(np.arange(1, 6), size=2, replace=False)
        for b in (b1, b2):
            # 双峰采样：显著阻力↑ 或 显著阻力↓，跳过 [0.7, 1.5] 的温和区
            if RNG.random() < 0.5:
                f = float(RNG.uniform(1.8, 4.5))
            else:
                f = float(RNG.uniform(0.25, 0.55))
            net.branch_resistance[int(b)] *= f

        sol = solve(net)
        feat = extract_features(sol, noise_scale=1.0, dT_bias=dT_bias)
        X.append(feat)
        y_class.append(3)
        y_branch.append(min(int(b1), int(b2)) * 10 + max(int(b1), int(b2)))
    return np.array(X), np.array(y_class), np.array(y_branch)


def gen_pump(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """泵频偏移。"""
    X, y_class, y_branch = [], [], []
    for _ in range(n):
        net = build_station_5br()
        net.T_supply = float(RNG.uniform(58, 70))
        dT_bias = RNG.normal(0, 1.0, size=5)
        # 泵频显著偏离正常范围
        net.pump_speed = float(RNG.choice([
            RNG.uniform(0.55, 0.80),
            RNG.uniform(1.10, 1.25),
        ]))
        sol = solve(net)
        feat = extract_features(sol, noise_scale=1.0, dT_bias=dT_bias)
        X.append(feat)
        y_class.append(4)
        y_branch.append(0)
    return np.array(X), np.array(y_class), np.array(y_branch)


def gen_load_swing(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """负荷大幅波动（但水力正常）。"""
    X, y_class, y_branch = [], [], []
    for _ in range(n):
        net = build_station_5br()
        net.pump_speed = float(RNG.uniform(0.6, 1.2))
        net.T_supply = float(RNG.uniform(55, 70))
        dT_bias = RNG.normal(0, 3.0, size=5)  # 更大温差波动
        sol = solve(net)
        feat = extract_features(sol, noise_scale=1.0, dT_bias=dT_bias)
        X.append(feat)
        y_class.append(5)
        y_branch.append(0)
    return np.array(X), np.array(y_class), np.array(y_branch)


def gen_sensor_drift(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在正常运行上叠加较大传感器漂移。"""
    X, y_class, y_branch = [], [], []
    for _ in range(n):
        net = build_station_5br()
        net.pump_speed = float(RNG.uniform(0.85, 1.10))
        net.T_supply = float(RNG.uniform(58, 70))
        dT_bias = RNG.normal(0, 1.0, size=5)
        sol = solve(net)
        # 噪声放大 3 倍
        feat = extract_features(sol, noise_scale=3.0, dT_bias=dT_bias)
        X.append(feat)
        y_class.append(6)
        y_branch.append(0)
    return np.array(X), np.array(y_class), np.array(y_branch)


def build_dataset() -> pd.DataFrame:
    print("生成正常 1500 条...", end=" ", flush=True)
    X0, y0, b0 = gen_normal(1500); print("OK")
    print("生成阻力上升 1000 条...", end=" ", flush=True)
    X1, y1, b1 = gen_branch_resistance(1000, "up"); print("OK")
    print("生成阻力下降 1000 条...", end=" ", flush=True)
    X2, y2, b2 = gen_branch_resistance(1000, "down"); print("OK")
    print("生成双支路失调 1500 条...", end=" ", flush=True)
    X3, y3, b3 = gen_double(1500); print("OK")
    print("生成泵频偏移 600 条...", end=" ", flush=True)
    X4, y4, b4 = gen_pump(600); print("OK")
    print("生成负荷工况波动 200 条...", end=" ", flush=True)
    X5, y5, b5 = gen_load_swing(200); print("OK")
    print("生成传感器漂移 100 条...", end=" ", flush=True)
    X6, y6, b6 = gen_sensor_drift(100); print("OK")

    X = np.concatenate([X0, X1, X2, X3, X4, X5, X6], axis=0)
    y = np.concatenate([y0, y1, y2, y3, y4, y5, y6], axis=0)
    b = np.concatenate([b0, b1, b2, b3, b4, b5, b6], axis=0)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["class"] = y
    df["branch"] = b
    return df


def plot_eda(df: pd.DataFrame, save_dir: Path) -> None:
    # 1) 类别分布
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df["class"].value_counts().sort_index()
    ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color="#2ecc71")
    ax.set_ylabel("样本数")
    ax.set_title("故障类别分布")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_dir / "eda_class_dist.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2) 特征相关热图
    num = df[FEATURE_NAMES]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    for i in range(len(FEATURE_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            v = corr.iloc[i, j]
            if abs(v) >= 0.5:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.8 else "black")
    fig.colorbar(im, ax=ax)
    ax.set_title("特征相关系数矩阵")
    fig.tight_layout()
    fig.savefig(save_dir / "eda_correlation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_dir = ROOT / "data"
    fig_dir = ROOT / "fig"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    df = build_dataset()
    out = data_dir / "dataset.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ 数据集已保存: {out} ({len(df)} 样本)")
    print(f"\n类别分布:\n{df['class'].value_counts().sort_index().to_string()}")

    plot_eda(df, fig_dir)
    print("\n✓ EDA 图已保存: fig/eda_*.png")


if __name__ == "__main__":
    main()
