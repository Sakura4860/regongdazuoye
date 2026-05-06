"""
灵敏度分析：对每条支路 ±20% 阻力扰动，观察所有候选测点的响应。
用来证明"测点布置不是均布，而是有可观测性依据的"。

输出：
- sensitivity_matrix.npy  (m×n)
- fig/sensitivity_heatmap.png
- 基于贪心 SVD 选择最小测点集
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code.networks.station_5br import build_station_5br
from code.solver import solve

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# 候选测点清单（14 维，对应 §3.1）
def extract_observation(sol) -> np.ndarray:
    """按固定顺序抽取所有候选测点。"""
    qs = [sol.Q_branches[i] for i in range(1, 6)]
    dps = [sol.dP_branches[i] for i in range(1, 6)]
    obs = np.array([
        sol.Q_total,
        sol.dP_header,
        sol.dP_main_supply,
        qs[0], qs[1], qs[2], qs[3], qs[4],
        dps[0], dps[1], dps[2], dps[3], dps[4],
        sol.H_pump,
    ])
    return obs


OBS_NAMES = [
    "Q0 (主管流量)",
    "ΔP_h (干管压差)",
    "ΔP_main (主管压降)",
    "Q1 (支路1)", "Q2", "Q3", "Q4", "Q5",
    "ΔP1 (支路1压降)", "ΔP2", "ΔP3", "ΔP4", "ΔP5",
    "H_pump (泵扬程)",
]

FAULT_NAMES = [f"支路{i}阻力+20%" for i in range(1, 6)] + \
              [f"支路{i}阻力-20%" for i in range(1, 6)] + \
              ["泵频率-10%"]


def run_sensitivity() -> tuple[np.ndarray, np.ndarray]:
    """返回 (obs_baseline, sensitivity_matrix)。

    Sensitivity[m, f] = (y_m(fault_f) - y_m(baseline)) / y_m(baseline)
    """
    net_base = build_station_5br()
    sol_base = solve(net_base)
    y0 = extract_observation(sol_base)

    results = []

    # 单支路阻力 +20%
    for bid in range(1, 6):
        net = build_station_5br()
        net.branch_resistance[bid] *= 1.2
        sol = solve(net)
        results.append(extract_observation(sol))

    # 单支路阻力 -20%
    for bid in range(1, 6):
        net = build_station_5br()
        net.branch_resistance[bid] *= 0.8
        sol = solve(net)
        results.append(extract_observation(sol))

    # 泵频 -10%
    net = build_station_5br()
    net.pump_speed = 0.9
    sol = solve(net)
    results.append(extract_observation(sol))

    Y = np.array(results)  # (n_fault, n_meas)
    S = (Y - y0[None, :]) / y0[None, :]  # 相对灵敏度
    return y0, S.T  # 转置：行=测点，列=故障


def greedy_select(S: np.ndarray, n_select: int = 7) -> list[int]:
    """贪心选择使 det(S_selected @ S_selected.T) 最大的 n_select 行。

    等价于选择让所选子矩阵奇异值最大的测点组合。
    """
    n_meas = S.shape[0]
    selected = []
    remaining = list(range(n_meas))

    for _ in range(n_select):
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            trial = selected + [idx]
            sub = S[trial, :]
            # 用 log-det 避免溢出；退化时用最小奇异值
            try:
                u, sv, vh = np.linalg.svd(sub, full_matrices=False)
                score = np.sum(np.log(sv[sv > 1e-12] + 1e-12))
            except np.linalg.LinAlgError:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def plot_heatmap(S: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(S * 100, cmap="RdBu_r", vmin=-30, vmax=30, aspect="auto")
    ax.set_xticks(range(len(FAULT_NAMES)))
    ax.set_xticklabels(FAULT_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(OBS_NAMES)))
    ax.set_yticklabels(OBS_NAMES, fontsize=9)
    ax.set_title("测点灵敏度矩阵 (单位: %相对变化)", fontsize=12)
    # 数值标注
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            val = S[i, j] * 100
            if abs(val) >= 1.5:
                ax.text(j, i, f"{val:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 10 else "black")
    fig.colorbar(im, ax=ax, label="相对灵敏度 (%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_selection(selected: list[int], S: np.ndarray, save_path: Path) -> None:
    """画出选中与未选中测点的"信息量"对比。"""
    scores = np.linalg.norm(S, axis=1)  # 每个测点的总灵敏度
    colors = ["#e74c3c" if i in selected else "#95a5a6" for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(range(len(scores)), scores, color=colors, edgecolor="black")
    ax.set_xticks(range(len(OBS_NAMES)))
    ax.set_xticklabels(OBS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("总灵敏度 ||S_m|| (无量纲)", fontsize=10)
    ax.set_title(f"测点信息量排序（红色 = 贪心选中的 {len(selected)} 个最小测点集）",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_dir = ROOT / "fig"
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    y0, S = run_sensitivity()
    print(f"灵敏度矩阵维度: {S.shape}")
    print(f"矩阵秩: {np.linalg.matrix_rank(S, tol=1e-3)}")
    print(f"条件数: {np.linalg.cond(S):.2f}")

    # 保存
    np.save(data_dir / "sensitivity_matrix.npy", S)
    np.savetxt(data_dir / "baseline_obs.csv", y0, fmt="%.4f",
               header=",".join(OBS_NAMES), comments="")

    selected = greedy_select(S, n_select=7)
    print("\n=== 贪心选出的最小测点集 ===")
    for i, idx in enumerate(selected, 1):
        print(f"  {i}. {OBS_NAMES[idx]}")
    sub = S[selected, :]
    print(f"\n子集条件数: {np.linalg.cond(sub):.2f}")
    print(f"子集秩: {np.linalg.matrix_rank(sub, tol=1e-3)}/{len(FAULT_NAMES)}")

    plot_heatmap(S, fig_dir / "sensitivity_heatmap.png")
    plot_selection(selected, S, fig_dir / "sensor_selection.png")
    print("\n✓ 图已保存: fig/sensitivity_heatmap.png, fig/sensor_selection.png")


if __name__ == "__main__":
    main()
