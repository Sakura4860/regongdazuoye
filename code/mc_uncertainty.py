"""
Monte Carlo 误差传递（JCGM 101:2008）
======================================

按 solution-plan §6.1 的不确定度来源表，对 4 个典型工况抽 N=10000 个样本，
得到 HII 的经验分布与 95% 置信区间；按 §6.2 规则判定工况是否跨越阈值。

运行：
    python code/mc_uncertainty.py

输出：
    results/mc_summary.csv
    fig/mc_hist.png
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code.hii import HII_LEVELS, classify, compute_hii
from code.networks.station_5br import (
    DESIGN_FLOW_PER_BRANCH,
    apply_valve_setting,
    build_station_5br,
    inject_branch_blockage,
)
from code.solver import solve

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------
# 不确定度参数（与 solution-plan §6.1 表格一致）
# ------------------------------------------------------------
@dataclass(frozen=True)
class UncertaintyConfig:
    """所有传感器的不确定度半宽（a），按矩形/三角分布给定。"""
    T_rect_half: float = 0.15         # ℃，Pt100 A 级
    P_rel_rect: float = 0.001         # ±0.1% FS，Rosemount 2088
    dP_rel_rect: float = 0.00075      # ±0.075% FS
    Q0_rel_rect: float = 0.005        # ±0.5% 电磁
    Qi_rel_rect: float = 0.02         # ±2% 一体式超声
    install_rel_tri: float = 0.01     # 三角分布 ±1% 安装偏差


UNC = UncertaintyConfig()
N_MC = 10000
SEED = 2026


# ------------------------------------------------------------
# 噪声注入
# ------------------------------------------------------------
def sample_noisy_obs(
    Q_branches: Dict[int, float],
    dP_branches: Dict[int, float],
    T_sup_branches: Dict[int, float],
    T_ret_branches: Dict[int, float],
    rng: np.random.Generator,
) -> tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """生成一次含噪声的观测：返回 (Q_noisy, dP_noisy, dT_noisy)。

    - Q_i: 矩形相对 ±2% + 三角相对 ±1% 安装偏差
    - ΔP_i: 矩形相对 ±0.075%（相对于满量程 200 kPa ≈ 0.15 kPa 半宽），
           近似转成相对误差
    - T_sup, T_ret: 独立矩形 ±0.15 ℃ → ΔT 误差 ~0.21 ℃ 半宽
    """
    Q_noisy = {}
    for bid, Q in Q_branches.items():
        rect = rng.uniform(-UNC.Qi_rel_rect, UNC.Qi_rel_rect)
        tri = rng.triangular(-UNC.install_rel_tri, 0.0, UNC.install_rel_tri)
        Q_noisy[bid] = Q * (1 + rect + tri)

    dP_noisy = {}
    for bid, dp in dP_branches.items():
        # 绝对 0.15 kPa 半宽近似（FS 200kPa × 0.075%）
        dP_noisy[bid] = dp + rng.uniform(-0.15, 0.15)

    dT_noisy = {}
    for bid in T_sup_branches:
        T_s = T_sup_branches[bid] + rng.uniform(-UNC.T_rect_half, UNC.T_rect_half)
        T_r = T_ret_branches[bid] + rng.uniform(-UNC.T_rect_half, UNC.T_rect_half)
        dT_noisy[bid] = T_s - T_r
    return Q_noisy, dP_noisy, dT_noisy


def mc_propagate(
    Q_branches: Dict[int, float],
    dP_branches: Dict[int, float],
    T_sup_branches: Dict[int, float],
    T_ret_branches: Dict[int, float],
    Q_design: Dict[int, float],
    n_mc: int = N_MC,
    seed: int = SEED,
) -> np.ndarray:
    """返回 HII 的 MC 样本 (n_mc,)。"""
    rng = np.random.default_rng(seed)
    samples = np.empty(n_mc, dtype=np.float64)
    for k in range(n_mc):
        Qn, dPn, dTn = sample_noisy_obs(
            Q_branches, dP_branches, T_sup_branches, T_ret_branches, rng
        )
        samples[k] = compute_hii(Qn, Q_design, dPn, dTn).hii
    return samples


# ------------------------------------------------------------
# 判定规则（§6.2）
# ------------------------------------------------------------
def judge_with_ci(hii_lo: float, hii_hi: float) -> str:
    """基于 95% 置信区间的判定。若区间跨越任一阈值 → 延迟判定。"""
    thresholds = [lv[0] for lv in HII_LEVELS[:-1]]  # [0.10, 0.20, 0.35]
    # 是否有任一阈值落在 CI 内 → 跨越
    for thr in thresholds:
        if hii_lo < thr < hii_hi:
            return "跨越阈值(延迟判定)"
    # CI 完整落在某一等级区间内
    return "可靠" + classify((hii_lo + hii_hi) / 2.0)[0]


# ------------------------------------------------------------
# 场景：用不同扰动构造 4 个 HII 目标等级
# ------------------------------------------------------------
def scenario_normal():
    return build_station_5br()


def scenario_blockage_mild():
    net = build_station_5br()
    inject_branch_blockage(net, 4, 2.5)    # 轻度堵塞，HII ≈ 0.12（轻度失调阈值附近）
    return net


def scenario_valve_mid():
    net = build_station_5br()
    apply_valve_setting(net, {1: 1.0, 2: 0.55, 3: 1.0, 4: 1.0, 5: 1.0})
    return net


def scenario_valve_severe():
    net = build_station_5br()
    apply_valve_setting(net, {1: 1.0, 2: 0.30, 3: 1.0, 4: 1.0, 5: 1.0})
    return net


def scenario_threshold():
    net = build_station_5br()
    inject_branch_blockage(net, 4, 2.15)   # HII 刚好在 0.10 附近，CI 预期跨阈值
    return net


SCENARIOS: Dict[str, Callable] = {
    "正常工况": scenario_normal,
    "支路4堵塞(×2.15,阈值边缘)": scenario_threshold,
    "支路4堵塞(×2.5)": scenario_blockage_mild,
    "支路2阀门55%": scenario_valve_mid,
    "支路2阀门30%": scenario_valve_severe,
}


# ------------------------------------------------------------
# 运行
# ------------------------------------------------------------
def run_all() -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    Q_design = {i: DESIGN_FLOW_PER_BRANCH for i in range(1, 6)}

    rows = []
    mc_all: Dict[str, np.ndarray] = {}
    for name, build in SCENARIOS.items():
        net = build()
        sol = solve(net)
        # 真实 HII（未加噪声）
        hii_true = compute_hii(
            sol.Q_branches, Q_design, sol.dP_branches,
            {b: sol.T_supply_branches[b] - sol.T_return_branches[b]
             for b in sol.Q_branches},
        ).hii

        # MC 采样
        samples = mc_propagate(
            sol.Q_branches, sol.dP_branches,
            sol.T_supply_branches, sol.T_return_branches,
            Q_design, n_mc=N_MC, seed=SEED,
        )
        hii_lo = float(np.percentile(samples, 2.5))
        hii_hi = float(np.percentile(samples, 97.5))
        hii_mean = float(samples.mean())
        judged = judge_with_ci(hii_lo, hii_hi)
        point_level, _ = classify(hii_true)

        rows.append({
            "scenario": name,
            "HII_true": round(hii_true, 4),
            "HII_mean": round(hii_mean, 4),
            "HII_95_lo": round(hii_lo, 4),
            "HII_95_hi": round(hii_hi, 4),
            "CI_width": round(hii_hi - hii_lo, 4),
            "point_level": point_level,
            "CI_judgment": judged,
        })
        mc_all[name] = samples
        print(f"{name:<22s}  HII_true={hii_true:.3f}  "
              f"CI=[{hii_lo:.3f}, {hii_hi:.3f}]  ← {judged}")

    return pd.DataFrame(rows), mc_all


def plot_histograms(mc_all: Dict[str, np.ndarray], save_path: Path) -> None:
    n = len(mc_all)
    ncol = 3 if n > 4 else 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.2, nrow * 3.6))
    axes = np.atleast_1d(axes).flatten()
    for i, (name, samples) in enumerate(mc_all.items()):
        ax = axes[i]
        ax.hist(samples, bins=60, color="#3498db", edgecolor="white", alpha=0.85)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        for thr, _, _ in HII_LEVELS[:-1]:
            ax.axvline(thr, color="gray", linestyle="--", alpha=0.6, linewidth=1)
            ax.text(thr, ax.get_ylim()[1] * 0.92, f"{thr:.2f}",
                    fontsize=7, color="gray", ha="center")
        ax.axvline(lo, color="#e74c3c", linewidth=1.5, label=f"2.5% = {lo:.3f}")
        ax.axvline(hi, color="#e74c3c", linewidth=1.5, label=f"97.5% = {hi:.3f}")
        ax.axvline(samples.mean(), color="#2c3e50", linewidth=2,
                   label=f"均值 = {samples.mean():.3f}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("HII")
        ax.set_ylabel("样本数")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    # 关掉空子图
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Monte Carlo (N={N_MC}) HII 后验分布及 95% 置信区间",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] MC 直方图: {save_path}")


def main() -> None:
    results_dir = ROOT / "results"
    fig_dir = ROOT / "fig"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    df, mc_all = run_all()
    out_csv = results_dir / "mc_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] MC 汇总表: {out_csv}")

    plot_histograms(mc_all, fig_dir / "mc_hist.png")

    # 打印人类可读表
    print("\n=== MC 判定结果 ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
