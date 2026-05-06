"""
综合水力失调指数 HII
===================

基于 solution-plan §5.1：

θ_Q,i = (Q_i − Q_i,设计) / Q_i,设计
θ_P   = std{ΔP_i} / mean{ΔP_i}
θ_T,i = (ΔT_i − ΔT̄) / ΔT̄

HII = w_Q · max_i|θ_Q,i| + w_P · θ_P + w_T · max_i|θ_T,i|

权重默认 w_Q=0.5 / w_P=0.3 / w_T=0.2。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# 分级阈值
HII_LEVELS = [
    (0.10, "正常",   "继续监视"),
    (0.20, "轻度失调", "下次巡检复查"),
    (0.35, "中度失调", "调节阀门/循环泵频率"),
    (np.inf, "重度失调", "启动诊断 → AI 定位支路"),
]


@dataclass
class HIIResult:
    """HII 计算结果及其分项贡献。"""

    hii: float
    level: str
    suggestion: str
    theta_Q_max: float
    theta_P: float
    theta_T_max: float
    per_branch_Q: Dict[int, float] = field(default_factory=dict)
    per_branch_T: Dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "hii": self.hii,
            "level": self.level,
            "suggestion": self.suggestion,
            "theta_Q_max": self.theta_Q_max,
            "theta_P": self.theta_P,
            "theta_T_max": self.theta_T_max,
        }


def classify(hii: float) -> tuple[str, str]:
    """按 HII 返回 (等级, 建议)。"""
    for thr, level, sug in HII_LEVELS:
        if hii < thr:
            return level, sug
    return HII_LEVELS[-1][1], HII_LEVELS[-1][2]


def compute_hii(
    Q_branches: Dict[int, float],
    Q_design: Dict[int, float],
    dP_branches: Dict[int, float],
    dT_branches: Dict[int, float],
    weights: Optional[Dict[str, float]] = None,
) -> HIIResult:
    """计算综合水力失调指数。

    Args:
        Q_branches: 实测支路流量 {bid: Q_i(m^3/h)}。
        Q_design: 设计支路流量 {bid: Q_i_design}。
        dP_branches: 实测支路两端压差 {bid: ΔP_i(kPa)}。
        dT_branches: 实测支路温差 {bid: ΔT_i(℃) = T_sup - T_ret}。
        weights: 权重字典，默认 w_Q=0.5, w_P=0.3, w_T=0.2。

    Returns:
        HIIResult 数据类。
    """
    if weights is None:
        weights = {"Q": 0.5, "P": 0.3, "T": 0.2}

    # θ_Q,i —— 支路流量偏差
    theta_Q = {}
    for bid, Q in Q_branches.items():
        Qd = Q_design.get(bid, Q)
        theta_Q[bid] = (Q - Qd) / max(abs(Qd), 1e-6)
    theta_Q_max = max(abs(v) for v in theta_Q.values()) if theta_Q else 0.0

    # θ_P —— 支路压差分散度
    dp_vals = np.array(list(dP_branches.values()), dtype=float)
    if len(dp_vals) >= 2 and dp_vals.mean() > 1e-6:
        theta_P = float(dp_vals.std() / dp_vals.mean())
    else:
        theta_P = 0.0

    # θ_T,i —— 温差一致性
    dt_vals = np.array(list(dT_branches.values()), dtype=float)
    dt_mean = float(dt_vals.mean()) if len(dt_vals) else 1.0
    if abs(dt_mean) > 1e-6:
        theta_T = {bid: (v - dt_mean) / dt_mean
                   for bid, v in dT_branches.items()}
    else:
        theta_T = {bid: 0.0 for bid in dT_branches}
    theta_T_max = max(abs(v) for v in theta_T.values()) if theta_T else 0.0

    # 综合
    hii = (weights["Q"] * theta_Q_max +
           weights["P"] * theta_P +
           weights["T"] * theta_T_max)
    level, sug = classify(hii)
    return HIIResult(
        hii=hii, level=level, suggestion=sug,
        theta_Q_max=theta_Q_max, theta_P=theta_P, theta_T_max=theta_T_max,
        per_branch_Q=theta_Q, per_branch_T=theta_T,
    )


# ---------------------------------------------------------
# 从 StateSolution 直接计算的便捷接口
# ---------------------------------------------------------
def hii_from_solution(
    sol,
    Q_design: Dict[int, float],
    weights: Optional[Dict[str, float]] = None,
) -> HIIResult:
    """直接用求解器输出构造 HII。"""
    dT = {bid: sol.T_supply_branches[bid] - sol.T_return_branches[bid]
          for bid in sol.Q_branches}
    return compute_hii(
        Q_branches=sol.Q_branches,
        Q_design=Q_design,
        dP_branches=sol.dP_branches,
        dT_branches=dT,
        weights=weights,
    )


# ---------------------------------------------------------
# 自检
# ---------------------------------------------------------
if __name__ == "__main__":
    import io
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")

    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from code.networks.station_5br import (
        DESIGN_FLOW_PER_BRANCH,
        apply_valve_setting,
        build_station_5br,
        inject_branch_blockage,
    )
    from code.solver import solve

    Q_des = {i: DESIGN_FLOW_PER_BRANCH for i in range(1, 6)}

    scenarios = {
        "正常工况": lambda n: None,
        "支路2阀关30%": lambda n: apply_valve_setting(
            n, {1: 1.0, 2: 0.3, 3: 1.0, 4: 1.0, 5: 1.0}),
        "支路4堵塞×2.5": lambda n: inject_branch_blockage(n, 4, 2.5),
        "支路4堵塞×5.0": lambda n: inject_branch_blockage(n, 4, 5.0),
        "泵频 80%":     lambda n: setattr(n, "pump_speed", 0.8),
    }

    print(f"{'场景':<18s}  {'HII':>6s}  {'等级':<10s}  θ_Q    θ_P    θ_T")
    print("-" * 70)
    for name, apply in scenarios.items():
        net = build_station_5br()
        apply(net)
        sol = solve(net)
        r = hii_from_solution(sol, Q_des)
        print(f"{name:<18s}  {r.hii:6.3f}  {r.level:<10s}  "
              f"{r.theta_Q_max:.3f}  {r.theta_P:.3f}  {r.theta_T_max:.3f}")
