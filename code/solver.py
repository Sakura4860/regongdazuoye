"""
稳态水力求解器
==============

拓扑假设：单源 → 主管 → n 条并联支路 → 回主管 → 泵。
对这种拓扑 Hardy-Cross 退化为"节点连续性 + 支路压差相等"的耦合方程。
我们用 Brent 对支路压差做根求解，内层对每条支路解二次。

求解接口：solve(net) -> StateSolution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import brentq

from .twin import CP_WATER, RHO_WATER, Network, Pipe


@dataclass
class StateSolution:
    """水力热力稳态解。"""

    Q_total: float                    # 主管流量 m^3/h
    Q_branches: Dict[int, float]      # 支路流量
    dP_main_supply: float             # 供水主管压降 kPa
    dP_main_return: float             # 回水主管压降
    dP_header: float                  # 支路两端压差（各支路相同）
    dP_branches: Dict[int, float]     # 每支路实际压降（含支路管路+用户设备）
    H_pump: float                     # 泵扬程
    T_supply: float                   # 二次供水温度
    T_return: float                   # 二次回水总出口温度（混合后）
    T_supply_branches: Dict[int, float]
    T_return_branches: Dict[int, float]
    iterations: int


# ---------------------------------------------------------
# 支路阻力模型
# ---------------------------------------------------------
def branch_dP(net: Network, branch_id: int, Q_b: float) -> float:
    """给定支路流量，返回该支路总压降 kPa。

    包含：支路供管 + 支路回管 + 用户设备阻力（R_user * Q^2）。
    """
    R_user = net.branch_resistance[branch_id]
    dp = R_user * Q_b * abs(Q_b)
    for p in net.pipes:
        if p.branch_id == branch_id:
            dp += p.head_loss(Q_b)
    return dp


def solve_branch_flow(net: Network, branch_id: int, dP_target: float) -> float:
    """给定支路两端压差，反解支路流量。

    由于 dP = K(Q)*Q^2，且 K 微弱依赖 Q（通过 Reynolds），用
    不动点迭代：Q_new = sqrt(dP / K(Q_old))。
    """
    if dP_target <= 0:
        return 0.0
    Q = 10.0  # 初值 m^3/h
    for _ in range(50):
        K_user = net.branch_resistance[branch_id]
        K_pipes = 0.0
        for p in net.pipes:
            if p.branch_id == branch_id:
                K_pipes += p.K_resistance(Q)
        K_total = K_user + K_pipes
        if K_total <= 1e-12:
            return 0.0
        Q_new = np.sqrt(dP_target / K_total)
        if abs(Q_new - Q) < 1e-4:
            return Q_new
        Q = 0.5 * (Q + Q_new)  # 阻尼
    return Q


def solve(
    net: Network,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> StateSolution:
    """主求解函数。

    思路：未知量 = 供回主管末端的压差 dP_header。
    给定 dP_header → 解出各支路 Q_i → Q_total = Σ Q_i → 检查泵方程。

    残差: F(dP_header) = H_pump(Q_total) - R_main(Q_total) * Q_total^2 - dP_header
    用 brentq 在 [1, H_pump_max] 上找零点。
    """
    def residual(dP_header: float) -> float:
        Q_b = {}
        for bid in net.branch_resistance:
            Q_b[bid] = solve_branch_flow(net, bid, dP_header)
        Q_tot = sum(Q_b.values())
        dp_main_sup = net.main_supply_pipe.head_loss(Q_tot) if net.main_supply_pipe else 0.0
        dp_main_ret = net.main_return_pipe.head_loss(Q_tot) if net.main_return_pipe else 0.0
        H = net.pump_head(Q_tot)
        return H - dp_main_sup - dp_main_ret - dP_header

    # 搜索区间
    # 最大扬程约 H_pump_max 时流量最小；最小 dP_header 时总流量最大
    H_max = net.pump_head(0.0)
    try:
        dp_h_solve = brentq(residual, 1.0, H_max - 1.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        # 扩大区间
        dp_h_solve = brentq(residual, 0.01, H_max * 0.999, xtol=tol, maxiter=max_iter)

    # 收敛后再计算一次状态
    Q_b = {bid: solve_branch_flow(net, bid, dp_h_solve) for bid in net.branch_resistance}
    Q_tot = sum(Q_b.values())
    dp_main_sup = net.main_supply_pipe.head_loss(Q_tot) if net.main_supply_pipe else 0.0
    dp_main_ret = net.main_return_pipe.head_loss(Q_tot) if net.main_return_pipe else 0.0
    H = net.pump_head(Q_tot)
    dp_branches = {bid: branch_dP(net, bid, Q_b[bid]) for bid in Q_b}

    # 温度：假设各支路供水温度 = 主管供水温度，回水温度由设定或默认得到
    T_sup_each = {bid: net.T_supply for bid in Q_b}
    T_ret_each = {bid: net.branch_return_T.get(bid, net.T_supply - 15.0) for bid in Q_b}
    # 混合回水总温度
    if Q_tot > 1e-6:
        T_ret_total = sum(Q_b[bid] * T_ret_each[bid] for bid in Q_b) / Q_tot
    else:
        T_ret_total = net.T_supply

    return StateSolution(
        Q_total=Q_tot,
        Q_branches=Q_b,
        dP_main_supply=dp_main_sup,
        dP_main_return=dp_main_ret,
        dP_header=dp_h_solve,
        dP_branches=dp_branches,
        H_pump=H,
        T_supply=net.T_supply,
        T_return=T_ret_total,
        T_supply_branches=T_sup_each,
        T_return_branches=T_ret_each,
        iterations=max_iter,
    )


# ---------------------------------------------------------
# 辅助：打印稳态结果
# ---------------------------------------------------------
def pretty_print(sol: StateSolution) -> str:
    lines = [
        f"=== 稳态水力解 ===",
        f"总流量 Q0      = {sol.Q_total:7.3f} m^3/h",
        f"泵扬程 H_pump  = {sol.H_pump:7.2f} kPa",
        f"供主管压降    = {sol.dP_main_supply:7.2f} kPa",
        f"回主管压降    = {sol.dP_main_return:7.2f} kPa",
        f"支路两端压差  = {sol.dP_header:7.2f} kPa",
        f"二次供水温度  = {sol.T_supply:6.2f} ℃",
        f"二次回水温度  = {sol.T_return:6.2f} ℃",
        f"",
        f"支路编号  流量(m^3/h)  压降(kPa)  供温(℃)  回温(℃)",
    ]
    for bid in sorted(sol.Q_branches):
        lines.append(
            f"  {bid:>4d}    {sol.Q_branches[bid]:8.3f}    "
            f"{sol.dP_branches[bid]:7.2f}   {sol.T_supply_branches[bid]:6.2f}   "
            f"{sol.T_return_branches[bid]:6.2f}"
        )
    return "\n".join(lines)
