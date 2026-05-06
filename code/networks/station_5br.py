"""
换热站二次侧管网 5 支路实例
==========================

按题目工况：
- DN100 主管（内径约 102 mm，钢管）
- 供水 65℃ / 回水 50℃（落在 55–70 / 40–55 范围内）
- 总循环流量 50 m^3/h（落在 30–80 范围内）
- 干管供回水压差 ~80 kPa（落在 40–120 范围内）
- 5 条支路（落在 4–6 范围内）

支路参数简化：
- 支路长度 20~60 m，DN50 管（内径 53 mm）
- 每条支路用户设备按等效阻力给定
- 阀门初始开度 = 1.0 (全开)
"""

from __future__ import annotations

from typing import Dict

from code.twin import Network, Pipe


# 支路基础设计参数（设计值对应 HII≈0）
DESIGN_FLOW_TOTAL = 50.0  # m^3/h
DESIGN_FLOW_PER_BRANCH = 10.0  # m^3/h，5 条支路均分
DESIGN_SUPPLY_T = 65.0
DESIGN_RETURN_T = 50.0


def build_station_5br() -> Network:
    """构建 5 支路换热站二次侧网络。"""
    net = Network()

    # ---------- 主管（DN100 钢管）----------
    net.main_supply_pipe = Pipe(
        pid=0, L=30.0, D=0.102, eps=4.6e-5, branch_id=0, extra_K=0.5
    )
    net.main_return_pipe = Pipe(
        pid=1, L=30.0, D=0.102, eps=4.6e-5, branch_id=0, extra_K=0.5
    )

    # ---------- 5 条支路（DN50）----------
    branch_lengths = {1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0, 5: 60.0}
    # 用户设备等效阻力（使各支路在设计流量 10 m^3/h 时压降近似一致）
    # 先做个粗略设计：希望设计时各支路两端压差 ≈ 50 kPa
    # 已知 hf_pipe ≈ f*L/D*v^2/2*rho, 对 L 变化做补偿：短支路 Ruser 大、长支路小
    # 简化：给每条支路一个"设备阻力系数"
    R_users = {
        1: 0.480,   # 短支路 → 设备阻力大（用户远端资用压头少）
        2: 0.470,
        3: 0.460,
        4: 0.450,
        5: 0.440,   # 长支路 → 设备阻力小
    }

    pid = 2
    for bid, L in branch_lengths.items():
        # 供管
        net.pipes.append(Pipe(pid=pid, L=L, D=0.053, eps=4.6e-5,
                              branch_id=bid, extra_K=1.0))
        pid += 1
        # 回管
        net.pipes.append(Pipe(pid=pid, L=L, D=0.053, eps=4.6e-5,
                              branch_id=bid, extra_K=1.0))
        pid += 1
        net.branch_resistance[bid] = R_users[bid]
        # 各支路设计回水温度（假设用户负荷一致）
        net.branch_return_T[bid] = DESIGN_RETURN_T

    # ---------- 循环泵（变频离心泵）----------
    net.pump_H0 = 180.0    # kPa
    net.pump_a = 0.03      # kPa/(m^3/h)^2
    net.pump_speed = 1.0   # 设计工况
    # 额定工况: H(50) = 180 - 0.03*2500 = 180 - 75 = 105 kPa

    net.T_supply = DESIGN_SUPPLY_T
    return net


def apply_valve_setting(net: Network, valve_openings: Dict[int, float]) -> None:
    """设定每条支路的阀门开度。valve_openings = {branch_id: 0.0~1.0}

    简化处理：把阀门附加阻力加到支路"用户设备阻力"里。
    开度越小，阻力越大（指数模型）。
    """
    base_R = {
        1: 0.480, 2: 0.470, 3: 0.460, 4: 0.450, 5: 0.440,
    }
    for bid, opening in valve_openings.items():
        opening = max(0.01, min(1.0, opening))
        # 等百分比阀模型：R = R_base / opening^2
        net.branch_resistance[bid] = base_R[bid] / opening ** 2


def inject_branch_blockage(net: Network, branch_id: int, factor: float) -> None:
    """在某条支路上叠加堵塞阻力。factor > 1.0 表示堵塞加剧。"""
    net.branch_resistance[branch_id] *= factor
