"""
换热站二次侧管网 水力仿真内核
===============================

基于 Hardy-Cross 迭代法，求解稳态流量与压降分布。
面向热工测试大作业（DN100 主管 + 4~6 条支路）。

坐标约定：
- 支路编号从 1 开始。
- 正向流动 = 供水方向。
- 压降 hf = K * |Q|^n * sign(Q)，n=2 (Darcy-Weisbach)。

Author: 热工测试小组
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------
# 物性参数 (热水, 50~70℃ 段)
# ------------------------------------------------------------
G = 9.81           # 重力加速度 m/s^2
RHO_WATER = 985.0  # 60℃ 水的密度 kg/m^3
MU_WATER = 4.66e-4 # 60℃ 水的动力粘度 Pa·s
CP_WATER = 4180.0  # 比热容 J/(kg·K)


# ------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------
@dataclass
class Pipe:
    """单段管道。

    参数
    ----
    pid : 管段 ID，全局唯一
    L   : 长度 m
    D   : 内径 m
    eps : 粗糙度 m（钢管 0.046mm = 4.6e-5 m）
    branch_id : 属于哪一支路（0 = 主管）
    """

    pid: int
    L: float
    D: float
    eps: float = 4.6e-5
    branch_id: int = 0
    # 额外局部阻力（阀门、弯头等），用等效长度或 K 系数表示
    extra_K: float = 0.0       # 直接按 hf = K * Q^2 叠加
    valve_open: float = 1.0    # 阀门开度 [0,1]，影响 extra_K

    def area(self) -> float:
        return np.pi * self.D ** 2 / 4.0

    def friction_factor(self, Q: float) -> float:
        """Colebrook-White 近似 (Swamee-Jain)。Q 单位 m^3/h。"""
        if abs(Q) < 1e-8:
            return 0.02
        Q_si = Q / 3600.0  # m^3/s
        v = Q_si / self.area()
        Re = RHO_WATER * abs(v) * self.D / MU_WATER
        if Re < 2000:
            return 64.0 / max(Re, 1.0)
        rel = self.eps / self.D
        f = 0.25 / (np.log10(rel / 3.7 + 5.74 / Re ** 0.9)) ** 2
        return f

    def K_resistance(self, Q: float) -> float:
        """总阻力系数 K，使 hf = K * Q^2 (Q m^3/h, hf kPa)。

        以 SI 算再换算: hf_SI = f L/D * v^2 / 2 * rho (Pa)
        Q(m^3/h) -> Q_si(m^3/s)=Q/3600, v=Q_si/A
        hf_Pa = f*L/D * RHO/2 * (Q/3600/A)^2
        hf_kPa = hf_Pa / 1000
        """
        f = self.friction_factor(Q)
        A = self.area()
        coef = (f * self.L / self.D + self.extra_K +
                (1.0 - self.valve_open) * 50.0 / max(self.valve_open, 1e-3))
        k_si = coef * RHO_WATER / 2.0 / A ** 2   # 对应 Q_si
        # 换算到 Q(m^3/h): k_h = k_si / 3600^2, 再换算单位成 kPa
        k_h = k_si / 3600.0 ** 2 / 1000.0
        return k_h

    def head_loss(self, Q: float) -> float:
        """返回沿程+局阻压降，kPa，含方向。"""
        K = self.K_resistance(Q)
        return K * Q * abs(Q)


@dataclass
class Network:
    """管网拓扑。

    约定：
    - 供回水对称，每条支路由一段"供管"和一段"回管"组成，本求解器
      以"等效阻力"的观点把供+回合并成"支路串联阻力"。
    - 主管前向阻力单独处理。
    """

    pipes: List[Pipe] = field(default_factory=list)
    # 支路总阻力（供+回+用户设备），按 branch_id 存
    branch_resistance: Dict[int, float] = field(default_factory=dict)
    # 主管阻力（供+回干管）
    main_supply_pipe: Optional[Pipe] = None
    main_return_pipe: Optional[Pipe] = None
    # 循环泵：给定扬程与 Q 的二次关系 H(Q) = H0 - a*Q^2
    pump_H0: float = 150.0   # kPa
    pump_a: float = 0.01     # kPa / (m^3/h)^2
    pump_speed: float = 1.0  # 变频比例 [0.5, 1.2]
    # 换热器一次侧给定二次侧供水温度（℃）
    T_supply: float = 65.0
    # 各支路末端回水温度（由负荷决定，简化给定）
    branch_return_T: Dict[int, float] = field(default_factory=dict)

    def n_branches(self) -> int:
        return len(self.branch_resistance)

    def pump_head(self, Q_total: float) -> float:
        """泵的 H-Q 曲线, Q m^3/h, H kPa。"""
        s = self.pump_speed
        return self.pump_H0 * s ** 2 - self.pump_a * (Q_total ** 2) * s
