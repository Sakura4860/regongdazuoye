"""绘制网络拓扑图 + 故障响应对比测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from code.networks.station_5br import (
    apply_valve_setting,
    build_station_5br,
    inject_branch_blockage,
)
from code.solver import solve

# 支持中文
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw_topology(save_path: Path) -> None:
    """绘制换热站二次侧网络拓扑（5 支路）。"""
    G = nx.DiGraph()

    # 节点
    G.add_node("HE", pos=(0, 0), kind="heat_exchanger")
    G.add_node("Pump", pos=(-1.5, 0), kind="pump")
    G.add_node("S_out", pos=(1.5, 0), kind="junction")
    G.add_node("R_in", pos=(1.5, -3), kind="junction")

    n_branch = 5
    for i in range(1, n_branch + 1):
        G.add_node(f"Bs{i}", pos=(3 + i * 0.8, 0), kind="branch_sup")
        G.add_node(f"U{i}", pos=(3 + i * 0.8, -1.5), kind="user")
        G.add_node(f"Br{i}", pos=(3 + i * 0.8, -3), kind="branch_ret")

    # 边（管段+设备）
    G.add_edge("Pump", "HE", kind="supply")
    G.add_edge("HE", "S_out", kind="supply")
    G.add_edge("R_in", "Pump", kind="return")
    for i in range(1, n_branch + 1):
        G.add_edge("S_out", f"Bs{i}", kind="supply_header")
        G.add_edge(f"Bs{i}", f"U{i}", kind="branch_sup")
        G.add_edge(f"U{i}", f"Br{i}", kind="branch_ret")
        G.add_edge(f"Br{i}", "R_in", kind="return_header")

    pos = nx.get_node_attributes(G, "pos")
    kinds = nx.get_node_attributes(G, "kind")
    color_map = {
        "heat_exchanger": "#e27d60",
        "pump": "#85cdca",
        "junction": "#c1c1c1",
        "branch_sup": "#f9c784",
        "branch_ret": "#f9c784",
        "user": "#c38d9e",
    }
    node_colors = [color_map[kinds[n]] for n in G.nodes]

    fig, ax = plt.subplots(figsize=(11, 5))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=1400, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    edge_colors = {
        "supply": "#d63031",
        "supply_header": "#d63031",
        "branch_sup": "#fab1a0",
        "return_header": "#0984e3",
        "return": "#0984e3",
        "branch_ret": "#74b9ff",
    }
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=edge_colors[d["kind"]],
                               width=2.0, arrows=True,
                               arrowstyle="-|>", arrowsize=15, ax=ax)

    # 测点标注（示意）
    measurements = [
        (-0.7, 0.25, "T1/P1/Q0"),
        (-0.7, -3.25, "T2/P2"),
        (1.5, -1.5, "ΔP"),
        (3.8, -0.3, "Ts,i"),
        (3.8, -2.7, "Tr,i"),
        (3.8, -1.5, "Q_i, θ_i"),
    ]
    for x, y, label in measurements:
        ax.annotate(label, xy=(x, y), fontsize=8, color="darkgreen",
                    ha="center", bbox=dict(boxstyle="round,pad=0.2",
                                           fc="white", ec="green", alpha=0.9))

    ax.set_title("换热站二次侧 5 支路网络拓扑与测点示意", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ 拓扑图已保存: {save_path}")


def fault_response_test() -> None:
    """跑 4 个典型故障场景，对比流量响应。"""
    scenarios = {
        "正常工况": lambda net: None,
        "支路2阀门过关(开度30%)":
            lambda net: apply_valve_setting(net, {1: 1.0, 2: 0.3, 3: 1.0, 4: 1.0, 5: 1.0}),
        "支路4堵塞(阻力×2.5)":
            lambda net: inject_branch_blockage(net, 4, 2.5),
        "循环泵频率降至80%":
            lambda net: setattr(net, "pump_speed", 0.8),
    }

    print(f"\n{'='*60}\n故障响应对比\n{'='*60}")
    print(f"{'场景':30s} {'Q0':>8s} {'Q1':>7s} {'Q2':>7s} {'Q3':>7s} {'Q4':>7s} {'Q5':>7s} {'ΔP_h':>8s}")
    for name, apply_fault in scenarios.items():
        net = build_station_5br()
        apply_fault(net)
        sol = solve(net)
        qs = [sol.Q_branches[i] for i in range(1, 6)]
        print(f"{name:30s} {sol.Q_total:8.2f} "
              f"{qs[0]:7.2f} {qs[1]:7.2f} {qs[2]:7.2f} {qs[3]:7.2f} {qs[4]:7.2f} "
              f"{sol.dP_header:8.2f}")


def main() -> None:
    fig_dir = ROOT / "fig"
    fig_dir.mkdir(exist_ok=True)
    draw_topology(fig_dir / "network.png")
    fault_response_test()


if __name__ == "__main__":
    main()
