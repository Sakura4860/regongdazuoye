"""简单的 demo：构建 5 支路网络并求解。"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保 repo 根在 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.networks.station_5br import build_station_5br
from code.solver import pretty_print, solve


def main() -> None:
    net = build_station_5br()
    print(f"管段总数: {len(net.pipes) + 2}")
    print(f"支路总数: {net.n_branches()}")
    sol = solve(net)
    print(pretty_print(sol))

    # 校核：流量平衡
    q_sum = sum(sol.Q_branches.values())
    print(f"\n流量平衡检查: Q_total={sol.Q_total:.3f}, ΣQ_i={q_sum:.3f}, 偏差={abs(q_sum - sol.Q_total):.4f}")


if __name__ == "__main__":
    main()
