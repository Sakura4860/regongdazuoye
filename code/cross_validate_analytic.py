"""
独立求解器交叉验证
==================

用 scipy.optimize.fsolve 把整个网络作为多元非线性方程组求解，与主方案
solver.solve()（brentq + 不动点迭代）对比。若两个独立数值算法在多个扰动
工况下得到的解一致（相对差异 <1%），则可确认物理模型实现正确。

方程组（7 个未知量：Q_0, Q_1..5, ΔP_header）：
  1) Q_0 − Σ Q_i = 0
  2..6) ΔP_header − (R_user_i + R_pipe_i(Q_i)) · Q_i · |Q_i| = 0
  7) H_pump(Q_0) − (ΔP_main_sup + ΔP_main_ret)(Q_0) − ΔP_header = 0

产出：results/cross_validation.csv
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from code.networks.station_5br import (
    apply_valve_setting,
    build_station_5br,
    inject_branch_blockage,
)
from code.solver import solve


def fsolve_network(net) -> dict:
    """用 fsolve 求解整个网络，返回稳态解。"""
    branches = sorted(net.branch_resistance.keys())
    n_b = len(branches)

    def branch_K(bid, Q):
        """支路 bid 在流量 Q 下的总阻力系数 (用户 + 管道)。"""
        K = net.branch_resistance[bid]
        for p in net.pipes:
            if p.branch_id == bid:
                K += p.K_resistance(Q)
        return K

    def main_K(Q0):
        K = 0.0
        if net.main_supply_pipe is not None:
            K += net.main_supply_pipe.K_resistance(Q0)
        if net.main_return_pipe is not None:
            K += net.main_return_pipe.K_resistance(Q0)
        return K

    def residuals(x):
        Q0 = x[0]
        Qb = x[1:1 + n_b]
        dP_h = x[-1]

        r = np.empty(n_b + 2)
        # (1) 连续性
        r[0] = Q0 - Qb.sum()
        # (2-6) 支路压差相等
        for i, bid in enumerate(branches):
            K = branch_K(bid, Qb[i])
            r[1 + i] = dP_h - K * Qb[i] * abs(Qb[i])
        # (7) 泵平衡
        Kmain = main_K(Q0)
        r[-1] = net.pump_head(Q0) - Kmain * Q0 * abs(Q0) - dP_h
        return r

    # 初值
    x0 = np.concatenate([[50.0], np.full(n_b, 10.0), [60.0]])
    sol_vec, info, ier, msg = fsolve(residuals, x0, full_output=True, xtol=1e-8)
    if ier != 1:
        print(f"  [warn] fsolve 收敛标志 ier={ier}: {msg}")

    Q0 = float(sol_vec[0])
    Qb = {bid: float(sol_vec[1 + i]) for i, bid in enumerate(branches)}
    dP_h = float(sol_vec[-1])
    return {
        "Q_total": Q0,
        "Q_branches": Qb,
        "dP_header": dP_h,
        "residual_norm": float(np.linalg.norm(info["fvec"])),
    }


def compare(solve_sol, fsolve_sol) -> dict:
    """计算两个求解器的相对差异。"""
    rel_Q0 = abs(solve_sol.Q_total - fsolve_sol["Q_total"]) \
        / max(abs(solve_sol.Q_total), 1e-6)
    rel_dP = abs(solve_sol.dP_header - fsolve_sol["dP_header"]) \
        / max(abs(solve_sol.dP_header), 1e-6)
    rel_Q_branches = {}
    for bid, Q_hc in solve_sol.Q_branches.items():
        Q_fs = fsolve_sol["Q_branches"][bid]
        rel_Q_branches[bid] = abs(Q_hc - Q_fs) / max(abs(Q_hc), 1e-6)
    rel_Q_max = max(rel_Q_branches.values())
    return {
        "rel_Q_total": rel_Q0,
        "rel_dP_header": rel_dP,
        "rel_Q_branch_max": rel_Q_max,
        "per_branch": rel_Q_branches,
    }


SCENARIOS = [
    ("正常工况",        lambda net: None),
    ("支路2阀55%",      lambda net: apply_valve_setting(
        net, {1: 1.0, 2: 0.55, 3: 1.0, 4: 1.0, 5: 1.0})),
    ("支路2阀30%",      lambda net: apply_valve_setting(
        net, {1: 1.0, 2: 0.30, 3: 1.0, 4: 1.0, 5: 1.0})),
    ("支路4堵塞×2.5",  lambda net: inject_branch_blockage(net, 4, 2.5)),
    ("支路4堵塞×5.0",  lambda net: inject_branch_blockage(net, 4, 5.0)),
    ("泵频 80%",       lambda net: setattr(net, "pump_speed", 0.8)),
    ("泵频 110%",      lambda net: setattr(net, "pump_speed", 1.1)),
]


def run_all() -> pd.DataFrame:
    rows = []
    print(f"{'场景':<18s}  {'HC Q0':>8s}  {'FS Q0':>8s}  "
          f"{'rel_Q0':>8s}  {'rel_dP':>8s}  {'maxRelBr':>9s}  res_norm")
    print("-" * 82)
    for name, apply in SCENARIOS:
        net_hc = build_station_5br()
        if apply is not None:
            apply(net_hc)
        sol_hc = solve(net_hc)

        net_fs = build_station_5br()
        if apply is not None:
            apply(net_fs)
        sol_fs = fsolve_network(net_fs)

        diff = compare(sol_hc, sol_fs)
        rows.append({
            "scenario": name,
            "Q0_HC": round(sol_hc.Q_total, 4),
            "Q0_fsolve": round(sol_fs["Q_total"], 4),
            "dP_HC": round(sol_hc.dP_header, 3),
            "dP_fsolve": round(sol_fs["dP_header"], 3),
            "rel_Q_total_pct": round(diff["rel_Q_total"] * 100, 4),
            "rel_dP_header_pct": round(diff["rel_dP_header"] * 100, 4),
            "rel_Q_branch_max_pct": round(diff["rel_Q_branch_max"] * 100, 4),
            "fsolve_residual": round(sol_fs["residual_norm"], 2),
        })
        print(f"{name:<18s}  {sol_hc.Q_total:8.3f}  {sol_fs['Q_total']:8.3f}  "
              f"{diff['rel_Q_total']*100:7.4f}%  {diff['rel_dP_header']*100:7.4f}%  "
              f"{diff['rel_Q_branch_max']*100:8.4f}%  {sol_fs['residual_norm']:.2e}")
    return pd.DataFrame(rows)


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    df = run_all()
    out_csv = results_dir / "cross_validation.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 最大相对差异
    max_rel = max(df["rel_Q_total_pct"].max(),
                  df["rel_dP_header_pct"].max(),
                  df["rel_Q_branch_max_pct"].max())
    print(f"\n最大相对差异: {max_rel:.4f}%")
    if max_rel < 1.0:
        print("[PASS] 两个独立求解器一致性验证通过（相对差异 < 1%）。")
    else:
        print(f"[WARN] 相对差异 {max_rel:.4f}% 超过 1% 阈值，请检查。")

    print(f"\n[OK] 交叉验证结果: {out_csv}")


if __name__ == "__main__":
    main()
