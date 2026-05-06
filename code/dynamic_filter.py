"""
动态响应处理
============

按 solution-plan §5.3 实现四件事：
1) EWMA 滑动滤波 (α=0.2)
2) 工况归一化 (Q_i / Q_0)
3) 稳态判定 (滑窗 5min, σ/μ < 3%)
4) 温度反卷积 (一阶 τ≈30s)

合成一段 30 min、1 Hz 采样的"阀门慢关→新稳态"时序，演示上述处理。
产出 fig/dynamic_response.png。
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# -------------------------------------------------------------
# 核心算法
# -------------------------------------------------------------
def ewma(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """EWMA 滑动滤波。y_t = α·x_t + (1-α)·y_{t-1}。"""
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


def rolling_steady_mask(
    x: np.ndarray, window: int = 300, thr: float = 0.03
) -> np.ndarray:
    """对每个时刻判断过去 window 秒是否稳态（σ/μ < thr）。

    Returns:
        bool array，True 表示该时刻处于稳态（可以判定）。
    """
    mask = np.zeros(len(x), dtype=bool)
    for t in range(window, len(x)):
        w = x[t - window:t]
        mu = w.mean()
        if mu > 1e-6:
            mask[t] = (w.std() / mu) < thr
    return mask


def normalize_by_total(Q_branch: np.ndarray, Q0: np.ndarray) -> np.ndarray:
    """流量工况归一化：Q_i 除以总流量 Q_0。"""
    return Q_branch / np.maximum(Q0, 1e-6)


def temp_deconv(T_meas: np.ndarray, tau: float = 30.0, dt: float = 1.0,
                pre_smooth_alpha: float = 0.1) -> np.ndarray:
    """Pt100 一阶热惯性反卷积：T_real ≈ T_meas + τ·dT/dt。

    直接对原始测量做微分会把高频噪声放大 τ/dt 倍。先用 EWMA 低通滤波，
    再做反卷积，是工程上的标准做法。

    Args:
        T_meas: 测量温度时序 (℃)
        tau:    时间常数 (s)，Pt100 铠装 ≈ 30s
        dt:     采样间隔 (s)
        pre_smooth_alpha: 预滤波 EWMA 系数，越小越平滑（默认 0.1）
    """
    T_smooth = ewma(T_meas, alpha=pre_smooth_alpha)
    dTdt = np.gradient(T_smooth, dt)
    return T_smooth + tau * dTdt


# -------------------------------------------------------------
# 合成时间序列：阀门慢关 → 新稳态
# -------------------------------------------------------------
def synth_scenario(seed: int = 2026) -> dict:
    """生成 30 min / 1Hz 的阀门慢关场景。

    - 0-600 s: 正常稳态
    - 600-900 s: 支路2阀门由 100% 线性关到 40%（过渡期）
    - 900-1800 s: 新稳态

    所有流量/压差/温度按稳态水力关系近似线性化，加测量噪声。
    """
    rng = np.random.default_rng(seed)
    n = 1800
    t = np.arange(n, dtype=float)

    # 目标稳态值（来自 solve 的典型数）
    Q0_nom = 50.0        # m^3/h 总流量
    Q0_post = 47.5       # 新稳态（阀关导致略降）

    Q2_nom = 10.0        # 支路 2 原流量
    Q2_post = 5.5        # 支路 2 阀关 40% 后流量

    Q_other_nom = 10.0   # 其他支路设计流量
    # 其他支路在阀关后流量略增（因为水往阻力小的地方走）
    Q_other_post = (Q0_post - Q2_post) / 4.0

    # 阀门过渡段的平滑 ramp (tanh)
    # 从 t=600 到 t=900，线性过渡
    ramp = np.clip((t - 600) / 300.0, 0.0, 1.0)

    Q0_true = Q0_nom + (Q0_post - Q0_nom) * ramp
    Q2_true = Q2_nom + (Q2_post - Q2_nom) * ramp
    Qo_true = Q_other_nom + (Q_other_post - Q_other_nom) * ramp

    # 测量噪声（ppm 级别的 1Hz 白噪声）
    noise_Q0 = rng.normal(0, 0.005 * Q0_nom, size=n)           # 0.5% 电磁
    noise_Q2 = rng.normal(0, 0.02 * Q2_nom, size=n)            # 2% 超声
    noise_Qo = rng.normal(0, 0.02 * Q_other_nom, size=n)

    Q0_meas = Q0_true + noise_Q0
    Q2_meas = Q2_true + noise_Q2
    Qo_meas = Qo_true + noise_Qo

    # 温度（真实）：假定支路 2 阀关后温差下降（流量小于设计 → 散热慢）
    dT2_nom = 15.0
    dT2_post = 8.0
    dT2_true = dT2_nom + (dT2_post - dT2_nom) * ramp

    # 温度测量：Pt100 一阶滞后 τ=30s + 测量噪声
    tau = 30.0
    T_s_true = 65.0 + np.zeros(n)   # 供水温度假定稳定
    T_r_true = T_s_true - dT2_true   # 支路 2 回水温度

    # 一阶滞后：T_meas = T_true ⊛ (1 - exp(-t/τ))
    def first_order_lag(x, tau, dt=1.0):
        y = np.empty_like(x)
        y[0] = x[0]
        k = dt / tau
        for i in range(1, len(x)):
            y[i] = y[i - 1] + k * (x[i - 1] - y[i - 1])
        return y

    T_r_meas_no_noise = first_order_lag(T_r_true, tau)
    T_r_meas = T_r_meas_no_noise + rng.normal(0, 0.15 / np.sqrt(3), size=n)

    return {
        "t": t,
        "Q0_true": Q0_true, "Q0_meas": Q0_meas,
        "Q2_true": Q2_true, "Q2_meas": Q2_meas,
        "Qo_true": Qo_true, "Qo_meas": Qo_meas,
        "T_r_true": T_r_true, "T_r_meas": T_r_meas,
        "dT2_true": dT2_true,
    }


# -------------------------------------------------------------
# 由时序推导 HII（简化版：只看 θ_Q 分量，演示滤波效果）
# -------------------------------------------------------------
def hii_series(Q2: np.ndarray, Qo: np.ndarray,
               Q2_design: float = 10.0, Qo_design: float = 10.0,
               w_Q: float = 0.5) -> np.ndarray:
    """演示用：只用流量偏差贡献的简化 HII。"""
    theta_Q2 = (Q2 - Q2_design) / Q2_design
    theta_Qo = (Qo - Qo_design) / Qo_design
    theta_max = np.maximum(np.abs(theta_Q2), np.abs(theta_Qo))
    return w_Q * theta_max


# -------------------------------------------------------------
# 绘图
# -------------------------------------------------------------
def plot_all(data: dict, save_path: Path) -> None:
    t = data["t"] / 60.0   # 转为分钟

    # 1) EWMA on Q2
    Q2_ewma = ewma(data["Q2_meas"], alpha=0.2)
    Qo_ewma = ewma(data["Qo_meas"], alpha=0.2)
    Q0_ewma = ewma(data["Q0_meas"], alpha=0.2)

    # 2) 稳态判定（基于 Q0）
    steady = rolling_steady_mask(data["Q0_meas"], window=300, thr=0.03)

    # 3) 简化 HII 时序
    hii_raw = hii_series(data["Q2_meas"], data["Qo_meas"])
    hii_flt = hii_series(Q2_ewma, Qo_ewma)

    # 4) 温度反卷积
    T_r_deconv = temp_deconv(data["T_r_meas"], tau=30.0, dt=1.0)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    # ---- 子图 1：支路 2 流量 ----
    ax = axes[0]
    ax.plot(t, data["Q2_meas"], color="#bdc3c7", lw=0.8, label="原始测量")
    ax.plot(t, Q2_ewma, color="#c0392b", lw=1.8, label="EWMA (α=0.2)")
    ax.plot(t, data["Q2_true"], color="#2980b9", lw=1.4, ls="--", label="真值")
    ax.set_ylabel("支路 2 流量 (m³/h)")
    ax.set_title("支路 2 流量：阀门慢关过程（600–900 s），EWMA 滤波效果")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.axvspan(10, 15, color="orange", alpha=0.15, label="过渡期")

    # ---- 子图 2：温度反卷积 ----
    ax = axes[1]
    ax.plot(t, data["T_r_true"], color="#2980b9", lw=1.6, ls="--", label="真温")
    ax.plot(t, data["T_r_meas"], color="#bdc3c7", lw=0.8, label="Pt100 测量")
    ax.plot(t, T_r_deconv, color="#27ae60", lw=1.6, label="反卷积 (τ=30s)")
    ax.set_ylabel("支路 2 回水温度 (℃)")
    ax.set_title("温度一阶反卷积：补偿 Pt100 热惯性")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- 子图 3：稳态判定 ----
    ax = axes[2]
    ax.plot(t, data["Q0_meas"], color="#7f8c8d", lw=0.7, label="Q0 测量")
    ax.plot(t, Q0_ewma, color="#2c3e50", lw=1.5, label="Q0 EWMA")
    # 非稳态时段阴影
    non_steady = ~steady
    # 画阴影：找到连续的 False 段
    i = 0
    first = True
    while i < len(non_steady):
        if non_steady[i]:
            j = i
            while j < len(non_steady) and non_steady[j]:
                j += 1
            ax.axvspan(t[i], t[j - 1], color="#e74c3c", alpha=0.18,
                       label="挂起诊断" if first else None)
            first = False
            i = j
        else:
            i += 1
    ax.set_ylabel("主管流量 Q0 (m³/h)")
    ax.set_title("稳态判定 (σ/μ < 3% on 5min 窗)：非稳态期挂起诊断")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- 子图 4：简化 HII 时序 ----
    ax = axes[3]
    ax.plot(t, hii_raw, color="#bdc3c7", lw=0.8, label="HII（原始）")
    ax.plot(t, hii_flt, color="#8e44ad", lw=1.8, label="HII（EWMA）")
    ax.axhline(0.10, ls=":", color="gray")
    ax.axhline(0.20, ls=":", color="gray")
    ax.axhline(0.35, ls=":", color="gray")
    ax.text(0.3, 0.105, "轻度", fontsize=8, color="gray")
    ax.text(0.3, 0.205, "中度", fontsize=8, color="gray")
    ax.text(0.3, 0.355, "重度", fontsize=8, color="gray")
    ax.set_ylabel("简化 HII (仅流量分量)")
    ax.set_xlabel("时间 (min)")
    ax.set_title("HII 时序：滤波可显著压低伪失调信号")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 0.40)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 动态响应图: {save_path}")


def main() -> None:
    fig_dir = ROOT / "fig"
    fig_dir.mkdir(exist_ok=True)

    data = synth_scenario()
    plot_all(data, fig_dir / "dynamic_response.png")

    # 打印关键数字
    Q2_ewma = ewma(data["Q2_meas"], 0.2)
    print(f"支路 2 阀关前后:")
    print(f"  真值:   {data['Q2_true'][0]:.2f} → {data['Q2_true'][-1]:.2f} m³/h")
    print(f"  测量 1Hz 噪声 σ: {(data['Q2_meas'] - data['Q2_true']).std():.3f}")
    print(f"  EWMA 后噪声 σ:   {(Q2_ewma - data['Q2_true']).std():.3f}")

    T_deconv = temp_deconv(data["T_r_meas"], 30.0)
    err_raw = (data["T_r_meas"] - data["T_r_true"]).std()
    err_dec = (T_deconv - data["T_r_true"]).std()
    print(f"\n温度反卷积:")
    print(f"  原始 σ: {err_raw:.3f} ℃  |  反卷积后 σ: {err_dec:.3f} ℃")

    steady = rolling_steady_mask(data["Q0_meas"], 300, 0.03)
    print(f"\n稳态判定: {steady.sum()}/{len(steady)} 点为稳态"
          f" ({100 * steady.sum() / len(steady):.1f}%)")


if __name__ == "__main__":
    main()
