# 换热站管网水力工况在线测试与水力失调诊断

基于数据驱动的换热站管网水力工况诊断方案，包含水力计算、故障模拟、敏感度分析和不确定性量化等模块。

## 项目结构

```
code/
├── solver.py              # 水力计算求解器
├── train.py               # 故障诊断模型训练
├── faults.py              # 故障模拟与注入
├── inference.py           # 在线推理诊断
├── twin.py                # 数字孪生模型
├── sensitivity.py         # 敏感度分析
├── mc_uncertainty.py      # 蒙特卡洛不确定性分析
├── dynamic_filter.py      # 动态滤波
├── hii.py                 # 液压阻抗分析
├── cross_validate_analytic.py  # 交叉验证分析
├── plot_topology.py       # 管网拓扑可视化
├── demo.py                # 演示脚本
└── networks/              # 管网模型定义
    └── station_5br.py     # 五段环形管网

doc/
├── 技术说明书.md          # 技术文档
└── 附录/                  # 附录内容

plan/
├── solution-plan.md        # 解决方案
└── step-by-step-checklist.md  # 实施清单
```

## 主要功能

- **水力计算**：基于图论的管网水力工况求解
- **故障模拟**：阀门失调、管道堵塞、泵故障等模拟
- **敏感度分析**：关键参数对系统的影响评估
- **不确定性分析**：蒙特卡洛方法量化测量不确定性
- **动态滤波**：实时数据滤波与异常检测
- **数字孪生**：实时状态估计与预测

## 环境依赖

```bash
pip install numpy scipy pandas matplotlib networkx
```

## 使用示例

```bash
python code/demo.py          # 运行演示
python code/solver.py         # 水力计算
python code/train.py          # 训练诊断模型
```

## 相关文档

- [技术说明书](doc/技术说明书.md)
- [解决方案](plan/solution-plan.md)
