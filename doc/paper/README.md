# LaTeX 论文编译说明

## 文件结构

```
doc/paper/
├── main.tex          # 论文主文件（自包含，含 thebibliography）
├── fig/              # 插图目录（9 张 png，已从 ../../fig 复制）
│   ├── network.png
│   ├── sensitivity_heatmap.png
│   ├── sensor_selection.png
│   ├── eda_class_dist.png
│   ├── eda_correlation.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_rf.png (未在正文引用，可选删除)
│   ├── shap_summary.png
│   ├── mc_hist.png
│   └── dynamic_response.png
└── README.md
```

## Overleaf 编译

1. 在 Overleaf 创建新项目（Blank Project）。
2. 将 `doc/paper/` 整个目录的内容上传（可打包 zip 上传）。
3. 项目设置 → Compiler → **XeLaTeX**。
4. 点击 Recompile。

## 依赖宏包

模板用的是标准科研宏包，Overleaf TeX Live 默认都带：
`ctex`（中文）、`graphicx`、`booktabs`、`tabularx`、`amsmath/amssymb`、`float`、`caption`、`subcaption`、`siunitx`、`enumitem`、`authblk`、`hyperref`、`xcolor`、`geometry`。

## 本地编译（如需）

```bash
cd doc/paper
xelatex main.tex
xelatex main.tex   # 再跑一次以解决交叉引用
```

## 结构概览

- 摘要（含关键词）
- §1 引言（研究背景、文献综述、贡献点）
- §2 测试目标与系统概览
- §3 关键参数测量与传感器选型
- §4 测点布置优化（灵敏度矩阵 + 贪心 SVD）
- §5 支路流量测量方案比选（加权评分表）
- §6 水力失调诊断算法（HII + XGBoost + SHAP）
- §7 不确定度分析与动态响应处理（MC + EWMA + 反卷积 + 交叉验证）
- §8 结论与展望
- 参考文献（31 条）
- 附录 A：27 路测点完整表 + 间接计算量 + 系统连接
- 附录 B：方案比选候选表 + 决策逻辑 + 造价估算 + 风险清单
- 附录 C：9 项不确定度来源 + MC 原理 + 判定规则伪代码 + 误差链路
- 附录 D：代码仓库结构

## 字数估算

正文约 4500 中文字，附录 A--D 完整内联（所有表格数据直接可见），图 9 张，表 10+ 张，公式 7 式，参考文献 31 条。
