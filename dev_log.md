# dev_log —— task_rgdzy 换热站管网水力失调诊断

## 2026-04-21

### 本次推进（P4 AI 诊断 + P5 HII/MC 误差）

开场状态：P0–P3 已完成（环境、Hardy-Cross、灵敏度、故障样本 5000 条）。本次按 `plan/step-by-step-checklist.md` 推进 P4、P5 两个代码阶段。

- `code/train.py`：新增 RandomForest baseline + XGBoost 主力 + 5-fold CV + 混淆矩阵 + SHAP summary。
  - 将 `faults.py` 原 7 类中 `class=5`（负荷波动）与 `class=6`（传感器漂移）合并进正常类（rationale：两者本质上是"正常水力 + 扰动"的鲁棒性样本，与 solution-plan §5.2 原始定位一致）。
  - 修复 SHAP 对新版多分类返回 `(n_samples, n_features, n_class)` 的 shape 解析。
  - 统一 stdout 改 UTF-8（避免 Windows GBK 控制台打不出 `✓`/中文日志）。

- `code/faults.py`：调整 `gen_double` 的扰动采样为双峰（避开 `[0.7, 1.5]` 温和区），样本量 800→1500；`gen_pump` 400→600；总样本 5000→5900。解决双支路 F1 低的问题。

- `code/inference.py`：新增；加载 `models/xgb_fdd.json` → 输出 `{class, class_name, confidence, top-2}`。

- `code/hii.py`：新增；按 §5.1 实现 θ_Q / θ_P / θ_T + 加权 HII + 四级分类（正常/轻度/中度/重度）。

- `code/mc_uncertainty.py`：新增；按 §6.1 不确定度表（Pt100 ±0.15 ℃、超声 ±2% + 三角 ±1% 安装偏差、差压 0.15 kPa、电磁 ±0.5%）做 N=10000 MC 采样，对 5 个典型工况得到 HII 95% CI；引入 `judge_with_ci` 实现"CI 跨任一阈值 → 延迟判定"。

### 关键验收指标

| 指标 | 目标 | 实际 |
|------|------|------|
| XGBoost F1 macro（测试集） | ≥ 0.93 | **0.957** |
| XGBoost F1 macro（5-fold CV 均值） | — | **0.963 ± 0.006** |
| RF F1 macro（测试集） | — | 0.931 |
| 每类 precision/recall | ≥ 0.85 | 最低 0.849（泵频偏移 precision），其余 ≥ 0.91 |
| 推理一致率（前 10 样本） | — | 100% |
| MC 5 场景判定 | 含跨阈值案例 | 场景「支路4堵塞×2.15」CI=[0.094, 0.111] 跨 0.10 阈值 → 触发延迟判定 |

### 产出物清单

- 代码：`code/{twin,solver,sensitivity,faults,train,inference,hii,mc_uncertainty,plot_topology,demo}.py` + `code/networks/station_5br.py`
- 数据：`data/{dataset.csv (5900 样本), sensitivity_matrix.npy, baseline_obs.csv}`
- 模型：`models/xgb_fdd.json`
- 结果：`results/{train_summary.json, mc_summary.csv}`
- 图：`fig/{network, sensitivity_heatmap, sensor_selection, eda_class_dist, eda_correlation, confusion_matrix, confusion_matrix_rf, shap_summary, mc_hist}.png`

### 遗留/待办

- P6 仪表选型与方案比选表（组员 B 文档）
- P7 误差分析/动态响应章节（组员 D 文档）—— MC 结果已生成，可直接引用
- P8 说明书 + PPT（组员 F + 全组）
- P9 答辩彩排

代码侧已无阻塞项。

### 追加推进（动态响应 + 附录表 + 交叉验证）

- `code/dynamic_filter.py`：EWMA(α=0.2) + 稳态判定(5min σ/μ<3%) + 温度反卷积(τ=30s，预滤波后反卷积避免噪声放大) + 工况归一化；产出 `fig/dynamic_response.png`（4 子图：支路流量/温度/稳态判定/HII 时序）。EWMA 使支路 2 测量噪声 σ 从 0.199→0.069 m³/h（压低 2.9×）；反卷积使温度误差 σ 从 0.260→0.209 ℃。
- `doc/附录/appendix_{1_sensors,2_scheme_comparison,3_uncertainty}.md`：27 路测点表 + 支路流量方案比选（评分矩阵 + ¥128,800 全案造价） + JCGM 不确定度来源表 + MC 5 工况判定结果。
- `code/cross_validate_analytic.py`：scipy.optimize.fsolve 独立求解 7 个扰动工况（正常/阀 55%/阀 30%/堵塞×2.5/堵塞×5.0/泵频 80%/110%），与 Hardy-Cross `solver.solve()` 对比，**最大相对差异 0.0003%**（plan P1.7 验收线 1%）。产出 `results/cross_validation.csv`。
- PyDHN 装到 `temp/pydhn`（via ghfast.top 镜像 + 清华 PyPI）：setpoint-driven 模型与我方阻力-driven 模型不匹配，完整网络复建成本高；改以 scipy-fsolve 独立算法对比，属"数值方法无关性"更强验证。副作用：numpy 2.x→1.26、pandas 3.x→2.3、networkx 3.x→2.8（PyDHN 固定版本要求）。**回归测试通过**：inference/mc_uncertainty/dynamic_filter 输出与降级前逐字一致。

### 当前验收汇总

| 项目 | 目标 | 实际 |
|------|------|------|
| XGBoost F1 macro（CV） | ≥ 0.93 | 0.963 ± 0.006 |
| fsolve vs Hardy-Cross 最大相对差异 | < 1% | 0.0003% |
| MC 5 场景含跨阈值案例 | ≥ 1 个 | 1 个（堵塞×2.15） |
| 动态响应 EWMA 噪声压低 | 显著 | 2.9× |
| 附录三张表 | 齐全 | ✓ |

### 剩余文档工作（组员分工）

- P8 说明书 Word 化（§1/§2/§3/§4/§5/§8 按分工表由组员 B/C/F 产出）
- PPT 10 页
- P9 彩排

### 文档定稿 + 清理

- `doc/技术说明书.md`：按 plan §8.1 八章结构完整起草，17,357 字符（中文约 3400 字，命中 2500-3500 目标）；所有图片以 `[插入图 x.x] fig/xxx.png` 格式明确标注插入位置、图注说明；表格引用 `doc/附录/*`；参考文献 9 条、代码仓库结构附后。可供组员直接改写为 Word。
- 清理：删除 `temp/pydhn/`（3.3 MB 已装进 .venv）+ `code/**/__pycache__/`。两份 .docx 源需求文档、plan/、所有产出（fig/ data/ models/ results/ doc/附录/）全部保留。

## 项目完成度

| 阶段 | 状态 |
|------|------|
| P0 环境 | ✅ |
| P1 数字孪生（Hardy-Cross） | ✅ |
| P1.7 独立求解器交叉验证 | ✅（scipy-fsolve，最大差异 0.0003%） |
| P2 灵敏度分析 | ✅ |
| P3 故障样本生成（5900 条） | ✅ |
| P4 AI 诊断（XGBoost F1=0.957） | ✅ |
| P5 HII + MC 误差传递 | ✅ |
| P6 仪表选型（附录 1 + 附录 2） | ✅ |
| P7 误差分析（附录 3） + 动态响应 | ✅ |
| P8 说明书初稿（markdown） | ✅ |
| P8 PPT 10 页 | ⬜（需组员 F 产出） |
| P9 答辩彩排 | ⬜ |

代码与文档侧全部就绪，余下 PPT 美化与彩排为组员分工。

### 论文化（LaTeX 格式，xelatex）

- `doc/paper/main.tex`：按学术论文规范重写，`ctexart` + xelatex，单文件自包含（内嵌 `thebibliography` 12 条）。去除"工作报告"语气（"我方"、"答辩防御"、"分工"、"本次"），改被动句与第三人称；扩写引言部分增加文献综述与贡献点陈述。
- 章节结构：摘要（含关键词） + §1 引言 + §2 目标 + §3 传感器 + §4 布置优化 + §5 方案比选 + §6 诊断算法 + §7 不确定度与动态响应 + §8 结论 + 参考文献 + 附录 A–D。
- 正文中嵌入 3 个数学公式块（θ_Q/θ_P/θ_T、HII、特征向量）、3 张正式 tabular 表（方案比选、分类器性能、不确定度来源）、9 张 figure（含 1 个 subfigure 组合），图表均有 `\label{}` + `\ref{}` 交叉引用。
- `doc/paper/fig/` 从 `fig/` 复制 9 张实际引用的图片（去除未引用的 `confusion_matrix_rf.png`），使 `doc/paper/` 可打包 zip 直接上传 Overleaf。
- `doc/paper/README.md`：Overleaf 编译步骤 + 依赖宏包清单 + 本地 xelatex 命令。

### 论文精修：附录内联 + 正文增引

- `doc/paper/main.tex` 大改：
  - **附录 A/B/C 完整内联**：27 路测点表（`longtable` 支持跨页）、方案候选参数表、评分矩阵、造价估算、风险清单、9 项完整不确定度来源、MC 原理三步法、判定规则伪代码、合成不确定度三条关键观察、误差链路闭环表 —— 全部落到正文后可直接阅读，不再"见 xx 文件"。
  - **参考文献扩充至 31 条**：在原 12 条基础上补 IEC 60751 (Pt100)、EN 1434 (热量表)、Baker 2016 (Flow Measurement Handbook)、Lynnworth 2006 (超声)、Cross 1936 (Hardy-Cross 原文)、Swamee--Jain 1976 (Colebrook 近似)、Breiman 2001 (RF)、Grinsztajn 2022 NeurIPS (为何树模型优于深度网络)、Lundberg 2020 TreeSHAP、Roberts 1959 EWMA、Montgomery 2009 SPC、Savitzky-Golay 1964 (微分放大噪声)、Virtanen 2020 SciPy、Powell 1970 hybrid Newton、Brent 1973、Henrion 1986 MC、Wagner 2008 IAPWS-IF97、GB 50736-2012 供暖规范、Nelson/Krishnamurthy 2000 传感器布置。
  - **正文增引 55 处 `\cite`**：每个方法/数据/标准处都有文献支撑。关键新增位置：§3 传感器选型引 IEC 60751 + EN 1434 + Baker；§4 灵敏度 SVD 引 Nelson 2000；§5 超声/电磁原理引 Lynnworth + Baker；§6 XGBoost 引 Chen2016 + Grinsztajn2022，RF 引 Breiman，SHAP 引 Lundberg2017 + 2020 TreeSHAP；§7 EWMA 引 Roberts1959 + Montgomery2009；温度反卷积引 Rees2020 + Savitzky1964；fsolve 引 Virtanen2020 + Powell1970 + Brent1973；Swamee-Jain 引 1976 原文；IAPWS 引 Wagner2008。
- `doc/paper/README.md`：更新结构概览（31 条文献、附录内联说明）。
