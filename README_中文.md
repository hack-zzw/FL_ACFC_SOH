# 电池SOH论文复现代码

本项目复现论文当前方法、主要对照、消融和外部验证的数值结果。默认流程为：读取随包数据和权重，重新预测，计算指标，生成论文结果图，再与固定参考结果核对。

## 开始运行

在本目录打开Anaconda Prompt，使用已配置的CUDA环境：

```text
conda activate pytorch
python main.py
```

结果统一写入 `results/generated/`。原始输入、权重和参考结果保持只读。可以从任意工作目录用绝对路径启动 `main.py`。

安装新环境时使用 `environment.yml`，创建后激活 `soh-paper-repro`。版本和硬件记录见 `configs/environment/`。已验证的环境是Windows、Python 3.13.9、PyTorch 2.9.0+cu128；完整主模型重放需要CUDA。尚未在全新安装的环境上验收。

## 三个入口

| 入口 | 用途 |
|---|---|
| `python main.py` | 重放全部论文模型，重新计算统计并绘图 |
| `python train.py <任务>` | 运行已支持的训练或特征重建任务 |
| `python plot.py` | 根据已完成的本次运行，单独重画结果图 |

常用选项：

```text
python main.py --verify-only
python main.py --suite main --strict-main
python main.py --no-figures
python main.py --help
python train.py controls --help
```

`--suite`可选 `main`、`native`、`readouts`、`controls`、`ablations`、`external`、`all`。单项运行只重放所选实验；默认 `all` 才汇总全部图表。`--strict-main`要求三个主分量的预测逐元素相同。默认也检查预测一致性，并为外部验证的CPU浮点误差计算指标误差界。

## 结果在哪里

| 路径 | 内容 |
|---|---|
| `results/generated/metrics/paper/` | 主方法、外部验证及绘图所需指标 |
| `results/generated/source_data/Table1.csv` | 主方法与七条对照配置的汇总指标 |
| `results/generated/source_data/` | 本次运行生成的图表源数据 |
| `results/generated/figures/` | 图2—4的PDF、SVG、PNG和TIFF |
| `results/generated/validation/` | 模型重放和14项统计核对记录 |
| `results/generated/run_summary.json` | 本次完整运行状态 |

图1和图5是概念/框架示意图，随包保存在 `results/reference/figures/`。量化结果图使用本次模型产生的预测，不从参考预测代入。固定支持观测和示例电池选择位于 `data/figure_selection/`。

## 目录与方法对应关系

| 目录或文件 | 职责 |
|---|---|
| `configs/` | 固定配置、数据划分、实验索引、文件校验信息 |
| `data/` | 基础特征、窗口特征、控制实验输入、外部协议特征 |
| `dataloader/` | 数据读取、标准化、可观测特征和历史特征 |
| `models/encoder.py` | ACfC编码器与SOH预测网络 |
| `models/federated.py` | 源模型训练与联邦聚合 |
| `models/adaptation.py` | 支持集预测头适配和基础评估 |
| `models/domain/inference.py` | 最终分域预测、残差修正与覆盖范围路由 |
| `models/controls/`、`models/readouts/` | 匹配输入/参数规模的控制与共享范围实验 |
| `models/external/` | NASA和Oxford适配与预测 |
| `utils/` | 统计、重放调度、绘图及命令支持 |
| `checkpoints/` | 各实验固定权重；主模型与控制模型路径均已压平 |
| `results/reference/` | 用于一致性核对的冻结结果和验收摘要 |
| `results/source_data/` | 随论文发布的固定图表源数据 |

## 固定评价口径

源池含303块电池，12个源客户端；目标评价含75块电池。主任务每块目标电池使用前20条SOH标注支持记录，随后58,214条查询记录用于评价。三个固定分量先分别完成适配和后处理，再对五个域统一等权平均预测。各分量指标的平均不等于该集成的指标。

主结果：汇总MAE为0.009347051198636335，汇总R²为0.963488486334106，电池宏平均R²为0.9182293056922992。MAE以SOH比值计，乘100后为SOH百分点；MAPE按百分数报告。分域汇总与汇总查询指标分别保留。完整协议和论文结果对应关系见 `configs/method_lock.json`、`configs/result_map.csv`。

## 已支持的训练任务

```text
python train.py source --seed 20260903
python train.py heads --domain NCA --seed 20260903
python train.py readouts --scope domain --seed 20260903
python train.py controls --all
python train.py comparisons --configuration fedavg --seed 20260903
python train.py external --retrain --dataset NASA --budget 20 --seed 20260903
python train.py external --retrain --dataset Oxford --budget 20 --seed 20260903
```

源侧设计矩阵与嵌入在训练时自动重建，写入 `results/generated/feature_cache/`。可以单独运行 `python train.py prepare`。`source`训练源编码器和初始域状态；`heads`训练已登记的初始条件头。完整主方法的全部分域残差与路由拟合尚未串联为一个从零训练任务；最终完整拟合状态、推理代码及全部论文数值重放均已提供。

## 数据来源和发布信息

数据来源、获取链接和处理后数据说明见 `data/README.md`。本包从处理后特征开始。再分发条件以各原始数据源许可为准。作者、代码许可、公开仓库及DOI待正式发布时在 `configs/release_metadata.json`中填写。第三方代码声明见 `THIRD_PARTY_NOTICES.md`，可用性声明草稿见 `AVAILABILITY.md`。
