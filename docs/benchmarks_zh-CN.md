# 基准与评分协议

[English](benchmarks.md) · **简体中文**

[项目概览](../README_zh-CN.md) · [使用指南](quickstart_zh-CN.md) · [模型接入](models_zh-CN.md)

JevalKit 接入了八个 Jev 相关基准家族，提供十个 TSV 数据视图。
Image/Video 是 VisualDecisionBench 的子视图，不单独计为基准家族。

| 家族 | 评测数据 | 评分 |
|---|---|---|
| VisualDecisionBench | 完整发布版 `eval.jsonl`：1,696 个共享输入、9,143 个问题，其中图像 2,000 个、视频 7,143 个 | 硬标签准确率；软标签 NLL/Brier；有序 MAE/RPS；按模态、类型、领域统计 |
| JevBench | 231 个固定版本的公开任务：easy 48 个、original 72 个、hard 111 个 | 随包保存的官方任务评分器，以及校准、任务家族和释义变体报告 |
| Decision Index | 0.3 版公开套件，应用原生子集和排除规则后共 140,178 个请求 | 上游版本专用评分器；随机水平校正后的加权公开指数；43 个组成数据集 |
| Typed-Decisions | 固定版本 `all/test`：400 个共享输入、2,000 个问题 | 显式标签准确率；软参考分布 KL、Brier、交叉熵 |
| DecisionBench | 固定版本标准 `eval`：23,900 条样本，三种决策原语 | 全样本准确率；候选项对齐后的 NLL 与校准；保留原始有序数值 |
| Procedural Typed Decisions | 固定版本 `all/test`：18,000 个输入、57,016 个问题 | 精确规则判断；纯后验概率问题只计概率误差，不构造硬标签 |
| zh-decision-bench | 固定 v0.2 公开验证集：378 个输入、467 个问题 | 中文分类与评分；右闭合的 15 分箱参考 ECE |
| JevAlt-Bench | 固定版本英语、土耳其语、德语测试文件：3,116 个输入、8,434 个问题 | 标签准确率；硬标签与软标签概率指标；语言切片；等频分箱参考 ECE |

每个[基准目录](../jevalkit/datasets/)保存数据准备、评分实现及 `manifest.json`，其中固定了 TSV 哈希、数量和评分协议。
发布数据的下载方式见[使用指南](quickstart_zh-CN.md#download-data)。

本项目的 `JevBench` 特指 [`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench)，
版本为 `bb05a335bc809e61b20c0f745d25499a82b326fc`，与 Valen 公开比较采用的版本一致。
其他同名项目的任务不同，成绩不能混用。Noul 推理保留原始任务自定义的 true/false 判定标准。

Decision Index 保留问题 ID、选项及顺序、结构化输入、标准答案、分组、官方排除规则和固定套件的来源元信息。
没有标签的诊断问题仍保留在请求中，目标标记为 `{"unscored": true}`，不计入概率或准确率指标。
主指标必须使用官方加权指数；通用字段准确率仅作诊断，不能替代该指数。

推理输入不包含标准答案、推理说明、评测 ID 或来源元信息。
原生运行时使用作者发布的温度和上下文限制。模型无法表示的模态或问题类型记为 unsupported，
这些问题仍体现在覆盖率和准确率中。

概率必须为有限数值，位于 [0,1]，并覆盖全部候选项。
通用指标允许概率和存在不超过 0.01 的舍入误差：保存原始回答，仅在计算通用指标时归一化。
JevBench 官方评分器保留其发布版的严格与舍入容差，以及概率并列时按字典序取 argmax 的规则；官方汇总单独保存。
Decision Index 接入也处理官方校验器在 0.01 边界处的浮点误差，仅用临时副本完成校验；
固定版本的官方评分器始终接收原始回答，兼容处理的次数会记入报告。

NLL 的概率下限为 1e-12；多分类 Brier 为平方误差之和；ECE 使用 15 个等宽置信度分箱。
有序 RPS 为前 K−1 个等级上累计概率误差平方的均值。
没有显式分类标签的软目标不计入硬标签准确率。
若来源同时提供标签和概率分布，准确率与 ECE 使用标签，概率评分使用完整分布。
每个请求的同步计时包含预处理和原生推理，不含加载、预热及评分；同一多问题请求只计一次延迟。

数据集和提供方的许可及使用限制继续适用。工具采用发布的评测快照及其来源许可，见[第三方说明](../THIRD_PARTY.md)。

## 其他决策基准的协议

TSV 快照保留完整共享请求和来源中的候选项顺序，推理前会通过哈希检查拒绝已变更的数据。

- **Typed-Decisions：** 解码 JSON 状态，在同一请求中保留全部五个问题，同时保存显式标签和归一化的软参考分布。
  教师标注衡量的是与教师的一致程度；通用模型和经过工作流微调的模型可能有不同的数据接触情况，比较时需保留这一区别。
- **DecisionBench：** 使用上游 System One 映射：首个二元候选项对应 Noul `true`；Choice 保留原始候选 ID；
  Score 按原始有序数值排序，并保存 ID 逆映射。评分时恢复原始候选顺序和有序数值，用于并列处理及有序误差计算。
  不使用 compact 字段。参考 NLL 的概率下限为 1e-15，通用 NLL 仍为 1e-12。
- **Procedural Typed Decisions：** 保留全部任务、难度和问题。纯概率 Noul 目标保持为软分布，
  参与概率损失和后验 MAE，不生成硬准确率或经验 ECE。固定的 18 任务发布版与 RSI 较早的评测快照不同。
- **zh-decision-bench：** 使用 v0.2 发布数据，包括 323 条路由和 55 条业务样本；中文 Score 标签映射到来源给定的等级位置，
  并保留 `true,false` 的并列顺序。Hub 卡片的数量表已过时，实际数据为 378 条样本、467 个问题。
  这是公开验证集，评测时不重新拟合温度。
- **JevAlt-Bench：** 按文件归属确定测试划分，忽略部分记录中占位的 `train` 字段。
  公开输入的 `abstain` 标志对应一个显式的 `unknown` Choice 候选项，标准答案不决定模型输入。
  这一接口是对作者弃答标志的显式映射，并在结果中标记。字典形式的 Score 标准保留插入顺序和标签映射；
  上游额外问题字段序列化到指令中。与作者成绩比较前需匹配这些接口约定。

`metrics` 保存 JevalKit 通用指标。上述五个基准的 `official` 字段沿用历史命名，内容实际是
JevalKit 参考协议报告，以 `scorer=jevalkit-decision-protocol-v1` 标识，不表示运行了上游评测程序。
报告包含任务、领域、语言和难度切片，硬标签 NLL/Brier、软目标 KL/Brier，以及适用的参考 ECE。
JevBench 和 Decision Index 使用各自基准提供的评分器。

## 新增基准

创建 `jevalkit/datasets/<benchmark>/`，放入 `__init__.py` 和 `manifest.json`。
在包内实现 `prepare(root=None, source=None, **options)`，如需专用评分器，再实现
`evaluate(dataset, records, predictions, output)`。
在 `jevalkit/datasets/__init__.py` 注册包名、家族和模态后，CLI 和评测器会通过该条目调用实现。
使用 `_shared.commit_dataset` 生成经过验证的 TSV，并为来源映射和评分规则补充测试样例。
