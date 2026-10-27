# 使用指南

[English](quickstart.md) · **简体中文**

[项目概览](../README_zh-CN.md) · [模型接入](models_zh-CN.md) · [评分协议](benchmarks_zh-CN.md)

## 安装

```bash
python -m pip install -e '.[hf,data]'
export JEVALKIT_DATA_ROOT="$PWD/data"
```

命令行工具需要 Python 3.10 或更高版本。还需安装所选模型作者的依赖，部分原生 SDK 需要 Python 3.12 或更高版本。
已验证的版本组合见[运行环境](models_zh-CN.md#tested-runtimes)。

<a id="download-data"></a>

## 下载数据

从 Hugging Face 下载可直接评测的 TSV：

```bash
hf download Valen-Team/JevalKit_Datasets --repo-type dataset \
  --include "datasets/*.tsv" --local-dir "$JEVALKIT_DATA_ROOT"
```

如只需一个基准，可将 `datasets/*.tsv` 改为 `datasets/JevBench.tsv`。
JevalKit 从随包发布的[基准目录](../jevalkit/datasets/)读取数据格式、数量和哈希，
这些公开快照无需另备本地 manifest，也无需转换或构造数据集。

VisualDecisionBench 还需下载并解压原始媒体文件：

```bash
hf download Valen-Team/VisualDecisionBench assets.zip --repo-type dataset \
  --local-dir "$JEVALKIT_DATA_ROOT/media/VisualDecisionBench"
python -m zipfile -e "$JEVALKIT_DATA_ROOT/media/VisualDecisionBench/assets.zip" \
  "$JEVALKIT_DATA_ROOT/media/VisualDecisionBench"
```

Decision Index 需安装评分器并下载 0.3 版公开文本输入，0.4.1 榜单也沿用这部分数据：

```bash
python -m pip install -e '.[decision-index]'
hf download multimodalart/decision-index-suite-0.3 \
  selected-rows.jsonl.gz added-rows.jsonl.gz gsm8k-rows.jsonl.gz \
  excluded-questions.json manifest.json --repo-type dataset \
  --local-dir "$JEVALKIT_DATA_ROOT/raw/DecisionIndex-0.3"
```

使用官方转换器准备 0.4.1 的公开 Text/Vision TSV：

```bash
python -m jevalkit prepare DecisionIndex_Text \
  --source "$JEVALKIT_DATA_ROOT/raw/DecisionIndex-0.3"
python -m pip install 'pillow==12.3.0'
export DECISION_INDEX_FONT="/path/to/arialbd.ttf"
python -m jevalkit prepare DecisionIndex_Vision
```

Vision 准备过程会下载固定版本的原始数据并核对参考哈希。
通过 `hf auth login` 登录的账号需已获批 [Winoground 访问权限](https://huggingface.co/datasets/facebook/winoground)，
Mind2Web 还需要 Arial Bold 字体。评测两个公开轨道时使用
`--data DecisionIndex_Text DecisionIndex_Vision`。Valen 的原生图像预算会自动设置，详见[评分协议](benchmarks_zh-CN.md)。

## 下载模型并评测

```bash
hf download Valen-Team/Valen-0.8B \
  --local-dir "$JEVALKIT_DATA_ROOT/models/Valen-0.8B"

python -m jevalkit run --model Valen-0.8B --data JevBench \
  --option model="$JEVALKIT_DATA_ROOT/models/Valen-0.8B" \
  --output "$JEVALKIT_DATA_ROOT/results"
```

使用 `list models` 和 `list benchmarks` 查看注册名称。`--data` 后可指定多个基准。
`--limit 10` 用于冒烟测试；`--resume` 用于在数据、权重和配置均未变化时恢复中断的任务。
`verify BENCHMARK --media` 会检查完整 TSV 及其引用的媒体文件。

每次运行保存 `predictions.jsonl`、`manifest.json` 和 `metrics.json`。
使用 `python -m jevalkit score RUN_DIR` 可对已保存的预测重新评分，无需加载模型权重。

## API 与作者服务

在环境变量中配置服务凭据或地址，再选择对应的注册模型。例如，OpenAI Decisions 从 `OPENAI_API_KEY` 读取凭据：

```bash
python -m jevalkit run --model OpenAI-Decisions --data JevBench \
  --output "$JEVALKIT_DATA_ROOT/results"
```

服务地址变量和作者 SDK 路径见[模型说明](models_zh-CN.md)。
[本地服务配置](../examples/served-model.json)可在评测前启动、结束后关闭作者服务；
[托管 API 配置](../examples/hosted-apis.json)从环境变量读取凭据。

## 矩阵与并行运行

```bash
python -m jevalkit matrix examples/decision-benchmarks.json
```

矩阵中的每一行都可以指定独立的 `python`、权重 `options` 和输出路径。
各行通过隔离的子进程运行，某一行失败后，其余独立任务会继续执行。

将单次评测分配给多个 worker 时，所有 worker 使用相同的 `--world-size N`，
并分别设置唯一的 `--rank`，取值为 `0` 到 `N-1`。矩阵配置也支持 `world_size` 和 `rank`。
全部分片结束后，将结果合并到一个新的空目录：

```bash
python -m jevalkit merge /path/to/shard-000 /path/to/shard-001 \
  --model Pplx-Decider --data DecisionIndex \
  --output "$JEVALKIT_DATA_ROOT/results/merged/Pplx-Decider/DecisionIndex"
```

合并时会核对代码、运行环境、权重哈希及完整请求覆盖情况，再重新计算全局指标。
不完整的分片不会进入主榜。本地 HTTP 服务的 manifest 包含 worker 路径，应使用不分片的矩阵行。

## 结果

```bash
python -m jevalkit leaderboard "$JEVALKIT_DATA_ROOT/results" --output "$JEVALKIT_DATA_ROOT/leaderboard"
```

可[打开排行榜](https://huggingface.co/spaces/Valen-Team/JevalKit-Leaderboard)，也可查看本地生成的 Markdown、TSV 和 JSON。
比较成绩时需匹配数据哈希和评分协议；比较延迟还需匹配硬件条件。
同一模型和基准采用最近一次符合条件的完整结果，不按重复测量中的最高分选择。
