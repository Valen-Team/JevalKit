# 模型接入

[English](models.md) · **简体中文**

[项目概览](../README_zh-CN.md) · [使用指南](quickstart_zh-CN.md) · [评分协议](benchmarks_zh-CN.md)

当前注册了 20 个模型家族，同一家族的尺寸变体不重复计数。
原生适配器调用作者发布的实现，HTTP 适配器连接作者服务或托管 API。
注册只表示接入了接口，实际权重评测情况需查看排行榜及对应 manifest。

| 名称 | 家族 | 接口 | 依赖与配置 | 来源 |
|---|---|---|---|---|
| Valen-0.8B | Valen | 原生接口 | jevalkit[hf] | [源码](https://github.com/Valen-Team/Valen) |
| Intern-Decision-0.8B | Intern-Decision | 原生接口 | Python 3.12+; 作者的 requirements.txt | [源码](https://github.com/InternLM/Intern-Decision) |
| Valen-2B | Valen | 原生接口 | jevalkit[hf] | [源码](https://github.com/Valen-Team/Valen) |
| Intern-Decision-2B | Intern-Decision | 原生接口 | Python 3.12+; 作者的 requirements.txt | [源码](https://github.com/InternLM/Intern-Decision) |
| Valen-4B | Valen | 原生接口 | jevalkit[hf] | [源码](https://github.com/Valen-Team/Valen) |
| Intern-Decision-4B | Intern-Decision | 原生接口 | Python 3.12+; 作者的 requirements.txt | [源码](https://github.com/InternLM/Intern-Decision) |
| Laya | Laya | 原生接口 | laya | [源码](https://github.com/NandhaKishorM/laya) |
| Liquid-d1-3B | Liquid-d1 | 原生接口 | transformers>=5.14; 作者依赖 | [源码](https://huggingface.co/LiquidAI/d1-3B) |
| Decision1-Kai | vLLM-Decision1 | 原生接口 | jevalkit[hf] | [源码](https://huggingface.co/vllm-sr/Decision-1.0-Kai-0.6B) |
| Clef | Clef | 原生接口 | 作者的 requirements.txt | [源码](https://huggingface.co/Cloudflare/clef) |
| Clef-Flash | Clef | 原生接口 | 作者的 requirements.txt | [源码](https://huggingface.co/Cloudflare/clef-flash) |
| Pplx-Decider | Pplx-Decider | 原生接口 | Python 3.12+; 作者的 requirements.txt | [源码](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) |
| JevK5 | JevK5 | 原生接口 | jevk5 | [源码](https://huggingface.co/alibiserikbay/JevK5) |
| StartLux-0.8B | StartLux | 原生接口 | 作者的 requirements.txt | [源码](https://huggingface.co/startlux-models/StartLux-Decision-v1-0.8B) |
| StartLux-v1.1-{0.8B,2B,4B,9B,27B}-Instant | StartLux | HTTP | startlux-decide；STARTLUX_ENDPOINT | [源码](https://github.com/StartLuxLabs/StartLux-Decision) |
| Crivo-150M | Crivo | 原生接口 | jevalkit[hf,decision-index] | [源码](https://huggingface.co/tardellirs/crivo-150m) |
| RSI-Jev-v4 | RSI-Jev | 原生接口 | 作者的 rsijev 包； `--option source=<checkout>` | [源码](https://github.com/Shanghua-Gao/RSI-Jev) |
| Vjev-4B | Vjev | 原生接口 | 作者的 requirements.txt | [源码](https://huggingface.co/yah01/vjev-vision) |
| QEV | QEV | 原生接口 | 作者的 veyra 包； `--option source=<checkout>/src` | [源码](https://github.com/ken-jo/qev) |
| OmniJev | OmniJev | 原生接口 | 作者的 mso 包； `--option source=<checkout>` `--option base_model=<Qwen3.5 weights>` | [源码](https://github.com/tinnel123666888/OmniJev) |
| Visual-Jev | Visual-Jev | 原生接口 | 作者的 vdm 包； `--option source=<checkout>/code` `--option base_model=<Qwen3-VL weights>` | [源码](https://github.com/guanxuyu-sv/Visual-Jev) |
| Jev | Jev | HTTP | Jev API 访问权限及 JEV_ENDPOINT (完整 /v1/systemone 地址) | [源码](https://typesafe.ai) |
| OpenAI-Decisions | OpenAI-Decisions | HTTP | OPENAI_API_KEY | [源码](https://developers.openai.com/api/docs/guides/decisions) |
| Kev | Kev | HTTP | 作者的 kev.serve 服务； KEV_ENDPOINT (完整 /v1/systemone 地址) | [源码](https://github.com/jaredpalmer/kev) |
| H2O-Lightning | H2O-Lightning | HTTP | 作者的 serve.sh 服务； H2O_ENDPOINT (完整 /v1/systemone 地址) | [源码](https://huggingface.co/h2oai/h2o-lightning-4b) |
| OpenJev | OpenJev | HTTP | 作者的 helper/shim.py 服务； OPENJEV_ENDPOINT (完整 /v1/systemone 地址) | [源码](https://huggingface.co/openjev/openjev) |

已有权重可通过 `--option model=/path/to/checkpoint` 指定，外部作者包可通过
`--option source=/path/to/checkout` 指定。OmniJev 和 Visual Jev 还需要 `--option base_model=/path/to/base`。

Valen 原生支持文本、图像和视频。Intern-Decision、RSI-Jev、Vjev、QEV、Liquid d1 和 StartLux
遵循作者的文本与图像接口。Visual Jev 要求恰好一张图像，OmniJev 要求视觉输入。
Crivo 发布的 Decision Index 引擎仅支持文本 Choice/Noul。不支持的媒体输入会明确报告，不会静默移除。

Intern-Decision 的符号表最多支持 62 个候选项；QEV 最多支持 16 个候选项和一张图像。
超限输入标记为 unsupported，保存的基准仍保留全部原始选项。Clef 使用原生编码器和输出头，
在推理前检查完整输入长度，避免便捷封装静默截断状态文本。
StartLux v1 可通过 `--option allow_slow=true --option graphs=false` 显式启用作者的慢速 Torch 路径；
元信息记录内核可用性、计算图使用情况和校准设置。

Decision1-Kai 和 StartLux 最多支持十个 Score 等级。Vjev 和 OmniJev 的 Noul 输出头只接收指令，
因此自定义 true/false 判定标准会保留在指令中。只接受文本字段的 SDK 会接收结构化描述的紧凑 JSON；
Choice 描述缺省时使用候选项 ID。已知的历史输入约定错误保留在详细报告中，并从主榜排除。

OmniJev 额外的弃答概率保留在 `native_answers` 中。验证完整原生分布后，
闭集基准使用给定候选项上的条件概率评分；完全弃答的结果记为 refusal。运行元信息记录这一处理方式。

Kev、H2O-Lightning 和 OpenJev 使用作者发布的 Jev 兼容服务协议。
将 KEV_ENDPOINT、H2O_ENDPOINT 或 OPENJEV_ENDPOINT 设置为完整的 `/v1/systemone` 地址。
Jev 使用 JEV_ENDPOINT 和 TYPESAFE_API_KEY，也可通过 `--option token_env=JEV_API_KEY` 指定密钥环境变量。
当前注册版本为 `jev-1.14.0`，每次响应都会核验实际版本。
图像通过 TypeSafe 原生 assets 字段及必需的图像请求头传输。
`--option long_context=true` 可启用作者提供的 262k 上下文模式，两项功能开关均记入运行记录。
OpenAI 使用 OPENAI_API_KEY。
凭据从环境变量读取，不写入结果 manifest。
配置后可用 `python -m jevalkit matrix examples/hosted-apis.json` 运行[托管 API 矩阵](../examples/hosted-apis.json)，
对两个服务执行完整 JevBench。配置文件不保存凭据，实际 API 评测需提供各服务的访问凭据。

OpenJev 的单张截图通过作者的 `state.image` 字段发送，H2O-Lightning 使用顶层 `images` 字段。
协议与本地 HTTP 测试不代表真实服务权重的评测，也不会生成排行榜成绩。

H2O-Lightning 的 `model.safetensors` 是主干权重，`label_rows.safetensors` 是作者独立的决策读出层。
启动其 `serve.sh` 前，应为 vLLM 本地加载器生成只选择主干文件的标准 HF 索引：

```bash
python -m jevalkit.models.h2o_lightning /path/to/h2o-lightning-4b model.safetensors
bash /path/to/h2o-lightning-4b/serve.sh
```

服务桥接层单独读取原始标签行，并作用于池化后的隐藏状态。
索引记录全部主干张量名，准备过程中会拒绝覆盖选择不同文件的已有索引。
部署指纹包含两个原始权重文件和生成的索引。

OpenJev 应使用固定版本推荐的 `READOUT_TARGETED=1` 路径，读取每个候选 token 的分数。
旧版默认路径会为未出现在词表 top-K 中的候选项填入固定对数概率。
已验证的参考部署采用作者公布的 `READOUT_T=0.85`、`READOUT_NOUL_T=1.829074`、
`READOUT_NOUL_BIAS=0`、`READOUT_INSTR_STYLE=pyrepr` 和 FP8 服务配置。
这些常量未在 JevalKit 测试集上拟合，服务记录保存全部选定设置。

Intern-Decision 每个共享请求最多接受 16 个问题，Decision1-Kai 要求状态文本非空。
超出发布版 SDK 限制的请求记为 unsupported，原始请求完整保留。

StartLux v1.1 Instant 使用作者的 `startlux-decide serve` 服务，通过 `STARTLUX_ENDPOINT`
配置完整的 `/v1/systemone` 地址，鉴权可用 `STARTLUX_API_KEY`。服务必须加载所选型号的权重；
自定义服务名称需设置 `--option expected_response_model=NAME`。请求固定使用
`think_mode=never`、`allow_none=false`、`none_check=true`，并写入运行记录。
权重和作者源码指纹可通过现有的[托管服务示例](../examples/served-model.json)保存。

<a id="tested-runtimes"></a>

## 已验证的运行环境

| 运行环境 | Python | PyTorch / CUDA | Transformers |
|---|---|---|---|
| Valen | 3.10 | 2.6.0 / 12.4 | 5.4.0 |
| 作者原生 SDK | 3.12 | 2.10.0 / 12.8 | 5.14.1 |
| H2O-Lightning / OpenJev 服务 | 独立的 vLLM 0.30.0 环境 | 2.13.0 / 13.0 | 5.19.0 |

作者依赖不兼容时应使用独立环境。矩阵行可指定绝对 `python` 路径，SDK 源码可通过 `--option source=PATH` 指定。
每份结果 manifest 都会记录实际库版本、权重和原生源码指纹。运行方法见[使用指南](quickstart_zh-CN.md)。

## 新增模型

创建 `jevalkit/models/<model>.py`，实现 `base.py` 中的接口，并在 `jevalkit/config.py` 注册导入路径。
尺寸变体可共用实现。作者包应在初始化时导入，避免其他模型因缺少该 SDK 而无法使用。
调用约定相同的实现可复用 `_http.py` 或 `_hf.py`。
使用 `jeval check --model <name> --output check.json` 检查接口，并运行相关适配器测试。
