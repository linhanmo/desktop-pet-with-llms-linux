# LlamaLite Technical Report（Windows）

## Abstract

LlamaLite 是一个以 Llama-family 为骨架的 Decoder-only Transformer 实现，面向中文与代码混合语料的预训练与后续微调场景。模型采用 Pre-Norm 的 RMSNorm、SwiGLU FFN、RoPE 位置编码，以及 GQA（Grouped-Query Attention）以降低 KV cache 与带宽开销。本文档以技术报告的形式描述 LlamaLite 的模型架构、关键算子与训练目标，并对齐 Qwen2.5 / Qwen2.5-Coder 技术报告中与 1.5B 规模相关的设计选择（例如 RoPE base frequency 的上调与 embedding tying）。

## 1. Tokenizer & Special Tokens

LlamaLite 使用与 Qwen 系列兼容的 BBPE（byte-level BPE）词表，并通过 `ChineseTokenizer` 进行封装。当前工程默认词表规模为 151665。

Qwen2.5 / Qwen2.5-Coder 报告中的词表规模为 151,643/151,646，并引入了更多控制 token（例如工具、FIM、repo/file 边界等）。LlamaLite 的词表规模略有差异，通常来自于额外预留的控制 token 或工程侧的兼容扩展。

对代码能力训练而言，FIM（Fill-in-the-Middle）相关控制 token 具有重要意义。Qwen2.5-Coder 给出的 file-level FIM 组织形式为：

`<|fim_prefix|>{code_pre}<|fim_suffix|>{code_suf}<|fim_middle|>{code_mid}<|endoftext|>`

Repo-level FIM 通过 `<|repo_name|>` 与 `<|file_sep|>` 进一步组织跨文件上下文。

## 2. Architecture

LlamaLite 是标准的 decoder-only Transformer（因果掩码），每层使用 Pre-Norm 结构，并包含注意力子层与 FFN 子层两段残差。对第 ℓ 层而言，计算流程可以概括为两段“归一化 → 子层变换 → 残差相加”：

1) 注意力分支：先对输入做 RMSNorm，再做自注意力，然后将注意力输出与原输入相加形成中间状态。

2) FFN 分支：对中间状态做 RMSNorm，再通过 SwiGLU FFN，然后将 FFN 输出与中间状态相加得到下一层输入。

### 2.1 RMSNorm（Pre-Norm）

RMSNorm 对每个 token 的 hidden 向量做基于均方根的缩放归一化（不减均值）。直观理解是：用该向量的“能量大小”来缩放向量，使其数值尺度更稳定；同时再乘上可学习的逐维缩放参数，并使用一个很小的 epsilon 避免除零。

### 2.2 Attention with GQA

注意力子层首先通过三组线性投影生成 Query、Key、Value。1.5B 配置中通常启用 QKV bias，即这三组投影都带有偏置项。

GQA（Grouped-Query Attention）的关键是：Query 头数量大于 KV 头数量，多个 Query 头共享同一组 KV 头。这样在推理时：

- KV cache 的存储量与带宽开销随 KV 头数而变化，因此可以显著降低 KV cache 的体积
- 仍保留更多 Query 头以维持表达能力

因果注意力采用标准的 scaled dot-product attention，并在 softmax 前加入因果 mask，保证每个位置只能关注到历史 token。工程实现优先使用 PyTorch 的 SDPA 路径，以便复用更高效的注意力内核。

### 2.3 RoPE（base frequency 上调）

RoPE（Rotary Positional Embedding）将位置编码以“旋转”的方式注入到每个 head 的向量维度对中，通常作用在 Query 与 Key 上。每个维度对对应一个与维度相关的频率，位置越靠后旋转角度越大。

参考 Qwen2.5 的 long-context 训练策略，可将 RoPE base frequency 从默认的 10000 上调到 1e6，以改善长上下文外推的稳定性；本工程的 1.5B 配置采用 1e6。

### 2.4 SwiGLU FFN

SwiGLU FFN 是一种门控前馈结构：输入经过两路线性投影，其中一路通过 SiLU 激活形成“门”，再与另一路逐元素相乘，最后再通过一层线性映射回隐藏维度。与传统 GELU FFN 相比，SwiGLU 通常在同等算力下表现更好，尤其在代码与多领域场景中更常见。

### 2.5 Output Head & Embedding Tying

输出 head 将最后一层隐状态映射到词表大小的 logits。启用 embedding tying 时，输出投影与 token embedding 共享同一组权重矩阵：这可以减少参数量，并在小到中等规模模型上常带来更稳健的训练表现；本工程在 1.5B 配置中启用该选项。

## 3. Training Objective（NTP + Masked Loss）

训练目标以 Next Token Prediction 为主：对序列中每个位置预测下一个 token，并用 softmax cross-entropy 作为基本损失。实现上通常会配合以下策略：

- Label smoothing：将目标分布从严格 one-hot 变为“正确 token 概率较高、其余 token 分到少量概率”的平滑分布，有助于提升泛化并降低过度自信
- Z-loss（logit regularization）：对每个位置的 log-sum-exp（等价于 softmax 分母的对数）加入惩罚项，抑制 logits 的整体漂移，提高数值稳定性
- Ignore index / masked tokens：对 padding、FIM 控制段、跨样本拼接边界等不需要计入损失的位置做掩码，只对有效预测位置统计损失，并按有效位置数量归一化

## 4. Reference Configurations（core/config.py）

| 配置 | vocab_size | ctx_len | hidden | layers | n_heads | n_kv_heads | head_dim | ffn_dim | tie | qkv_bias | rope_theta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 150M | 151665 | 1024 | 512 | 24 | 8 | 2 | 64 | - | - | ✗ | 10000 |
| 450M | 151665 | 1024 | 1024 | 24 | 16 | 4 | 64 | - | - | ✗ | 10000 |
| 1.5B | 151665 | 4096 | 1536 | 28 | 12 | 2 | 128 | 8960 | ✓ | ✓ | 1e6 |

## 5. System Mapping（Windows）

Windows 工程主要以纯 PyTorch 模块实现模型结构，训练与推理路径对应关系可概括为：

- `core/transformer.py`：顶层 Transformer 堆叠与前向流程组织
- `core/attention.py`：注意力结构（含 GQA、mask、SDPA 路径等）
- `core/layers.py`：RMSNorm、FFN（SwiGLU）等基础层组件
- `core/model.py`：语言模型封装（embedding、输出 head、embedding tying 等）
- `core/config.py`：模型规模配置（150M/450M/1.5B）

## 6. Codebase Map

```text
llm/
├── core/
│   ├── attention.py
│   ├── layers.py
│   ├── transformer.py
│   ├── model.py
│   └── config.py
├── chinese_tokenizer.py
└── qwen_tokenizer/
```

## 7. Safetensors & GGUF Quantization

本工程训练与导出的权重文件使用 `.safetensors`。如果需要将 HuggingFace 目录结构的模型转换为 GGUF 并进行 Q4_K_M 量化，可使用本仓库自带的量化脚本（内部包含 llama.cpp）。

### 7.1 Python 依赖

`llama.cpp/convert_hf_to_gguf.py` 需要在具备以下依赖的 Python 环境中运行：

- `transformers`
- `torch`
- `numpy`
- `sentencepiece`
- `safetensors`

建议在 `llm/` 下使用同一个 Python 环境执行安装与量化命令，避免出现 “No module named transformers” 之类的环境不一致问题。

### 7.2 量化命令

假设模型目录位于 `llm/checkpoints/<name>/`（包含 `config.json`、tokenizer 文件与 `model.safetensors`），示例：

```powershell
python .\quantize_checkpoints_to_gguf.py --model universal --qtype Q4_K_M
python .\quantize_checkpoints_to_gguf.py --model original  --qtype Q4_K_M
python .\quantize_checkpoints_to_gguf.py --model anime     --qtype Q4_K_M
```

输出 GGUF 默认写入对应的 `llm/checkpoints/` 目录。
