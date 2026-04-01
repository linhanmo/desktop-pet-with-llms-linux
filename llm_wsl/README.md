# LlamaLite Technical Report（Ubuntu22.04）

## Abstract

本目录是 LlamaLite 的 Ubuntu22.04 训练工程，目标是以最小的工程侵入将关键热点算子下沉至 CUDA/C++，并保持与 Qwen2.5 / Qwen2.5-Coder 在 1.5B 规模上高度一致的架构选择（Pre-Norm RMSNorm、SwiGLU、RoPE、GQA、embedding tying、QKV bias、RoPE base frequency 上调等）。本文档以技术报告形式描述：模型结构、训练目标，以及与 Ubuntu22.04 侧自定义算子/训练引擎的对应关系。

## 1. Tokenizer & Special Tokens

LlamaLite 通过 `ChineseTokenizer` 使用 Qwen 系列兼容 BBPE 词表：

- 词表规模：151665

Qwen2.5 / Qwen2.5-Coder 的公开报告中，dense 模型词表规模为 151,643/151,646，并额外引入了多类控制 token（工具、FIM、repo/file 边界等）。对代码能力训练，FIM 相关控制 token 与数据组织方式尤为关键：

- File-level FIM：
  - `<|fim_prefix|>{code_pre}<|fim_suffix|>{code_suf}<|fim_middle|>{code_mid}<|endoftext|>`
- Repo-level FIM：
  - 通过 `<|repo_name|>` 与 `<|file_sep|>` 组织跨文件上下文，并在最后一个文件内插入 FIM 片段

## 2. Architecture

### 2.1 Pre-Norm Decoder-only Transformer

每一层采用 Pre-Norm 的两段残差结构：先对输入做 RMSNorm，再做自注意力并与输入相加；随后对中间状态再做一次 RMSNorm，通过 FFN（SwiGLU）后再与中间状态相加。该结构能在较深网络中提供更稳定的优化行为，并与 Qwen2.5 / Qwen2.5-Coder 的实现习惯一致。

### 2.2 RMSNorm

RMSNorm 基于每个 token 的向量均方根进行缩放归一化（不减均值），再乘上可学习的逐维缩放参数，并使用 epsilon 提升数值稳定性。与 LayerNorm 相比，它通常有更低的计算与内存开销，也更常见于当前主流 decoder-only 架构。

### 2.3 GQA Attention

注意力子层通过三组线性投影生成 Query、Key、Value；当启用 QKV bias 时，这三组投影包含偏置项。

GQA 的核心是：Query 头数大于 KV 头数，多个 Query 头分组共享同一组 KV 头。这样可以在推理时显著降低 KV cache 的存储与带宽压力，同时保留较多的 Query 头以维持表达能力。

因果注意力采用标准的 scaled dot-product attention，并在 softmax 前加入因果 mask，保证每个位置只能关注到历史 token。

### 2.4 RoPE（base frequency 上调）

RoPE（Rotary Positional Embedding）将位置编码以“旋转”的方式注入到每个 head 的向量维度对中，通常作用在 Query 与 Key 上。每个维度对对应一个与维度相关的频率，位置越靠后旋转角度越大。

参考 Qwen2.5 的 long-context 训练策略，本工程的 1.5B 配置采用 rope base frequency 为 1e6（从默认的 10000 上调），以改善长上下文外推稳定性。

### 2.5 SwiGLU FFN

SwiGLU FFN 是门控前馈结构：输入经过两路线性投影，其中一路通过 SiLU 激活形成“门”，与另一路逐元素相乘后，再通过一层线性映射回隐藏维度。该结构在同等算力下通常优于传统 GELU FFN，并在代码与多领域语料下更常见。

### 2.6 Embedding Tying

当启用 tying 时，输出 head 与 token embedding 共享权重，通常在 1.5B 及更小规模上使用：
这可以减少参数量，并常带来更稳健的训练表现。

## 3. Training Objective（与 fused xentropy 对齐）

### 3.1 Softmax Cross-Entropy

基本目标是 next-token prediction：对每个位置的 logits 做 softmax 得到词表分布，然后取目标 token 的负对数概率作为逐 token 损失。

### 3.2 Label Smoothing

label smoothing 将严格 one-hot 的目标分布替换为平滑分布：正确 token 占主要权重，其余 token 分到少量均匀质量。这能抑制过度自信并提升泛化稳定性。

### 3.3 Z-loss

z-loss 会对每个位置的 log-sum-exp（等价于 softmax 分母的对数）加入惩罚项，以抑制 logits 的整体漂移并提升数值稳定性；实现上通常以一个可调系数控制其强度。

### 3.4 Ignore Index / Masked Tokens

对 FIM/padding/拼接边界等场景，常通过 mask（记作 m_i）或 `ignore_index` 实现只对有效位置计损失：
做法是只对有效位置累积逐 token 损失，并按有效位置数量归一化。

## 4. Reference Configurations（core/config.py）

| 配置 | vocab_size | ctx_len | hidden | layers | n_heads | n_kv_heads | head_dim | ffn_dim | tie | qkv_bias | rope_theta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 150M | 151665 | 1024 | 512 | 24 | 8 | 2 | 64 | - | - | ✗ | 10000 |
| 450M | 151665 | 1024 | 1024 | 24 | 16 | 4 | 64 | - | - | ✗ | 10000 |
| 1.5B | 151665 | 4096 | 1536 | 28 | 12 | 2 | 128 | 8960 | ✓ | ✓ | 1e6 |

## 5. System Mapping（Ubuntu22.04 / CUDA Extension）

Ubuntu22.04 工程侧的自定义扩展将模型结构中的若干热点算子固化为接口，以减少中间张量、提升带宽利用与降低 Python 调度开销：

- `rms_norm` / `rms_norm_backward`：对应 2.2
- `rope` / `rope_backward`：对应 2.4
- `xentropy_forward` / `xentropy_backward`：对应 3（指令微调侧的 fused softmax-cross-entropy）

## 6. CUDA 算子（Ubuntu22.04 Extension）

Ubuntu22.04 工程将若干高频算子实现为 PyTorch C++ Extension，以便在训练中减少中间张量、降低内存带宽压力，并降低 Python 调度开销。扩展按训练侧重点拆为三个模块，避免所有算子强耦合在同一个动态库中：

- 基础扩展：默认模块名为 `llm_wsl_extension`，实现位于 `csrc/`，导出 RMSNorm 与 RoPE 等通用算子
- 指令微调扩展：默认模块名为 `llm_wsl_extension_instruction`，实现位于 `csrc_instruction/`，除通用算子外还导出 fused softmax-cross-entropy
- 宠物聊天扩展：默认模块名为 `llm_wsl_extension_petchat`，实现位于 `csrc_petchat/`，用于 PetChat 训练/实验分支的算子集合（当前导出接口与指令微调扩展保持一致，便于按需复用与后续分化）

### 6.1 绑定与调用路径

每个扩展模块都通过 `bindings.cpp` 使用 pybind11 将 C++/CUDA 函数导出为 Python 可调用接口（例如 `rms_norm`、`rope`）。Python 侧有两条主要入口：

- 推理/训练通用算子：`custom_ops.py` 以 `torch.autograd.Function` 封装，并在模型 patch 时将 `core.model.RMSNorm` 替换为 `FastRMSNorm`（CUDA 路径优先、CPU 路径回退）
- 指令微调损失：`train_instruction.py` 内部定义 `_FusedXEntropy`，在 extension 可用且输入在 CUDA 上时优先走 fused xentropy，否则回退到 `torch.nn.functional.cross_entropy`
- 宠物聊天训练：`train_petchat.py` 默认通过 `custom_ops.get_extension()` 优先加载 `llm_wsl_extension_petchat`，从而在 CUDA 上启用 RMSNorm/RoPE 的融合路径；loss 当前使用 `torch.nn.functional.cross_entropy`（需要时可对齐指令微调扩展，切换到 `xentropy_forward/xentropy_backward`）

### 6.2 算子清单与语义

基础扩展（`csrc/`）：

- `rms_norm(x, weight, epsilon) -> y`：RMSNorm 前向。要求 `x` 最后一维为 hidden_dim，`weight` 为一维缩放向量；实现中对 `x` 做 contiguous，并按 token 展平后以“每 token 一个 block”计算均方根并缩放输出。
- `rms_norm_backward(grad_y, x, weight, epsilon) -> grad_x`：RMSNorm 反向，仅返回对输入的梯度；对 `weight` 的梯度在 Python 侧通过张量运算计算并回填（见 `custom_ops.py` 的 `RMSNormFunction.backward`）。
- `rope(x, cos, sin) -> y`：RoPE 前向，按 head_dim 的二维成对维度进行旋转；实现假设 `x` 形状为 `[batch, seq, n_heads, head_dim]`（或等价布局，满足最后三维分别对应 `seq_len / num_heads / head_dim`），并使用 `cos/sin` 查表对每个位置进行旋转。
- `rope_backward(grad_y, cos, sin) -> grad_x`：RoPE 反向，对旋转进行逆变换得到输入梯度。

指令微调扩展（`csrc_instruction/`）额外提供：

- `xentropy_forward(logits, labels, smoothing, z_loss, ignore_index) -> (losses, logsumexp)`：fused softmax-cross-entropy 前向（CUDA）。该算子面向指令微调/监督微调场景，将 `logsumexp`、NLL、label smoothing 与 z-loss 合并到一次 CUDA kernel 中完成，减少 Python 调度与中间张量。
  - 形状与 dtype：`logits` 为 `[N, V]`（float16/bfloat16/float32），`labels` 为 `[N]`（int64）。
  - 数值路径：对每一行 `logits[row, :]` 先做 `max` 归约，再计算 `sum(exp(logits - max))` 得到稳定的 `logsumexp`；在此基础上计算 `nll = logsumexp - logit_y`，并按 `smoothing` 混合 `smooth_loss = logsumexp - mean(logits)`；当 `z_loss != 0` 时额外加入 `z_loss * logsumexp^2` 惩罚项。
  - 输出：返回 `(losses, logsumexp)`，两者均为 float32 的 `[N]`。
  - ignore_index：当 `label == ignore_index` 或 label 越界（`<0` / `>=V`）时，该位置 `losses/logsumexp` 写 0，并在反向中将对应 `grad_logits` 整行置零。
- `xentropy_backward(grad_losses, logits, logsumexp, labels, smoothing, z_loss, ignore_index) -> grad_logits`：与 `xentropy_forward` 配套的 fused 反传（CUDA）。
  - 输入：`grad_losses` 为 `[N]`（float/float16/bfloat16 均可，内部会转 float32 计算），`logits/logsumexp/labels` 与前向一致。
  - 输出：`grad_logits` 形状为 `[N, V]`，dtype 与 `logits` 一致。
  - 计算要点：复用前向保存的 `logsumexp` 直接得到 `p = exp(logits - logsumexp)`；对目标类减去 `(1 - smoothing)`，对所有类减去 `smoothing / V`；当 `z_loss != 0` 时按 `zfac = 1 + 2 * z_loss * logsumexp` 对 softmax 项做缩放，从而与前向的 `z_loss * logsumexp^2` 保持一致。

宠物聊天扩展（`csrc_petchat/`）提供：

- `rms_norm` / `rms_norm_backward`：语义与基础扩展一致（见上）。
- `rope` / `rope_backward`：语义与基础扩展一致（见上）。
- `xentropy_forward` / `xentropy_backward`：语义与指令微调扩展一致（见上）；用于在 PetChat 训练中需要 label smoothing / z-loss / ignore_index 时，将 softmax-cross-entropy 的前向与反向合并为 fused kernel。
- `ThreadManager`（实验性）：与其他扩展一致（见 6.4），为 PetChat 侧的线程/任务调度实验提供 C++ 侧入口。

### 6.3 形状、dtype 与性能假设

这些 CUDA 算子在接口上做了最小约束以换取更简单的内核结构：

- 大多数输入在进入 C++/CUDA 前都会被 `contiguous()`，以保证内核可以按线性地址访问；这也意味着非 contiguous 张量会发生一次额外拷贝。
- RMSNorm 内核按“每 token 一个 block”做归约，hidden_dim 较大时依赖 shared memory 做 reduction。
- RoPE 内核按 half_dim 并行，每个线程处理一个维度对；cos/sin 表以 `[seq_len, head_dim/2]` 的展平布局访问。
- fused xentropy 假设 logits 为二维矩阵 `[N, V]`，并以 256 线程对词表维度做 block-level reduction，显式计算 max 与 sum(exp) 以获得稳定的 logsumexp。

### 6.4 线程管理（实验性）

扩展中还提供了一个 C++ `ThreadManager` 类，用于在 CPU 侧维护简单的线程池接口（任务队列与批处理示例）。当前训练脚本会按配置尝试初始化该对象，但其主要用途是为工程提供一个可扩展的“C++ 侧线程调度”骨架，而非模型数值算子的一部分。

## 7. Codebase Map

```text
llm_wsl/
├── core/
├── csrc/
├── csrc_instruction/
├── csrc_petchat/
├── train.py
├── train_instruction.py
├── train_petchat.py
└── custom_ops.py
```

## 8. Safetensors & GGUF Quantization

本工程训练与导出的权重文件使用 `.safetensors`。如果需要将 HuggingFace 目录结构的模型转换为 GGUF 并进行 Q4_K_M 量化，可使用本仓库自带的量化脚本（内部包含 llama.cpp）。

### 8.1 Python 依赖

`llama.cpp/convert_hf_to_gguf.py` 需要在具备以下依赖的 Python 环境中运行：

- `transformers`
- `torch`
- `numpy`
- `sentencepiece`
- `safetensors`

### 8.2 量化命令

```bash
python ./quantize_checkpoints_to_gguf.py --model universal --qtype Q4_K_M
python ./quantize_checkpoints_to_gguf.py --model original  --qtype Q4_K_M
python ./quantize_checkpoints_to_gguf.py --model anime     --qtype Q4_K_M
```
