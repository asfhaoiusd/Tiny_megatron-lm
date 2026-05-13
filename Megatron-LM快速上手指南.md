# Megatron-LM / Megatron Core 新手快速上手指南

面向第一次接触本仓库的同学：先跑通一个最小示例，再理解目录与数据流程。官方文档以 NVIDIA 为准：[Megatron Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)。

---

## 1. 先搞清楚：这个仓库里有什么


| 名称                | 是什么                                                                | 适合谁                |
| ----------------- | ------------------------------------------------------------------ | ------------------ |
| **Megatron Core** | 可组合的 GPU 训练库（并行、Transformer 模块、checkpoint 等），包名一般为 `megatron-core` | 写训练框架、改模型结构、学分布式原理 |
| **Megatron-LM**   | 在 Core 之上的一套参考实现：预置训练脚本、`examples/` 等                              | 做研究、对照脚本改参数、快速实验   |


你本机目录里建议保持：**虚拟环境**（例如 `magetron`）与 `**Megatron-LM` 源码**并列；训练与示例都在源码树里执行。

---

## 2. 环境要求（不满足会很难跑）

- **GPU**：官方推荐 NVIDIA Turing 及以后架构；多卡训练依赖 NCCL。FP8 等特性需要 Hopper / Ada / Blackwell 等支持 FP8 的卡。
- **Python**：本仓库 `pyproject.toml` 要求 **≥ 3.12**（与 README 中「即将弃用 3.10」方向一致）。你当前 `magetron` 为 3.12.3，符合要求。
- **PyTorch**：**≥ 2.6.0**，且需与 CUDA 版本匹配（你环境里若已装 `torch`+`torchvision`，先 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 自检）。

没有 NVIDIA GPU 时，多数官方示例无法按原样运行（依赖 CUDA + NCCL）。

---

## 3. 安装（本地源码 + 可编辑安装）

在 PowerShell 中（路径按你本机调整）：

```powershell
# 1) 进入你的虚拟环境
F:\科研学习\python实践\magetronLM\magetron\Scripts\Activate.ps1

# 2) 进入仓库根目录
cd F:\科研学习\python实践\magetronLM\Megatron-LM

# 3) 安装 Megatron Core（可编辑模式，改代码立刻生效）
pip install -e .

# 可选：训练常用依赖（W&B、SentencePiece、transformers 等）
pip install -e ".[training]"
```

若从源码编译时内存吃紧，可先设小并行度再装（与官方说明一致）：

```powershell
$env:MAX_JOBS = "4"
pip install -e "."
```

更省事的方式是只装 PyPI 包（不含本仓库全部 `examples`）：`pip install "megatron-core[training]"`。要学习本仓库里的脚本，仍建议保留 `Megatron-LM` 目录并用 `pip install -e .`。

---

## 4. 第一个能跑的训练循环（官方「最小示例」）

仓库内脚本：`examples/run_simple_mcore_train_loop.py`。

**重要：** 该示例在代码里写死了 `tensor_model_parallel_size=2`，表示 **张量并行占 2 张 GPU**，因此需要 **至少 2 张 GPU** 且用 `torchrun` 起两个进程：

```powershell
cd F:\科研学习\python实践\magetronLM\Megatron-LM
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

预期现象：打印若干 `Iteration ... Losses reduced: ...`，最后在目录下生成 `ckpt` 分布式 checkpoint，并打印 `Successfully loaded the model`。

**只有 1 张卡时：** 该文件不能直接照抄运行。需要自行把 `initialize_distributed(tensor_model_parallel_size=2, ...)` 改为 `1`，并相应使用 `torchrun --nproc_per_node=1`（属于「改示例学原理」，不在官方一行命令保证范围内）。

更大规模示例（多机多卡、FP8 等）见 `examples/`，例如 LLaMA 相关脚本（对硬件要求更高）：

```text
examples/llama/train_llama3_8b_h100_fp8.sh
```

---

## 5. 想训自己的数据：三步概念

Megatron 训练侧通常吃 **预处理后的二进制数据**（`.bin` + `.idx`），不是直接读原始 txt。

1. **准备 JSONL**：每行一个 JSON，至少包含字段 `text`：
  ```json
   {"text": "第一句训练语料……"}
   {"text": "第二句……"}
  ```
2. **选 tokenizer**：例如 HuggingFace 词表文件路径，或仓库支持的其它 `--tokenizer-type`。
3. **调用预处理脚本**（在 `Megatron-LM` 根目录执行）：
  ```powershell
   python tools/preprocess_data.py `
     --input data.jsonl `
     --output-prefix processed_data `
     --tokenizer-type HuggingFaceTokenizer `
     --tokenizer-model "你的tokenizer路径" `
     --workers 8 `
     --append-eod
  ```

更细的参数与最佳实践见仓库内文档：`docs/user-guide/data-preparation.md`，以及官方：[Data preparation / Quickstart](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)。

---

## 6. 仓库结构（读代码时从哪看）

```text
Megatron-LM/
├── megatron/core/      # Megatron Core：并行、Transformer、checkpoint 等
├── megatron/training/  # 训练相关脚本逻辑
├── examples/           # 各类「能抄作业」的入口脚本
├── tools/              # preprocess、checkpoint 工具等
└── docs/               # 与官网呼应的 Markdown 文档
```

---

## 7. 建议的学习顺序（约 1～2 天）

1. 跑通 **第 4 节** 的 `run_simple_mcore_train_loop.py`（2 卡）。
2. 读 `examples/run_simple_mcore_train_loop.py` 里：`parallel_state`、`GPTModel`、`DistributedDataParallel`、`dist_checkpointing` 各干一件事。
3. 打开官方 [Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)，对照 **TP / PP / DP** 缩写理解日志里的进程组。
4. 再挑 `examples/` 里与你目标最接近的一条脚本（GPT / LLaMA / T5 / 多模态），只改数据路径与小批量做「缩小版」实验。

---

## 8. 常见问题

**Q：import megatron 报错？**  
先确认已 `cd` 到仓库根并在同一环境中执行过 `pip install -e .`。

**Q：连不上 GitHub？**  
源码已在本机 `Megatron-LM` 目录时，不影响学习；更新代码需自行解决网络或使用镜像。

**Q：和 HuggingFace 权重互转？**  
官方推荐 [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 做 HF ↔ Megatron checkpoint 转换与配方。

---

## 9. 权威链接（收藏即可）

- [Megatron Core 文档首页](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
- [安装说明（Install）](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)
- [第一次训练（Quickstart）](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)
- [上游仓库 NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

