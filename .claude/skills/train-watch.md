---
name: train-watch
description: Monitor and analyze LLM training metrics. Read metrics.json from single runs or summary.json from comparison runs to surface loss trends, training speed, parameter counts, and anomalies.
---

# train-watch — 训练监控 & 诊断

监控本项目（~30M MoELLM + TinyStories）的训练产物，输出可读对比表和异常诊断。

## 数据来源

两种输出格式：

1. **单次训练** — `pre_model/moellm_30m_{type}/metrics.json`
```json
{"attention_type": "mla", "valid_loss": 6.88, "steps": 500, "params": 29825568}
```
2. **对比训练** — `pre_model/attention_compare/summary.json`（数组，每元素一个 attention_type；同时各子目录下还有逐项 `metrics.json`）
```json
[{"attention_type": "mha", "params_m": 31.216, "train_loss": 21.20, "valid_loss": 6.58, "ms_per_step": 111.78, ...}, ...]
```

优先使用 `summary.json`（包含 `train_loss` + `ms_per_step`，更完整）。

## 执行流程

### Step 1 — 定位数据

先确认 `pre_model/` 下哪些子目录存在 `metrics.json`：

```bash
find pre_model -name "metrics.json" -o -name "summary.json"
```

### Step 2 — 运行分析脚本

```bash
python scripts/watch_metrics.py
```

此脚本参数：
- `--summary`：指定 summary.json 路径（默认 `pre_model/attention_compare/summary.json`）
- `--metrics`：指定单个 metrics.json 路径（可多次传入）
- `--compare`：同时传入多次训练的结果，做跨实验对比
- `--alert-loss-above`：valid_loss 高于此值标红（默认 10.0）
- `--alert-time-above`：ms/step 高于此值告警（默认 200）

### Step 3 — 解读输出

脚本会输出：
- **对比表**：`type | params | train_loss | valid_loss | ms/step`，最优项加 `(*)`
- **异常检测**：
  - valid_loss 离散太大 → 可能训练不稳定或学习率不当
  - 同类注意力参数量差异 > 5% → 配置不公，结论不可靠
  - ms/step 可能受预热步数影响，注意 `warmup_steps` 配置
- **趋势判断**（若有多次运行的历史数据）：
  - loss 不降反升 → learning rate 可能过高
  - step time 突增 → 可能触发了 CPU fallback 或显存不足 swap

## 常见问题的诊断建议

| 现象 | 可能原因 | 排查方向 |
|------|---------|---------|
| valid_loss 远高于 train_loss | 过拟合；valid 故事太少 | 增加 max_valid_stories，减小模型 |
| MLA loss 显著高于 MHA | MLA 参数翻译不够公平 | 检查 `_mla_dims_for_30m()` 是否匹配参数量 |
| ms/step 三者接近 | GPU 计算瓶颈在 MoE 而非 attention | 正常，~30M 参数量下 attention 占比低 |
| CUDA 不可用回退 CPU | RTX 50 系 sm_120 不兼容 | 装 PyTorch nightly CUDA 12.8+ |
| loss=NaN | 学习率过高 / 数值不稳定 | 降 lr 到 1e-4，开 grad clipping 确认生效 |

## 使用示例

```
/train-watch                          → 全量扫描 pre_model/ 下所有 metrics
/train-watch --compare mha mla        → 只对比 mha 和 mla
/train-watch --alert-loss-above 8.0   → 更严格的 loss 告警阈值
```
