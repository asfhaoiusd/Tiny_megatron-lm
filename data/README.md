# 数据目录

## TinyStories

用于 MoELLM 小规模语言建模实验的合成英文短故事集（论文：[TinyStories](https://arxiv.org/abs/2305.07759)）。

### 下载

在仓库根目录执行：

```bash
python scripts/download_tinystories.py
```

默认保存到 `data/tinystories/`：

| 文件 | 说明 | 约大小 |
|------|------|--------|
| `TinyStories-train.txt` | 训练集，每行一个故事 | ~1.9 GB |
| `TinyStories-valid.txt` | 验证集 | ~19 MB |

国内网络默认使用 [hf-mirror.com](https://hf-mirror.com)。若可直连 Hugging Face：

```bash
python scripts/download_tinystories.py --mirror https://huggingface.co
```

仅下载验证集（快速试跑）：

```bash
python scripts/download_tinystories.py --only valid
```

下载中断后重新运行同一命令即可断点续传。

### 读取示例

```python
from pathlib import Path

path = Path("data/tinystories/TinyStories-train.txt")
with path.open(encoding="utf-8") as f:
    story = f.readline().strip()
print(story[:200])
```

大文件训练时请用按行迭代或分块读取，避免一次性 `read()` 进内存。
