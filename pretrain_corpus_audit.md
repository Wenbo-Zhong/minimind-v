# 本地预训练语料核实

核实日期：2026-10-13。数据根目录：`/home/zwb/ai/data/text`。
详细抽样结果：`pretrain_corpus_audit.json`。复现脚本：`scripts/audit_pretrain_corpus.py`。

## 已核实的数量

| 来源 | 文件数 | 数量依据 | 当前 tokenizer 文本 token 数 |
| --- | ---: | --- | ---: |
| FineWeb-Edu 10BT | 14 | Parquet 元数据精确计数 9,672,101 行；完整读取 token_count 列得到原始计数 9,972,918,177 | 抽样估计 9.53B，约 95% 抽样区间 8.92–10.13B |
| SkyPile subset | 87 | 104,799,875,762 字节；全文件覆盖的随机字节位置抽样，共 2,784 次 | 抽样估计 23.38B，约 95% 抽样区间 23.26–23.50B |
| 合计 | 101 | 原始语料存量，不是训练消耗量 | 约 32.91B，即 329 亿 |

FineWeb 的原始 token_count 使用数据发布方的分词统计，不能直接当作当前 tokenizer 的计数。当前 tokenizer 的总词表大小为 151,665，模型 embedding 预留 151,936 行。

估计没有进行全量分词、全量去重、近似去重或内容质量筛选；区间只表示抽样误差。SkyPile 抽到的不同位置中存在 28 个重复文本，但抽样按字节位置进行，长文更容易被抽到；不能把这个比例当作全库重复率。FineWeb 的抽样估计依赖 row group 基本等大的条件。

Ubuntu 根分区 `/dev/nvme1n1p4` 为 btrfs，检查时约 1.3 TiB 可用。

## 实际训练消耗量

- `checkpoints/pretrain_text_repair_probe_1024_resume.pth`：step=10,000；明确记录 tokens_seen=122,880,000，即 1.2288 亿。
- `checkpoints/pretrain_repair_packed_1024_resume.pth`：step=50,000；没有 tokens_seen。
- `checkpoints/pretrain_vlm_1024_resume.pth`：step=50,000；没有 tokens_seen。
- 不能依据文件名中的 400k 或语料存量确认历史累计训练了 30B tokens。缺少当时 batch、长度、有效标签数量、训练轮次及各轮初始化来源。
- 仅作为算式示例：单卡 batch=32、长度=450、400,000 个 micro-steps 的位置数上限是 5.76B，padding 和非监督位置还会使有效 token 数更低。梯度累积不再乘到这个数字里。此例不是已确认的历史配置。

## 加载器和配置观察

`dataset/lm_dataset.py` 当前依次读取排序后的文件，并依次读取行/row group；多来源按权重轮流取样，来源耗尽后循环。没有全局或缓冲区随机打乱。

因此，有限 steps 的 epoch 不代表遍历全部语料。重新启动且不恢复数据位置的短训练可能重复读取各 worker 的语料前部。JSONL 每个 worker 仍扫描所有行后按行号分片，属于额外的读取开销。

- `configs/pretrain_repair_packed.json`：SkyPile/FineWeb = 1:1，两者均启用 packing；完整等长 packed 样本时，token 配比近似 50:50。
- `configs/local_datasets.json` 的 pretrain 来源：OpenHermes/FineWeb/SkyPile/Magicoder = 2:1:1:1，名义样本比例 40%/20%/20%/20%。文本未启用 packing；样本比例不是有效 token 比例。
- `configs/local_datasets_stage2.json`：两种视觉来源各 weight=1，两种文本来源各 weight=4，名义样本比例视觉 20%、文本 80%；这也是样本比例而非有效 token 比例。

## 参数选择

按当前 MiniMind 实现在 meta device 实例化，仅计语言模型独立参数，不含视觉编码器/视觉投影：

| hidden_size | 层数 | 语言模型参数 |
| ---: | ---: | ---: |
| 1024 | 24 | 471,781,376 |
| 1024 | 32 | 577,180,672 |
| 1280 | 24 | 684,098,560 |
| 1280 | 32 | 847,304,960 |
| 1536 | 24 | 941,242,368 |
| 1536 | 32 | 1,177,198,080 |

329 亿原始 token 的存量允许考虑约 0.85–1.18B 的新模型；是否可行仍取决于去重清洗后保留量和实际训练时间。建议下一轮优先考虑 hidden_size=1280、32 层，约 0.85B；先解决打乱、覆盖率、token 预算和独立验证集，再从头预训练。

现有 checkpoint 的 embedding 大小与当前模型兼容。Git HEAD 中原始 tokenizer 是 6,400 词表，工作区是新的大词表，说明仓库曾发生 tokenizer 替换，但这不证明旧 checkpoint 曾用另一套 token ID 训练。历史 checkpoint 未保存 tokenizer 指纹，仍无法确认其语义映射来源；新实验应固定 tokenizer 并保存指纹。

检查时 `nvidia-smi` 无法连接驱动，因此本轮未测新模型的实际 GPU 吞吐或显存需求。

## 复现

在项目根目录执行：

```bash
/home/zwb/miniconda3/envs/minimind-v/bin/python scripts/audit_pretrain_corpus.py
```

脚本只读取语料并写统计 JSON，不启动训练，不改数据。
