import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import glob
import itertools
import zipfile
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from model.model_vlm import MiniMindVLM
from datasets import Dataset as HFDataset
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.5):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class VLMDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512, image_special_token='<|image_pad|>', image_token_len=64):
        super().__init__()
        self.dataset = HFDataset.from_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        # ChatML 格式：用 <|im_start|>assistant\n 和 <|im_end|>\n 定位 assistant 回复
        # 不能用 tokenizer.bos_token/eos_token，因为这个 tokenizer 的 bos_token=None, eos_token=<|endoftext|>
        # 而 chat template 实际用的是 <|im_start|> 和 <|im_end|>
        self.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.dataset)

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            content = turn['content'].replace('<image>', self.image_special_token) if turn.get('role') != 'system' else turn['content']
            messages.append({"role": turn['role'], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int):
        row = self.dataset[index]
        conversations = json.loads(row['conversations']) if isinstance(row['conversations'], str) else row['conversations']
        image_bytes = row['image_bytes']
        if not isinstance(image_bytes, list): image_bytes = [image_bytes]
        
        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_inputs_list = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes]
        if hasattr(image_inputs_list[0], 'keys'):
            image_data = {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0) for k in image_inputs_list[0].keys()}
        else:
            image_data = torch.stack(image_inputs_list)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data


class LocalMultiSourceVLMDataset(IterableDataset):
    """Stream the local corpora described by ``configs/local_datasets.json``.

    This deliberately uses Hugging Face's streaming readers: FineWeb and SkyPile
    are too large to first materialize as an Arrow cache.  Every text-only record
    receives a tiny black image, so it can share the VLM training path with image
    and text examples.
    """
    def __init__(self, config_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='<|image_pad|>', image_token_len=64, stage=None,
                 tokenizer_path=None, vision_model_path=None):
        super().__init__()
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        self.sources = [s for s in config['sources'] if s.get('enabled', True)
                        and (stage is None or stage in s.get('stages', [stage]))]
        if not self.sources:
            raise ValueError(f'配置 {config_path} 在 stage={stage!r} 下没有可用数据源')
        self.stage = stage
        self.mixing_unit = config.get('mixing_unit', 'samples')
        self.shuffle_sources = config.get('shuffle_sources', False)
        self.mix_seed = int(config.get('seed', 42))
        if self.mixing_unit not in ('samples', 'tokens'):
            raise ValueError(f'未知 mixing_unit: {self.mixing_unit!r}')
        if self.mixing_unit == 'tokens':
            if stage != 'pretrain' or max_length < 2:
                raise ValueError('token 比例混合需要 stage=pretrain 且 max_length >= 2')
            for source in self.sources:
                weight = source.get('weight', 1)
                if not source.get('packing'):
                    raise ValueError(f'token 比例混合要求数据源 {source["name"]!r} 开启 packing')
                if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                    raise ValueError(f'数据源 {source["name"]!r} 的 weight 必须是正整数')
                if not self._files(source['files']):
                    raise FileNotFoundError(f'数据源 {source["name"]!r} 没有匹配文件: {source["files"]}')
        # 懒加载：存储路径，worker 进程自行创建对象，避免 fork/spawn 序列化问题
        self.tokenizer_path = tokenizer_path
        self.vision_model_path = vision_model_path
        self._tokenizer = tokenizer
        self._preprocess = preprocess
        self.max_length = max_length
        self.image_special_token = image_special_token * image_token_len
        # ChatML 格式：tokenizer.bos_token=None, eos_token=<|endoftext|>，但 chat template 用 <|im_start|>/<|im_end|>
        self.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids
        black = Image.new('RGB', (8, 8), 'black')
        buffer = io.BytesIO(); black.save(buffer, format='JPEG')
        self.black_image = buffer.getvalue()

    def _lazy_init(self):
        """在 worker 进程中按需创建 tokenizer / preprocess，避免继承主进程的 C++ 对象导致死锁"""
        if self._tokenizer is None and self.tokenizer_path:
            print(f'[worker] loading tokenizer...', flush=True)
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
            print(f'[worker] tokenizer loaded', flush=True)
        if self._preprocess is None and self.vision_model_path:
            print(f'[worker] loading preprocessor...', flush=True)
            from transformers import SiglipImageProcessor
            self._preprocess = SiglipImageProcessor.from_pretrained(self.vision_model_path, local_files_only=True)
            print(f'[worker] preprocessor loaded', flush=True)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._lazy_init()
        return self._tokenizer

    @property
    def preprocess(self):
        if self._preprocess is None:
            self._lazy_init()
        return self._preprocess

    def _files(self, patterns):
        patterns = [patterns] if isinstance(patterns, str) else patterns
        return sorted({file for pattern in patterns for file in glob.glob(pattern, recursive=True)})

    @staticmethod
    def _role(role):
        return {'human': 'user', 'user': 'user', 'gpt': 'assistant',
                'assistant': 'assistant', 'system': 'system'}.get(role, role)

    def _conversations(self, row):
        turns = row.get('conversations', row.get('messages'))
        if isinstance(turns, str):
            turns = json.loads(turns)
        if turns:
            return [{'role': self._role(turn.get('role', turn.get('from', 'user'))),
                     'content': str(turn.get('content', turn.get('value', '')))}
                    for turn in turns if turn.get('content', turn.get('value'))]
        text = row.get('text', row.get('content', ''))
        # Raw pretraining text is causal-LM data, not an artificial assistant
        # answer to "please read this text".  The latter teaches web-page dumps
        # as responses to user questions.
        if text and self.stage == 'pretrain':
            return [{'role': 'raw', 'content': str(text)}]
        return []

    @staticmethod
    def _matches_filters(row, source):
        for field, expected in source.get('filters', {}).items():
            actual = row.get(field)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _stream_rows(self, source, shard=0, shards=1):
        # 直接读取本地文件，绕过 HF datasets streaming 的 worker 数限制。
        # 关键优化：把 sharding 推到 row group / line 级别，每个 worker 只读自己那一份，
        # 避免所有 worker 都全量读文件再 islice 丢弃，造成 N 倍 I/O 和 CPU 浪费。
        files = self._files(source['files'])
        if not files:
            return
        fmt = source.get('format') or ('parquet' if files[0].endswith('.parquet') else 'json')
        if fmt == 'parquet':
            # parquet 按 row group 分片：每个 worker 只读取 num_row_groups 中 idx % shards == shard 的 row group
            for file in files:
                pf = pq.ParquetFile(file)
                num_row_groups = pf.num_row_groups
                if num_row_groups < shards:
                    # row group 数少于 worker 数，改为按行分片
                    for rg_idx in range(num_row_groups):
                        table = pf.read_row_group(rg_idx)
                        for row_idx, row in enumerate(table.to_pylist()):
                            if row_idx % shards != shard:
                                continue
                            yield row
                else:
                    for rg_idx in range(num_row_groups):
                        if rg_idx % shards != shard:
                            continue
                        table = pf.read_row_group(rg_idx)
                        for row in table.to_pylist():
                            yield row
        else:  # json / jsonl
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    if file.endswith('.jsonl'):
                        for line_idx, line in enumerate(f):
                            if line_idx % shards != shard:
                                continue
                            line = line.strip()
                            if line:
                                yield json.loads(line)
                    else:
                        # 大 JSON 数组文件：用 ijson 流式解析，避免一次性载入内存
                        try:
                            import ijson
                            with open(file, 'rb') as fb:
                                for idx, row in enumerate(ijson.items(fb, 'item')):
                                    if idx % shards != shard:
                                        continue
                                    yield row
                        except ImportError:
                            # 退化为一次性加载
                            with open(file, 'r', encoding='utf-8') as f2:
                                for idx, row in enumerate(json.load(f2)):
                                    if idx % shards != shard:
                                        continue
                                    yield row

    def _samples(self, source, shard=0, shards=1):
        if self.stage == 'pretrain' and source.get('packing', False):
            token_buffer = []
            for row in self._stream_rows(source, shard, shards):
                if not self._matches_filters(row, source):
                    continue
                text = row.get('text', row.get('content', ''))
                # Instruction corpora can contribute to causal pretraining too:
                # serialize complete turns and supervise ALL tokens. SFT keeps
                # its separate assistant-only masking and truncation policy.
                if not text:
                    conversations = self._conversations(row)
                    if conversations:
                        text = self.tokenizer.apply_chat_template(
                            conversations, tokenize=False, add_generation_prompt=False)
                if not text:
                    continue
                token_buffer.extend(self.tokenizer(str(text), add_special_tokens=False).input_ids)
                token_buffer.append(self.tokenizer.eos_token_id)
                while len(token_buffer) >= self.max_length:
                    packed_ids = token_buffer[:self.max_length]
                    del token_buffer[:self.max_length]
                    yield [{'role': 'raw_ids', 'content': packed_ids}], self.black_image
            return

        archive = None
        try:
            if source.get('image_archive'):
                archive = zipfile.ZipFile(source['image_archive'])
            for row in self._stream_rows(source, shard, shards):
                if not self._matches_filters(row, source):
                    continue
                conversations = self._conversations(row)
                if not conversations:
                    continue
                if source.get('max_turns') and len(conversations) > int(source['max_turns']):
                    continue
                if source.get('max_chars'):
                    total_chars = sum(len(turn.get('content', '')) for turn in conversations)
                    if total_chars > int(source['max_chars']):
                        continue
                image_bytes = row.get('image_bytes')
                if image_bytes is None and source.get('image_archive'):
                    try:
                        image_bytes = archive.read(row[source.get('image_field', 'image')])
                    except (KeyError, TypeError):
                        continue
                elif image_bytes is None and source.get('image_root'):
                    image_name = row.get(source.get('image_field', 'image'))
                    image_path = os.path.join(source['image_root'], str(image_name))
                    try:
                        with open(image_path, 'rb') as f:
                            image_bytes = f.read()
                    except OSError:
                        continue
                elif image_bytes is None:
                    image_bytes = self.black_image
                if isinstance(image_bytes, list):
                    image_bytes = image_bytes[0]
                yield conversations, image_bytes
        finally:
            if archive:
                archive.close()

    def _encode(self, conversations, image_bytes):
        if len(conversations) == 1 and conversations[0]['role'] == 'raw_ids':
            input_ids = conversations[0]['content']
            if len(input_ids) != self.max_length:
                return None
            labels = input_ids.copy()
            image_data = MiniMindVLM.image2tensor(
                Image.open(io.BytesIO(image_bytes)).convert('RGB'), self.preprocess)
            return (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long), image_data)

        if len(conversations) == 1 and conversations[0]['role'] == 'raw':
            input_ids = self.tokenizer(conversations[0]['content'], add_special_tokens=False).input_ids
            input_ids = input_ids[:self.max_length - 1] + [self.tokenizer.eos_token_id]
            labels = input_ids.copy()
            padding = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding
            labels += [-100] * padding
            try:
                image_data = MiniMindVLM.image2tensor(
                    Image.open(io.BytesIO(image_bytes)).convert('RGB'), self.preprocess)
            except Exception:
                return None
            return (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long), image_data)

        conversations = pre_processing_chat(conversations)
        messages = []
        for turn in conversations:
            content = turn['content'].replace('<image>', self.image_special_token) if turn['role'] != 'system' else turn['content']
            messages.append({'role': turn['role'], 'content': content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = post_processing_chat(prompt)
        full_ids = self.tokenizer(prompt).input_ids
        input_ids = full_ids[:self.max_length]
        # 截断保护：样本超长被截断时，若最后一个 assistant 回复没有完整的 <|im_end|> 收尾，
        # 丢弃该样本——否则模型会学到"回答没有结束"，表现为生成时死循环复读
        if len(full_ids) > self.max_length:
            last_bos = -1
            for i in range(len(input_ids) - len(self.bos_id), -1, -1):
                if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                    last_bos = i
                    break
            if last_bos != -1:
                has_eos = any(input_ids[i:i + len(self.eos_id)] == self.eos_id
                              for i in range(last_bos, len(input_ids) - len(self.eos_id) + 1))
                if not has_eos:
                    return None
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = VLMDataset.generate_labels(self, input_ids)
        # 跳过没有有效 label 的样本（如 assistant 回复被 max_seq_len 截断），避免 cross_entropy 返回 nan
        if all(l == -100 for l in labels):
            return None
        try:
            image_data = MiniMindVLM.image2tensor(Image.open(io.BytesIO(image_bytes)).convert('RGB'), self.preprocess)
        except Exception:
            return None
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data

    def __iter__(self):
        # Split the global stream across DDP ranks and DataLoader workers.
        # sharding 推到 _stream_rows 层（按 row group / line 切），这里只做跨 source 的 round-robin
        worker = get_worker_info()
        # 在 worker 进程中重新创建 tokenizer/preprocess，避免继承主进程的 C++ 对象导致死锁
        if worker is not None:
            self._tokenizer = None
            self._preprocess = None
            self._lazy_init()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        shard, shards = rank * worker_count + worker_id, world_size * worker_count

        # 支持权重：每个 source 可配 "weight"（默认 1），控制 round-robin 中的采样比例
        # 图文数据量小，用 weight=1；纯文本数据量大，用 weight=4 → 图文:文本 ≈ 20:80
        weights = [source.get('weight', 1) for source in self.sources]
        mix_rng = random.Random(self.mix_seed + shard)

        # 循环迭代器：数据源耗尽后自动重新开始，避免小数据源（如图文）过早耗尽
        def cycling_iter(source):
            while True:
                yielded = False
                for x in self._samples(source, shard, shards):
                    yield x
                    yielded = True
                if not yielded:
                    if self.mixing_unit == 'tokens':
                        raise ValueError(
                            f'数据源 {source["name"]!r} 在 shard={shard}/{shards} 中'
                            '没有可用 packed 样本，无法维持 token 比例')
                    print(f'[worker {shard}] source "{source["name"]}" 为空，跳过', flush=True)
                    return  # 数据源在此 shard 下无数据，退出避免死循环

        iterators = [cycling_iter(source) for source in self.sources]

        def mixed_stream():
            while True:
                schedule = [i for i, weight in enumerate(weights) for _ in range(weight)]
                if self.shuffle_sources:
                    mix_rng.shuffle(schedule)
                for i in schedule:
                    iterator = iterators[i]
                    if iterator is None:
                        continue
                    try:
                        yield next(iterator)
                    except StopIteration:
                        iterators[i] = None
                if all(it is None for it in iterators):
                    return
        for conversations, image_bytes in mixed_stream():
            encoded = self._encode(conversations, image_bytes)
            if self.mixing_unit == 'tokens' and encoded is None:
                raise ValueError('packed 样本编码失败，无法维持 token 比例')
            if encoded is not None:
                yield encoded

    def source_summary(self):
        total = sum(max(0, int(source.get('weight', 1))) for source in self.sources)
        return [
            (source['name'], int(source.get('weight', 1)),
             100.0 * int(source.get('weight', 1)) / total,
             source.get('filters', {}))
            for source in self.sources
        ]

# 测试parquet数据读取和可视化
if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        pf = pq.ParquetFile(path); n = pf.num_row_groups; t = pa.concat_tables([pf.read_row_group(i * n // 5).slice(0, 1) for i in range(5)]); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            img_data = t['image_bytes'][i].as_py(); img_data = img_data[0] if isinstance(img_data, list) else img_data
            ax[i].imshow(Image.open(io.BytesIO(img_data))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'已保存{out}, 共{pf.metadata.num_rows}条')
