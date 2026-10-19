"""
下载代码能力训练数据集（国内可用版）
用法: python download_code_datasets.py
数据保存到: ~/ai/data/code/
"""
import os
import json
import time
import subprocess
import shutil

CODE_DIR = os.path.expanduser('~/ai/data/code')
os.makedirs(CODE_DIR, exist_ok=True)

# HuggingFace 镜像
HF_MIRROR = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = HF_MIRROR


def try_hf_snapshot(repo_id, repo_type='dataset', cache_dir=None):
    """用 huggingface_hub.snapshot_download 下载整个仓库（比 load_dataset 更可靠）"""
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            endpoint=HF_MIRROR,
            cache_dir=cache_dir,
        )
        return path
    except Exception as e:
        print(f"  snapshot_download 失败: {e}")
        return None


def try_git_clone(repo_id, repo_type='dataset'):
    """用 git clone 从 HF 镜像下载"""
    url = f"{HF_MIRROR}/{repo_type}s/{repo_id}"
    tmp = f"/tmp/hf_{repo_id.replace('/', '_')}"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    try:
        env = os.environ.copy()
        env['GIT_LFS_SKIP_SMUDGE'] = '1'  # 先跳过 LFS，后续按需下载
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', url, tmp],
            capture_output=True, text=True, timeout=120, env=env
        )
        if result.returncode == 0:
            # 检查是否有 LFS 文件需要拉取
            lfs_files = []
            for root, dirs, files in os.walk(tmp):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        with open(fp, 'r') as fh:
                            first_line = fh.readline(200)
                            if first_line.startswith('version https://git-lfs'):
                                lfs_files.append(fp)
                    except:
                        pass
            if lfs_files:
                print(f"  拉取 {len(lfs_files)} 个 LFS 文件...")
                subprocess.run(['git', 'lfs', 'pull'], cwd=tmp, timeout=300,
                              capture_output=True, text=True, env=os.environ.copy())
            return tmp
        else:
            print(f"  git clone 失败: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  git clone 异常: {e}")
        return None


def find_data_files(repo_path, extensions=('.json', '.jsonl', '.parquet')):
    """在下载的仓库中找数据文件"""
    found = []
    for root, dirs, files in os.walk(repo_path):
        # 跳过 .git 目录
        if '.git' in root:
            continue
        for f in files:
            if f.endswith(extensions) and f != '.gitattributes':
                found.append(os.path.join(root, f))
    return found


def load_records_from_file(filepath):
    """从 json/jsonl/parquet 文件加载记录"""
    import pyarrow.parquet as pq
    records = []
    if filepath.endswith('.parquet'):
        table = pq.read_table(filepath)
        for batch in table.to_batches():
            for row in batch.to_pylist():
                records.append(row)
    elif filepath.endswith('.jsonl'):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and 'data' in data:
                records = data['data']
            else:
                records = [data]
    return records


def save_json(records, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  -> {filepath} ({len(records)} 条)")


def save_jsonl(records, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"  -> {filepath} ({len(records)} 条)")


def download_magicoder():
    """Magicoder OSS-Instruct: 75k 高质量代码指令（SFT）"""
    print("\n[1/3] 下载 Magicoder OSS-Instruct-75K (SFT)...")
    repo_id = "ise-uiuc/Magicoder-OSS-Instruct-75K"

    # 方式1: snapshot_download
    path = try_hf_snapshot(repo_id)
    # 方式2: git clone
    if not path:
        path = try_git_clone(repo_id)

    if not path:
        print("  所有下载方式均失败，跳过")
        return 0

    files = find_data_files(path)
    if not files:
        print(f"  仓库中未找到数据文件，已下载文件: {os.listdir(path)}")
        return 0

    print(f"  找到数据文件: {[os.path.basename(f) for f in files]}")
    all_records = []
    for f in files:
        all_records.extend(load_records_from_file(f))

    # 转换格式
    converted = []
    for row in all_records:
        problem = row.get('problem', row.get('instruction', row.get('prompt', '')))
        solution = row.get('solution', row.get('response', row.get('completion', row.get('output', ''))))
        if problem and solution:
            converted.append({
                'conversations': [
                    {'role': 'user', 'content': str(problem)},
                    {'role': 'assistant', 'content': str(solution)}
                ]
            })

    if converted:
        save_json(converted, f'{CODE_DIR}/magicoder_oss_instruct.json')
    else:
        print(f"  转换失败，原始字段: {list(all_records[0].keys()) if all_records else '空'}")
    return len(converted)


def download_codealpaca():
    """CodeAlpaca: 20k 代码指令（SFT）"""
    print("\n[2/3] 下载 CodeAlpaca (SFT)...")
    repo_id = "sahil2801/CodeAlpaca"

    path = try_hf_snapshot(repo_id)
    if not path:
        path = try_git_clone(repo_id)

    if not path:
        # 备选: HuggingFaceH4/CodeAlpaca_20K
        print("  尝试备选: HuggingFaceH4/CodeAlpaca_20K")
        path = try_hf_snapshot("HuggingFaceH4/CodeAlpaca_20K")
        if not path:
            path = try_git_clone("HuggingFaceH4/CodeAlpaca_20K")

    if not path:
        print("  所有下载方式均失败，跳过")
        return 0

    files = find_data_files(path)
    if not files:
        print(f"  仓库中未找到数据文件")
        return 0

    all_records = []
    for f in files:
        all_records.extend(load_records_from_file(f))

    converted = []
    for row in all_records:
        prompt = row.get('prompt', row.get('instruction', ''))
        completion = row.get('completion', row.get('response', row.get('output', '')))
        if prompt and completion:
            converted.append({
                'conversations': [
                    {'role': 'user', 'content': str(prompt)},
                    {'role': 'assistant', 'content': str(completion)}
                ]
            })

    if converted:
        save_json(converted, f'{CODE_DIR}/codealpaca.json')
    return len(converted)


def download_github_code(max_samples=500000):
    """GitHub Code: 代码预训练数据"""
    print(f"\n[3/3] 下载代码预训练数据 (最多 {max_samples} 条)...")

    # 方式1: 尝试 codeparrot/github-code (需要 trust_remote_code)
    repo_id = "codeparrot/github-code"
    print(f"  尝试: {repo_id}")

    # 先试 snapshot_download 拿 parquet 文件
    path = try_hf_snapshot(repo_id)
    if not path:
        # 备选: bigcode/the-stack-smol
        print(f"  尝试备选: bigcode/the-stack-smol")
        path = try_hf_snapshot("bigcode/the-stack-smol")

    if not path:
        # 备选: 下载 HuggingFaceH4/the-stack-v2-train-smol-ids
        print(f"  尝试备选: HuggingFaceH4/the-stack-v2-train-smol-ids")
        path = try_hf_snapshot("HuggingFaceH4/the-stack-v2-train-smol-ids")

    if not path:
        # 最终备选: 用 modelscope
        print(f"  尝试 ModelScope: codefuse-ai/CodeExercise-Python-27k")
        try:
            from modelscope.msdatasets import MsDataset
            ds = MsDataset.load('codefuse-ai/CodeExercise-Python-27k', split='train')
            records = []
            for row in ds:
                code = str(row.get('text', row.get('content', row.get('code', ''))))
                if len(code) < 50 or len(code) > 10000:
                    continue
                records.append({
                    'text': code,
                    'language': 'python',
                    'conversations': [
                        {'role': 'user', 'content': '请阅读以下 Python 代码。'},
                        {'role': 'assistant', 'content': code}
                    ]
                })
                if len(records) >= max_samples:
                    break
            if records:
                save_jsonl(records, f'{CODE_DIR}/github_code.jsonl')
                return len(records)
        except Exception as e:
            print(f"  ModelScope 也失败: {e}")

        print("  所有下载方式均失败")
        print("  请手动下载代码数据到 ~/ai/data/code/github_code.jsonl")
        return 0

    # 解析下载的文件
    files = find_data_files(path, extensions=('.json', '.jsonl', '.parquet'))
    if not files:
        print(f"  仓库中未找到数据文件")
        return 0

    print(f"  找到 {len(files)} 个数据文件")
    records = []
    for f in files:
        if len(records) >= max_samples:
            break
        try:
            batch = load_records_from_file(f)
            for row in batch:
                if len(records) >= max_samples:
                    break
                code = str(row.get('code', row.get('content', row.get('text', ''))))
                lang = row.get('language', row.get('lang', 'unknown'))
                if len(code) < 50 or len(code) > 10000:
                    continue
                records.append({
                    'text': code,
                    'language': lang,
                    'conversations': [
                        {'role': 'user', 'content': f'请阅读以下 {lang} 代码。'},
                        {'role': 'assistant', 'content': code}
                    ]
                })
            print(f"  已加载 {len(records)} 条...")
        except Exception as e:
            print(f"  解析 {f} 失败: {e}")

    if records:
        save_jsonl(records, f'{CODE_DIR}/github_code.jsonl')
    return len(records)


def print_summary(results):
    print("\n" + "=" * 60)
    print("下载完成汇总:")
    print("=" * 60)
    total = 0
    for name, count in results.items():
        if count > 0:
            print(f"  {name}: {count:,} 条")
            total += count
    print(f"  {'总计':>20}: {total:,} 条")
    print(f"\n数据目录: {CODE_DIR}/")
    if total > 0:
        subprocess.run(['ls', '-lh', CODE_DIR])


if __name__ == '__main__':
    print(f"代码数据将下载到: {CODE_DIR}/")
    print(f"HuggingFace 镜像: {HF_MIRROR}")

    results = {}
    t0 = time.time()

    results['Magicoder OSS-Instruct (SFT)'] = download_magicoder()
    results['CodeAlpaca (SFT)'] = download_codealpaca()
    results['GitHub Code (Pretrain)'] = download_github_code(max_samples=500000)

    print(f"\n总耗时: {time.time() - t0:.0f} 秒")
    print_summary(results)
