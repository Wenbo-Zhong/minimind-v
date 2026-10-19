"""Evaluate a MiniMind checkpoint on held-out GSM8K and CMATH questions."""
import argparse
from eval_calculator import add_calculator_args, calculator_settings, run_dialogue
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer



def gsm_gold(answer):
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    return normalize_number(matches[-1]) if matches else None


def normalize_number(value):
    if value is None:
        return None
    value = str(value).replace(",", "").strip()
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else format(number, ".12g")


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
BARE_NUMBER_RE = re.compile(
    r"^\s*[*`#]*\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:个|只|岁|元|辆|本|块|米|千克|克|吨|人|小时|分钟|天|年|美元|公斤|页|支|场|次|%)*"
    r"[。.!]?\s*[*`#]*\s*$"
)
BARE_EQUATION_RE = re.compile(
    r"^\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*[+\-×*/÷]\s*"
    r"([-+]?\d[\d,]*(?:\.\d+)?)\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*[。.!]?\s*$"
)
ANSWER_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer\s*(?:is|:)|答案\s*(?:是|为|[:：])|最终答案\s*[:：]|####)",
    re.IGNORECASE,
)


def extract_prediction(answer):
    """Extract a numeric final answer without rewarding repetition or prefixes.

    With an explicit answer marker, inspect the text after the last marker and
    take a standalone final number or the RHS of a final one-line equation.
    Without a marker, only accept a standalone numeric final line or equation.
    """
    text = str(answer or "").strip()
    if not text:
        return None
    response_lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_response_lines = [
        re.sub(r"\s+", " ", line).strip().lower() for line in response_lines
    ]
    # Check the whole response before slicing to the last answer marker;
    # otherwise a loop that repeats "The answer is ..." would look valid.
    if len(normalized_response_lines) != len(set(normalized_response_lines)):
        return None
    markers = list(ANSWER_MARKER_RE.finditer(text))
    if markers:
        text = text[markers[-1].end():].lstrip(" :：\t-`").strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    # A final bare number after multiline reasoning may simply be the tail of
    # a repetition loop cut off by max_new_tokens. Long answers must use an
    # explicit answer marker so such truncated fragments are not scored.
    if not markers and len(lines) != 1:
        return None
    last_line = lines[-1]
    if last_line.endswith(('?', '？')):
        return None
    # A marked final answer may be formatted as an equation, e.g. "24 - 20 = 4".
    # Read its RHS, not the first operand that appears after the marker.
    equation = re.search(
        r"=\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)"
        r"\s*(?:个|只|岁|元|辆|本|块|米|千克|克|吨|人|小时|分钟|天|年|美元|公斤|页|支|场|次|%)*"
        r"\s*[。.!`$]*\s*$",
        last_line,
    )
    if equation:
        return normalize_number(equation.group(1))
    bare = BARE_NUMBER_RE.fullmatch(last_line)
    return normalize_number(bare.group(1)) if bare else None


def read_gsm_rows(path, limit):
    parquet = pq.ParquetFile(path)
    rows = []
    for group in range(parquet.num_row_groups):
        rows.extend(parquet.read_row_group(group).to_pylist())
        if limit is not None and len(rows) >= limit:
            break
    return rows if limit is None else rows[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum examples per dataset; use 0 for the full datasets')
    parser.add_argument('--max_new_tokens', type=int, default=192)
    parser.add_argument('--gsm8k_path', default='/home/zwb/ai/data/eval/gsm8k/main/test-00000-of-00001.parquet')
    parser.add_argument('--cmath_path', default='/home/zwb/ai/data/eval/cmath/cmath_test.jsonl')
    parser.add_argument('--output', default=None)
    add_calculator_args(parser)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_tool_calls < 0:
        parser.error("invalid token budget or tool call limit")

    if args.max_samples < 0:
        parser.error('--max_samples must be zero (full set) or a positive number')
    limit = None if args.max_samples == 0 else args.max_samples
    output = Path(args.output or f"{Path(args.weight).stem}.generalization.json")

    state = torch.load(args.weight, map_location='cpu', weights_only=False)
    metadata = state.get('metadata') if isinstance(state, dict) else None
    snapshot = Path(args.weight).parent / 'tokenizer'
    tokenizer_path = args.tokenizer_path or (str(snapshot) if metadata and snapshot.is_dir() else './model')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if metadata:
        from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
        from trainer.train_text import file_hash
        for name, expected in metadata['tokenizer_sha256'].items():
            if file_hash(Path(tokenizer_path) / name) != expected:
                raise ValueError('Tokenizer fingerprint mismatch: ' + name)
        model = MiniMindForCausalLM(MiniMindConfig(**metadata['model_config']))
        model.load_state_dict(state['model'], strict=True)
    else:
        from model.model_vlm import MiniMindVLM, VLMConfig
        model = MiniMindVLM(
            VLMConfig(hidden_size=1024, num_hidden_layers=24),
            vision_model_path='./model/siglip2-base-p32-256-ve',
        )
        model.load_state_dict({k: v for k, v in state.items() if 'mask' not in k}, strict=False)
    del state
    model = model.eval().to(
        args.device,
        dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32,
    )
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')

    def generate_turn(messages, remaining_tokens):
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=remaining_tokens,
                do_sample=False,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = generated[0, input_ids.shape[1]:]
        return tokenizer.decode(completion, skip_special_tokens=True).strip(), int(completion.numel())

    results = {'gsm8k': {'passed': 0, 'total': 0, 'results': []},
               'cmath': {'passed': 0, 'total': 0, 'results': []}}

    def save_checkpoint(complete=False):
        payload = {
            'weight': args.weight,
            'calculator': calculator_settings(args),
            'complete': complete,
            'max_samples_per_dataset': args.max_samples,
            'max_new_tokens': args.max_new_tokens,
            'answer_scoring': 'strict final numeric answer; repeated lines rejected',
            'results': results,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def record(dataset, row, question, gold, prompt):
        answer, dialogue = run_dialogue(generate_turn, prompt, args.max_new_tokens,
                                        args.calculator, args.max_tool_calls)
        pred = extract_prediction(answer)
        passed = pred is not None and normalize_number(gold) == pred
        entry = {
            'question': question,
            'gold': normalize_number(gold),
            'pred': pred,
            'passed': passed,
            'answer': answer,
            'dialogue': dialogue,
        }
        bucket = results[dataset]
        bucket['results'].append(entry)
        bucket['total'] += 1
        bucket['passed'] += int(passed)
        print(f"[{dataset} {bucket['total']}] {'PASS' if passed else 'FAIL'} "
              f"gold={entry['gold']} pred={pred} answer={answer[:180]!r}", flush=True)
        if bucket['total'] % 5 == 0:
            save_checkpoint()

    gsm_rows = read_gsm_rows(args.gsm8k_path, limit)
    print(f"GSM8K: evaluating {len(gsm_rows)} examples", flush=True)
    for row in gsm_rows:
        question = row['question']
        prompt = (
            'Solve this math word problem. Show only enough work to calculate it. '
            'Put the final numeric answer on the last line as `Final answer: <number>`.\n'
            + question
        )
        record('gsm8k', row, question, gsm_gold(row['answer']), prompt)

    cmath_rows = []
    with open(args.cmath_path, encoding='utf-8') as source:
        for line in source:
            if limit is not None and len(cmath_rows) >= limit:
                break
            cmath_rows.append(json.loads(line))
    print(f"CMATH: evaluating {len(cmath_rows)} examples", flush=True)
    for row in cmath_rows:
        question = row['question']
        prompt = (
            '请解答下面的数学题。请简要计算，并将最终数值答案单独放在最后一行，格式为“最终答案：数字”。\n'
            + question
        )
        record('cmath', row, question, row['golden'], prompt)

    save_checkpoint(complete=True)
    for name, summary in results.items():
        total = summary['total']
        print(f"{name}: {summary['passed']}/{total} ({summary['passed']/max(total, 1):.1%})")
    print('Result:', output)


if __name__ == '__main__':
    main()
