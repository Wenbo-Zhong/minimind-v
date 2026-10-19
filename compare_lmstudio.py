#!/usr/bin/env python3
"""Compare MiniMind with an LM Studio OpenAI-compatible local model."""
import argparse, json
from pathlib import Path
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:1234/v1')
    ap.add_argument('--model', default=None, help='LM Studio model id; omit to use the loaded model')
    ap.add_argument('--suite', default='./configs/eval_suite.json')
    ap.add_argument('--output', default='lmstudio_comparison.json')
    ap.add_argument('--max-tokens', type=int, default=96)
    args = ap.parse_args()
    suite = json.loads(Path(args.suite).read_text(encoding='utf-8'))['questions']
    base = args.base_url.rstrip('/')
    models = requests.get(base + '/models', timeout=10).json().get('data', [])
    model = args.model or (models[0]['id'] if models else None)
    if not model:
        raise SystemExit('LM Studio 没有返回已加载模型，请用 --model 指定模型名')
    print('LM Studio model:', model)
    results = []
    for item in suite:
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': '你是一个简洁、准确的中文 AI 助手。'},
                {'role': 'user', 'content': item['prompt']},
            ],
            'temperature': 0,
            'max_tokens': args.max_tokens,
            # Qwen3 may otherwise spend the whole budget in reasoning_content.
            'reasoning_effort': 'none',
        }
        r = requests.post(base + '/chat/completions', json=payload, timeout=180)
        r.raise_for_status()
        answer = r.json()['choices'][0]['message']['content'].strip()
        all_ok = all(x.lower() in answer.lower() for x in item.get('contains_all', []))
        choices = item.get('contains_any', [])
        any_ok = not choices or any(x.lower() in answer.lower() for x in choices)
        passed = all_ok and any_ok
        results.append({'id': item['id'], 'passed': passed, 'answer': answer})
        print(f"[{'PASS' if passed else 'FAIL'}] {item['id']}: {answer}")
    summary = {'model': model, 'passed': sum(x['passed'] for x in results), 'total': len(results), 'results': results}
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Score: {summary['passed']}/{summary['total']} ({summary['passed']/max(summary['total'],1):.1%})")
    print('Result:', args.output)


if __name__ == '__main__': main()
