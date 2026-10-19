"""Generate a deterministic, validated arithmetic curriculum for MiniMind SFT."""

import json
import random
from pathlib import Path


SEED = 20260830
TARGET_SIZE = 50_000
OUTPUT = Path(__file__).resolve().parent / "dataset" / "generated" / "arithmetic_curriculum.jsonl"
FOUNDATION_OUTPUT = Path(__file__).resolve().parent / "dataset" / "generated" / "arithmetic_foundation.jsonl"
random.seed(SEED)

TEMPLATES = {
    "add": ["{a}+{b}等于几？", "{a}加{b}等于多少？", "计算：{a}+{b}", "请计算 {a}+{b} 的结果。"],
    "sub": ["{a}-{b}等于几？", "{a}减{b}等于多少？", "计算：{a}-{b}", "请计算 {a}-{b} 的结果。"],
    "mul": ["{a}×{b}等于几？", "{a}乘以{b}等于多少？", "计算：{a}×{b}", "请计算 {a}×{b} 的结果。"],
    "div": ["{a}÷{b}等于几？", "{a}除以{b}等于多少？", "计算：{a}÷{b}", "请计算 {a}÷{b} 的结果。"],
}
SYMBOL = {"add": "+", "sub": "-", "mul": "×", "div": "÷"}


def result(operation, a, b):
    if operation == "add":
        return a + b
    if operation == "sub":
        return a - b
    if operation == "mul":
        return a * b
    if operation == "div":
        assert b != 0 and a % b == 0
        return a // b
    raise ValueError(operation)


def sample(operation, a, b, template_index=0):
    answer = result(operation, a, b)
    question = TEMPLATES[operation][template_index % len(TEMPLATES[operation])].format(a=a, b=b)
    response = f"{a}{SYMBOL[operation]}{b}={answer}。"
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": response},
        ],
        "arithmetic": {"operation": operation, "a": a, "b": b, "answer": answer},
    }


def build():
    rows = []

    # Foundation: every small pair appears in every surface form.  This fixes
    # the previous generator's complete absence of examples such as 1+1/10-5.
    for operation in ("add", "sub"):
        for a in range(21):
            for b in range(21):
                if operation == "sub" and b > a:
                    continue
                for template_index in range(len(TEMPLATES[operation])):
                    rows.append(sample(operation, a, b, template_index))

    # Complete coverage through 100, one varied phrasing per pair.
    for a in range(101):
        for b in range(101):
            rows.append(sample("add", a, b, a * 101 + b))
            if b <= a:
                rows.append(sample("sub", a, b, a * 101 + b))

    # Full multiplication table and its inverse division facts.
    for a in range(13):
        for b in range(13):
            for template_index in range(2):
                rows.append(sample("mul", a, b, template_index))
            if b > 0:
                for template_index in range(2):
                    rows.append(sample("div", a * b, b, template_index))

    # Medium-range practice. Addition/subtraction dominate deliberately.
    while len(rows) < 43_000:
        operation = random.choice(("add", "sub"))
        a, b = random.randint(0, 999), random.randint(0, 999)
        if operation == "sub" and b > a:
            a, b = b, a
        rows.append(sample(operation, a, b, random.randrange(4)))

    while len(rows) < TARGET_SIZE:
        if random.random() < 0.55:
            rows.append(sample("mul", random.randint(0, 99), random.randint(0, 12), random.randrange(4)))
        else:
            divisor = random.randint(1, 12)
            quotient = random.randint(0, 99)
            rows.append(sample("div", divisor * quotient, divisor, random.randrange(4)))

    random.shuffle(rows)
    return rows


def validate(rows):
    assert len(rows) == TARGET_SIZE
    facts = set()
    for index, row in enumerate(rows):
        meta = row["arithmetic"]
        expected = result(meta["operation"], meta["a"], meta["b"])
        assert meta["answer"] == expected, (index, row)
        response = row["conversations"][1]["value"]
        assert response == f'{meta["a"]}{SYMBOL[meta["operation"]]}{meta["b"]}={expected}。', (index, row)
        facts.add((meta["operation"], meta["a"], meta["b"]))
    for required in (("add", 1, 1), ("add", 2, 3), ("sub", 10, 5),
                     ("add", 17, 28), ("mul", 9, 7), ("div", 56, 8)):
        assert required in facts, f"missing required fact: {required}"


def main():
    rows = build()
    validate(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    foundation = []
    for row in rows:
        meta = row["arithmetic"]
        if ((meta["operation"] in ("add", "sub") and meta["a"] <= 20 and meta["b"] <= 20)
                or (meta["operation"] in ("mul", "div") and meta["a"] <= 144 and meta["b"] <= 12)):
            foundation.append(row)
    with FOUNDATION_OUTPUT.open("w", encoding="utf-8") as f:
        for row in foundation:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"generated and validated {len(rows)} rows: {OUTPUT}")
    print(f"generated foundation subset {len(foundation)} rows: {FOUNDATION_OUTPUT}")


if __name__ == "__main__":
    main()
