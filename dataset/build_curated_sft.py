#!/usr/bin/env python3
"""Build a small, auditable SFT mixture from the newly downloaded datasets.

The source files are never modified.  Rows are normalized to the format used by
LocalMultiSourceVLMDataset: {"conversations": [{"role", "content"}, ...]}.
"""
import argparse, hashlib, json, os, re
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/home/zwb/ai/data/sft")
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "dataset" / "curated_sft"


def clean(s):
    if isinstance(s, (list, dict)):
        s = json.dumps(s, ensure_ascii=False)
    s = str(s or "").replace("\x00", " ").strip()
    s = re.sub(r"[ \t]+", " ", s)
    return s


def valid(conv, max_chars=6000):
    if not conv or conv[0]["role"] not in ("user", "system"):
        return False
    if not any(x["role"] == "assistant" and len(x["content"]) >= 2 for x in conv):
        return False
    if sum(len(x["content"]) for x in conv) > max_chars:
        return False
    text = " ".join(x["content"] for x in conv)
    # Web dumps and malformed rows are particularly harmful to this 0.5B model.
    if len(text) < 8 or "<|im_start|>" in text or "<|im_end|>" in text:
        return False
    if re.search(r"(.{8,})\1\1", text):
        return False
    return True


def make_conv(messages):
    out = []
    for m in messages or []:
        role = {"human": "user", "gpt": "assistant"}.get(m.get("role", m.get("from", "")), m.get("role", m.get("from", "")))
        if role not in ("system", "user", "assistant"):
            continue
        content = clean(m.get("content", m.get("value", "")))
        if content:
            out.append({"role": role, "content": content})
    return out


def add(rows, seen, conv, source, stats, quota):
    if stats[source] >= quota or not valid(conv):
        return
    key = hashlib.sha1("\n".join(x["role"] + "\0" + x["content"] for x in conv).encode()).hexdigest()
    if key in seen:
        return
    seen.add(key)
    rows.append({"conversations": conv, "source": source})
    stats[source] += 1


def json_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        # COIG labels these files .json, but they are sometimes pretty-printed
        # objects concatenated one after another rather than one JSON array.
        text = path.read_text(encoding="utf-8")
        decoder, pos = json.JSONDecoder(), 0
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            value, pos = decoder.raw_decode(text, pos)
            if isinstance(value, list):
                yield from value
            else:
                yield value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUT / "train.jsonl"))
    ap.add_argument("--report", default=str(OUT / "report.json"))
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows, seen, stats, scanned = [], set(), {}, {}
    quotas = {"coig_exam": 8000, "coig_code": 10000, "coig_translated": 7000,
              "coig_values": 5000, "tulu": 25000, "infinity_core": 20000,
              "arithmetic": 12000, "identity": 6000}
    stats = {k: 0 for k in quotas}

    def scan(name, iterable, convert, quota=None):
        scanned[name] = 0
        for raw in iterable:
            scanned[name] += 1
            conv = convert(raw)
            add(rows, seen, conv, name, stats, quota or quotas[name])
            if stats[name] >= (quota or quotas[name]):
                break

    coig = ROOT / "COIG"
    scan("coig_exam", json_rows(coig / "exam_instructions.jsonl"), lambda r: make_conv([
        {"role": "user", "content": clean(r.get("textbox_q_instruction", "")) + "\n" + clean(r.get("textbox_q_context", "")) + "\n" + clean(r.get("textbox_question", ""))},
        {"role": "assistant", "content": clean(r.get("textbox_answer_analysis", "")) + "\n答案：" + clean(r.get("textbox_answer", ""))},]))
    scan("coig_code", json_rows(coig / "leetcode_instructions.jsonl"), lambda r: make_conv([
        {"role": "user", "content": clean(r.get("instruction", "")) + "\n" + clean(r.get("input", ""))},
        {"role": "assistant", "content": r.get("output", "")},]))
    scan("coig_translated", json_rows(coig / "translated_instructions.jsonl"), lambda r: make_conv([
        {"role": "user", "content": clean(r.get("trans_instruction", r.get("instruction", ""))) + "\n" + clean(r.get("trans_input", r.get("input", "")))},
        {"role": "assistant", "content": r.get("trans_output", r.get("output", ""))},]))
    for part in ("human_value_alignment_instructions_part1.json", "human_value_alignment_instructions_part2.json"):
        scan("coig_values", json_rows(coig / part), lambda r: make_conv([
            {"role": "user", "content": clean(r.get("instruction", "")) + ("\n" + clean(r.get("input", "")) if r.get("input") else "")},
            {"role": "assistant", "content": r.get("output", "")},]))

    # Tulu has long reasoning rows; cap each row before the existing loader's 768 tokens.
    tulu_files = sorted((ROOT / "tulu-3-sft-mixture" / "data").glob("*.parquet"))
    def tulu_iter():
        for f in tulu_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                for r in pf.read_row_group(i).to_pylist():
                    src = str(r.get("source", ""))
                    # Prefer code/math/instruction subsets; take a deterministic sample of others.
                    if not re.search(r"code|math|gsm|python|instruction|if", src, re.I):
                        ident = int(hashlib.md5(str(r.get("id", "")).encode()).hexdigest()[:8], 16)
                        if ident % 5:
                            continue
                    yield r
    scan("tulu", tulu_iter(), lambda r: make_conv(r.get("messages", [])))

    core = sorted((ROOT / "infinity-instruct" / "7M_core").glob("*.parquet"))
    def core_iter():
        for f in core:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                yield from pf.read_row_group(i).to_pylist()
    scan("infinity_core", core_iter(), lambda r: make_conv(r.get("conversations", r.get("messages", []))))

    # Keep only verified local narrow-task data, rather than the older noisy web SFT.
    for name, path in (("arithmetic", PROJECT / "dataset/generated/arithmetic_foundation.jsonl"),
                       ("arithmetic", PROJECT / "dataset/generated/arithmetic_curriculum.jsonl"),
                       ("identity", Path("/home/zwb/ai/data/identity/minimind_identity.jsonl"))):
        if path.exists():
            scan(name, json_rows(path), lambda r: make_conv(r.get("conversations", r.get("messages", []))))

    # Stable shuffle makes runs reproducible while mixing sources.
    import random
    random.Random(42).shuffle(rows)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"output": args.output, "rows": len(rows), "scanned": scanned,
              "accepted": stats, "duplicates_removed": sum(scanned.values()) - len(rows)}
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
