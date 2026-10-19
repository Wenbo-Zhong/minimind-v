#!/usr/bin/env python3
"""Split the audited mixture into weighted, short SFT sources.

This does not alter the downloaded data or the previous curated mixture.
"""
import json, re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
INFILE = PROJECT / "dataset/curated_sft/train.jsonl"
OUT = PROJECT / "dataset/curated_sft/balanced"


def chars(row):
    return sum(len(str(x.get("content", ""))) for x in row.get("conversations", []))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    buckets = {"identity": [], "arithmetic": [], "chinese": [], "code": [], "tulu_short": []}
    seen = {k: set() for k in buckets}
    scanned = accepted = 0
    for line in INFILE.open(encoding="utf-8"):
        scanned += 1
        row = json.loads(line)
        source = row.get("source", "")
        conv = row.get("conversations", [])
        total = chars(row)
        if not conv or total > 3000:
            continue
        if source == "identity":
            bucket = "identity"
        elif source == "arithmetic":
            bucket = "arithmetic"
        elif source == "coig_code" or (source == "tulu" and re.search(r"code|python", str(row).lower())):
            bucket = "code"
        elif source == "tulu":
            # Keep only short instruction/math/IF rows; long chain-of-thought is
            # intentionally excluded for this 0.5B behavior-repair pass.
            if total > 2200:
                continue
            bucket = "tulu_short"
        else:
            bucket = "chinese"
        key = json.dumps(conv, ensure_ascii=False, sort_keys=True)
        if key in seen[bucket]:
            continue
        seen[bucket].add(key)
        buckets[bucket].append(row)
        accepted += 1

    # Add the complete small identity set as an explicit anchor; the previous
    # global deduplication reduced it too aggressively.
    identity = Path("/home/zwb/ai/data/identity/minimind_identity.jsonl")
    if identity.exists():
        for line in identity.open(encoding="utf-8"):
            row = json.loads(line)
            conv = row.get("conversations", row.get("messages", []))
            key = json.dumps(conv, ensure_ascii=False, sort_keys=True)
            if key not in seen["identity"]:
                buckets["identity"].append({"conversations": conv, "source": "identity"})
                seen["identity"].add(key)

    for name, rows in buckets.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"scanned": scanned, "accepted": accepted,
              "counts": {k: len(v) for k, v in buckets.items()},
              "chars": {k: sum(chars(x) for x in v) for k, v in buckets.items()}}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
