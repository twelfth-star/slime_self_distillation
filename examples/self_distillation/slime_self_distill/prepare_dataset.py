from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Prepare dataset for slime self-distillation.")
    parser.add_argument("--input", type=str, required=True, help="Input jsonl path.")
    parser.add_argument("--output", type=str, required=True, help="Output jsonl path.")
    parser.add_argument("--teacher-prompt-key", type=str, default="teacher_prompt", help="Top-level teacher prompt key.")
    parser.add_argument("--metadata-key", type=str, default="metadata", help="Metadata field name.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get(args.metadata_key) or {}
            if args.teacher_prompt_key in row and args.teacher_prompt_key not in metadata:
                metadata[args.teacher_prompt_key] = row[args.teacher_prompt_key]
            row[args.metadata_key] = metadata
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

