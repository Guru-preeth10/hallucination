#!/usr/bin/env python3
"""
Translate the reduced English/Hindi subset into Kannada and Tamil using a simple
HTTP-based translation endpoint.

This is intended for the small research subset created by
collect/build_final_4lang_subset.py, not the full benchmark.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List

TARGET_CONFIG = {
    "Kannada": {"code": "kn", "unique_id_suffix": "06"},
    "Tamil": {"code": "ta", "unique_id_suffix": "07"},
}


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def translate_text(text: str, target_language: str) -> str:
    encoded_text = urllib.parse.quote(text)
    url = (
        "https://translate.googleapis.com/translate_a/single?client=gtx"
        f"&sl=en&tl={TARGET_CONFIG[target_language]['code']}&dt=t&ie=UTF-8&oe=UTF-8&q={encoded_text}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload or not payload[0]:
        raise ValueError("Translation API returned no payload")
    translated_parts = [item[0] for item in payload[0] if item and item[0]]
    return " ".join(translated_parts).strip()


def build_target_unique_id(source_unique_id: str, suffix: str) -> str:
    source_unique_id = str(source_unique_id)
    if len(source_unique_id) >= 2:
        return source_unique_id[:-2] + suffix
    return source_unique_id + suffix


def translate_rows(source_rows: List[Dict], target_language: str, output_path: Path) -> List[Dict]:
    translated_rows: List[Dict] = []
    if output_path.exists():
        translated_rows = load_rows(output_path)

    completed_question_ids = {str(row["question_id"]) for row in translated_rows}

    for row in source_rows:
        if row.get("language") != "English":
            continue
        if str(row["question_id"]) in completed_question_ids:
            continue
        try:
            translated_question = translate_text(str(row["question"]), target_language)
            translated_expected = translate_text(str(row["expected"]), target_language)
        except Exception as exc:
            print(f"Failed for qid={row['question_id']}: {exc}", flush=True)
            continue

        translated_row = dict(row)
        translated_row["language"] = target_language
        translated_row["question"] = translated_question
        translated_row["expected"] = translated_expected
        translated_row["unique_id"] = build_target_unique_id(
            row.get("unique_id", ""),
            TARGET_CONFIG[target_language]["unique_id_suffix"],
        )
        translated_rows.append(translated_row)
        completed_question_ids.add(str(row["question_id"]))
        write_jsonl(output_path, translated_rows)

    return translated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate the reduced subset into Kannada or Tamil.")
    parser.add_argument(
        "--input",
        default="dataset/final_4lang_seed/source_subset_en_hi.jsonl",
        help="English/Hindi source subset JSONL.",
    )
    parser.add_argument(
        "--target-language",
        required=True,
        choices=sorted(TARGET_CONFIG.keys()),
        help="Target language for translation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to dataset/final_4lang_seed/translated_<code>.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("dataset/final_4lang_seed") / f"translated_{TARGET_CONFIG[args.target_language]['code']}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_rows = load_rows(input_path)
    translated_rows = translate_rows(source_rows, args.target_language, output_path)
    print(f"Translated rows written: {len(translated_rows)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
