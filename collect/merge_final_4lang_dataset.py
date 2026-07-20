#!/usr/bin/env python3
"""
Merge the English/Hindi source subset with Kannada and Tamil translated rows.

Expected inputs:
    - dataset/final_4lang_seed/source_subset_en_hi.jsonl
    - dataset/final_4lang_seed/translated_kn.jsonl
    - dataset/final_4lang_seed/translated_ta.jsonl

Outputs:
    - dataset/final_4lang_seed/final_4lang_5000.jsonl
    - dataset/final_4lang_seed/final_4lang_5000.summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

FINAL_LANGUAGE_ORDER = ["English", "Hindi", "Kannada", "Tamil"]


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_alignment(rows: List[Dict]) -> Dict:
    grouped: Dict[Tuple[str, int], Dict[str, Dict]] = defaultdict(dict)
    for row in rows:
        key = (row["category"], row["question_id"])
        grouped[key][row["language"]] = row

    missing = []
    for key, language_rows in grouped.items():
        absent = [lang for lang in FINAL_LANGUAGE_ORDER if lang not in language_rows]
        if absent:
            missing.append({"group": key, "missing_languages": absent})

    if missing:
        preview = missing[:10]
        raise ValueError(
            f"Final merged dataset is missing languages for {len(missing)} groups. "
            f"Examples: {preview}"
        )

    return grouped


def write_summary(path: Path, rows: List[Dict], grouped: Dict[Tuple[str, int], Dict[str, Dict]]) -> None:
    category_counts = Counter(row["category"] for row in rows)
    language_counts = Counter(row["language"] for row in rows)

    summary = {
        "total_rows": len(rows),
        "total_unique_groups": len(grouped),
        "languages": FINAL_LANGUAGE_ORDER,
        "category_counts": dict(sorted(category_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the final 4-language benchmark dataset.")
    parser.add_argument(
        "--source",
        default="dataset/final_4lang_seed/source_subset_en_hi.jsonl",
        help="English/Hindi source subset.",
    )
    parser.add_argument(
        "--kannada",
        default="dataset/final_4lang_seed/translated_kn.jsonl",
        help="Kannada translated rows JSONL.",
    )
    parser.add_argument(
        "--tamil",
        default="dataset/final_4lang_seed/translated_ta.jsonl",
        help="Tamil translated rows JSONL.",
    )
    parser.add_argument(
        "--output",
        default="dataset/final_4lang_seed/final_4lang_5000.jsonl",
        help="Merged final dataset JSONL path.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial merge: drop groups missing any language instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(load_rows(Path(args.source)))
    rows.extend(load_rows(Path(args.kannada)))
    rows.extend(load_rows(Path(args.tamil)))

    rows.sort(key=lambda row: (row["category"], row["question_id"], FINAL_LANGUAGE_ORDER.index(row["language"])))

    # Build a mapping of groups to language rows
    temp = defaultdict(dict)
    for row in rows:
        key = (row["category"], row["question_id"])
        temp[key][row["language"]] = row

    if args.allow_partial:
        # Keep only fully-complete groups that have all languages
        grouped = {k: v for k, v in temp.items() if all(lang in v for lang in FINAL_LANGUAGE_ORDER)}
        dropped = [k for k, v in temp.items() if not all(lang in v for lang in FINAL_LANGUAGE_ORDER)]
        if dropped:
            print(f"Dropping {len(dropped)} incomplete groups due to missing languages.")

        # Rebuild rows to include only the kept groups, preserving language order
        filtered_rows: List[Dict] = []
        for key in sorted(grouped.keys(), key=lambda k: (k[0], k[1])):
            language_rows = grouped[key]
            for lang in FINAL_LANGUAGE_ORDER:
                filtered_rows.append(language_rows[lang])
        rows = filtered_rows
    else:
        grouped = validate_alignment(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    write_summary(output_path.with_suffix(".summary.json"), rows, grouped)

    print(f"Merged rows: {len(rows)}")
    print(f"Unique groups: {len(grouped)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
