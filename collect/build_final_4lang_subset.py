#!/usr/bin/env python3
"""
Build the aligned source subset for the final 4-language benchmark.

The target benchmark shape is:
    1250 unique (category, question_id) groups
    x 4 languages (English, Hindi, Kannada, Tamil)
    = 5000 total rows

This script builds the source side of that benchmark from the existing
10K BHRAM-IL dataset by selecting:
    - all safe-category groups from five categories
    - a deterministic subset from true/false

The output rows keep the original dataset schema and contain only the
existing English and Hindi source rows. Kannada and Tamil can later be
translated from this aligned source subset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SAFE_CATEGORY_QUOTAS: Dict[str, int] = {
    "factual_questions": 390,
    "chrono_questions": 196,
    "indian_questions": 227,
    "maths_questions": 175,
    "reasoning_questions": 141,
    "true_false_questions": 121,
}

SOURCE_LANGUAGES = ("English", "Hindi")


def build_quotas(selected_categories: Iterable[str], max_groups_per_category: int | None) -> Dict[str, int]:
    if max_groups_per_category is None:
        return {category: SAFE_CATEGORY_QUOTAS[category] for category in selected_categories}
    return {category: max_groups_per_category for category in selected_categories}


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_group_index(rows: Iterable[Dict]) -> Dict[Tuple[str, int], Dict[str, Dict]]:
    grouped: Dict[Tuple[str, int], Dict[str, Dict]] = defaultdict(dict)
    for row in rows:
        key = (row["category"], row["question_id"])
        grouped[key][row["language"]] = row
    return grouped


def select_groups(
    grouped_rows: Dict[Tuple[str, int], Dict[str, Dict]],
    quotas: Dict[str, int],
    required_languages: Iterable[str],
) -> List[Tuple[str, int]]:
    required_languages = tuple(required_languages)
    selected: List[Tuple[str, int]] = []

    for category, quota in quotas.items():
        candidates = [
            key
            for key, languages in grouped_rows.items()
            if key[0] == category and all(lang in languages for lang in required_languages)
        ]
        candidates.sort(key=lambda item: item[1])

        if len(candidates) < quota:
            raise ValueError(
                f"Category {category} has only {len(candidates)} eligible groups, "
                f"but quota is {quota}."
            )

        selected.extend(candidates[:quota])

    return selected


def build_source_subset_rows(
    grouped_rows: Dict[Tuple[str, int], Dict[str, Dict]],
    selected_groups: Iterable[Tuple[str, int]],
    languages: Iterable[str],
) -> List[Dict]:
    subset: List[Dict] = []
    for key in selected_groups:
        for language in languages:
            subset.append(grouped_rows[key][language])
    return subset


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_selected_group_manifest(
    path: Path,
    grouped_rows: Dict[Tuple[str, int], Dict[str, Dict]],
    selected_groups: Iterable[Tuple[str, int]],
) -> None:
    manifest_rows = []
    for category, question_id in selected_groups:
        languages_present = sorted(grouped_rows[(category, question_id)].keys())
        manifest_rows.append(
            {
                "category": category,
                "question_id": question_id,
                "languages_present": languages_present,
                "group_key": f"{category}:{question_id}",
            }
        )
    write_jsonl(path, manifest_rows)


def write_summary(
    path: Path,
    selected_groups: List[Tuple[str, int]],
    subset_rows: List[Dict],
    source_languages: Iterable[str],
    quotas: Dict[str, int],
) -> None:
    group_counts = Counter(category for category, _ in selected_groups)
    row_counts = Counter((row["category"], row["language"]) for row in subset_rows)

    summary = {
        "target_total_unique_groups": sum(quotas.values()),
        "actual_total_unique_groups": len(selected_groups),
        "source_languages": list(source_languages),
        "target_total_source_rows": len(selected_groups) * len(tuple(source_languages)),
        "actual_total_source_rows": len(subset_rows),
        "category_group_counts": dict(group_counts),
        "category_language_row_counts": {
            f"{category}::{language}": count
            for (category, language), count in sorted(row_counts.items())
        },
        "selection_quotas": quotas,
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the aligned English/Hindi source subset for the final 4-language benchmark."
    )
    parser.add_argument(
        "--input",
        default="dataset/BHRAM_IL_10K/dataset_10k.jsonl",
        help="Input benchmark JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/final_4lang_seed",
        help="Directory where subset artifacts will be written.",
    )
    parser.add_argument(
        "--categories",
        default=",".join(SAFE_CATEGORY_QUOTAS.keys()),
        help="Comma-separated list of categories to include, e.g. chrono_questions,factual_questions,indian_questions",
    )
    parser.add_argument(
        "--max-groups-per-category",
        type=int,
        default=None,
        help="Optional cap for how many groups to select from each category.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    grouped_rows = build_group_index(rows)

    selected_categories = [cat.strip() for cat in args.categories.split(",") if cat.strip()]
    if not selected_categories:
        raise ValueError("At least one category must be provided.")

    quotas = build_quotas(selected_categories, args.max_groups_per_category)
    for category in selected_categories:
        if category not in SAFE_CATEGORY_QUOTAS:
            raise ValueError(f"Unsupported category: {category}")

    selected_groups = select_groups(
        grouped_rows=grouped_rows,
        quotas=quotas,
        required_languages=SOURCE_LANGUAGES,
    )
    subset_rows = build_source_subset_rows(
        grouped_rows=grouped_rows,
        selected_groups=selected_groups,
        languages=SOURCE_LANGUAGES,
    )

    selected_groups.sort(key=lambda item: (item[0], item[1]))
    subset_rows.sort(key=lambda row: (row["category"], row["question_id"], row["language"]))

    write_jsonl(output_dir / "source_subset_en_hi.jsonl", subset_rows)
    write_selected_group_manifest(
        output_dir / "selected_groups.jsonl",
        grouped_rows=grouped_rows,
        selected_groups=selected_groups,
    )
    write_summary(
        output_dir / "summary.json",
        selected_groups=selected_groups,
        subset_rows=subset_rows,
        source_languages=SOURCE_LANGUAGES,
        quotas=quotas,
    )

    print(f"Selected groups: {len(selected_groups)}")
    print(f"Source rows written: {len(subset_rows)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
