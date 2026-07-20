#!/usr/bin/env python3
"""
Retry translation for rows listed in a translation failure log.

This is useful after running collect/translate_final_4lang_language.py and
obtaining a *.failures.jsonl file. It re-runs only the failed question IDs,
appends successful rows to the target output file, and rewrites the failure
log so it only contains the rows that still fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect.translate_final_4lang_language import (  # noqa: E402
    TARGET_CONFIG,
    build_target_unique_id,
    generate_translation_payload,
    load_model_and_tokenizer,
    load_rows,
    write_jsonl,
)


def load_failure_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return load_rows(path)


def build_source_index(rows: List[Dict]) -> Dict[str, Dict]:
    return {str(row["question_id"]): row for row in rows if row.get("question_id") is not None}


def retry_failed_rows(
    source_rows: List[Dict],
    failure_rows: List[Dict],
    target_language: str,
    output_path: Path,
    failures_path: Path,
    model_name: str,
    quantization: str,
    max_new_tokens: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = load_rows(output_path) if output_path.exists() else []
    completed_question_ids = {str(row["question_id"]) for row in existing_rows}

    source_index = build_source_index(source_rows)
    model, tokenizer = load_model_and_tokenizer(model_name, quantization)

    remaining_failures: List[Dict] = []
    translated_rows = existing_rows[:]
    target_meta = TARGET_CONFIG[target_language]

    for failure_row in failure_rows:
        question_id = str(failure_row.get("question_id"))
        if question_id in completed_question_ids:
            continue

        source_row = source_index.get(question_id)
        if source_row is None:
            print(f"Skipping unknown question_id={question_id} from failure log", flush=True)
            remaining_failures.append(failure_row)
            continue

        print(
            f"Retrying qid={question_id} -> {target_language}",
            flush=True,
        )
        try:
            payload = generate_translation_payload(
                model=model,
                tokenizer=tokenizer,
                row=source_row,
                target_language=target_language,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            failure_row = dict(failure_row)
            failure_row["retry_error"] = str(exc)
            remaining_failures.append(failure_row)
            print(f"Retry failed for qid={question_id}: {exc}", flush=True)
            continue

        translated_row = dict(source_row)
        translated_row["language"] = target_language
        translated_row["question"] = payload["question"]
        translated_row["expected"] = payload["expected"]
        translated_row["unique_id"] = build_target_unique_id(
            source_row.get("unique_id", ""),
            target_meta["unique_id_suffix"],
        )

        translated_rows.append(translated_row)
        completed_question_ids.add(question_id)
        write_jsonl(output_path, translated_rows)

    write_jsonl(failures_path, remaining_failures)
    print(f"Updated output: {output_path}")
    print(f"Remaining failures written to: {failures_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry failed translations from a failure log.")
    parser.add_argument(
        "--input",
        default="dataset/final_4lang_seed/source_subset_en_hi.jsonl",
        help="Input English/Hindi source subset JSONL.",
    )
    parser.add_argument(
        "--target-language",
        required=True,
        choices=sorted(TARGET_CONFIG.keys()),
        help="Target language to retry.",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path. Defaults to dataset/final_4lang_seed/translated_<lang>.jsonl",
    )
    parser.add_argument(
        "--failures",
        help="Failure log path. Defaults to <output>.failures.jsonl",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model used for translation.",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="4bit",
        help="Model quantization mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generation tokens per retry translation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    source_rows = load_rows(input_path)

    output_path = (
        Path(args.output)
        if args.output
        else Path("dataset/final_4lang_seed")
        / f"translated_{TARGET_CONFIG[args.target_language]['code']}.jsonl"
    )
    failures_path = (
        Path(args.failures)
        if args.failures
        else output_path.with_suffix(output_path.suffix + ".failures.jsonl")
    )

    retry_failed_rows(
        source_rows=source_rows,
        failure_rows=load_failure_rows(failures_path),
        target_language=args.target_language,
        output_path=output_path,
        failures_path=failures_path,
        model_name=args.model,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
