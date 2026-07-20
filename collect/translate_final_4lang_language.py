#!/usr/bin/env python3
"""
Translate the English seed rows of the final 4-language subset into one target language.

Input:
    dataset/final_4lang_seed/source_subset_en_hi.jsonl

Output:
    a JSONL file containing translated rows for one target language
    (typically Kannada or Tamil), preserving:
        - question_id
        - category
        - domain
        - unique_id pattern (with a new language suffix)

This script only translates the English rows so that all final languages remain
aligned to the same source question_id groups.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TARGET_CONFIG = {
    "Kannada": {
        "language_name": "Kannada",
        "language_id": 6,
        "unique_id_suffix": "06",
        "code": "kn",
    },
    "Tamil": {
        "language_name": "Tamil",
        "language_id": 7,
        "unique_id_suffix": "07",
        "code": "ta",
    },
}


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


def supports_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def load_model_and_tokenizer(model_name: str, quantization: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto"}
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def build_messages(row: Dict, target_language: str) -> List[Dict[str, str]]:
    category = row["category"]
    expected = str(row["expected"])

    answer_rule = (
        "If the category is true_false_questions, keep the expected answer exactly as "
        "\"True\" or \"False\" in English. Do not translate those two labels."
    )

    prompt = f"""
You are translating benchmark data from English into {target_language}.

Translate only the natural-language content.
Preserve exactly:
- facts
- dates
- numbers
- named entities
- chronology/order
- option structure if present

{answer_rule}

Return ONLY valid JSON with these exact keys:
{{
  "question": "...",
  "expected": "..."
}}

Category: {category}
English question: {row["question"]}
English expected answer: {expected}
""".strip()

    return [
        {
            "role": "system",
            "content": "You are a careful multilingual benchmark translator that returns only JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def generate_text(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    if supports_chat_template(tokenizer):
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        plain = "\n\n".join(message["content"] for message in messages)
        model_inputs = tokenizer(plain, return_tensors="pt")

    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_tokens = model_inputs["input_ids"].shape[-1]
    response_tokens = generated[0][prompt_tokens:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


def extract_json_object(text: str) -> Dict[str, str]:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    if not text.startswith("{"):
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1)

    data = json.loads(text)
    if "question" not in data or "expected" not in data:
        raise ValueError("Missing question or expected in translation JSON")
    return {
        "question": str(data["question"]).strip(),
        "expected": str(data["expected"]).strip(),
    }


def build_target_unique_id(source_unique_id: str, suffix: str) -> str:
    source_unique_id = str(source_unique_id)
    if len(source_unique_id) >= 2:
        return source_unique_id[:-2] + suffix
    return source_unique_id + suffix


def translate_rows(
    rows: List[Dict],
    model_name: str,
    target_language: str,
    output_path: Path,
    quantization: str,
    max_new_tokens: int,
) -> None:
    target_meta = TARGET_CONFIG[target_language]
    english_rows = [row for row in rows if row["language"] == "English"]

    completed = set()
    existing_rows: List[Dict] = []
    if output_path.exists():
        existing_rows = load_rows(output_path)
        completed = {row["question_id"] for row in existing_rows}

    model, tokenizer = load_model_and_tokenizer(model_name, quantization)
    translated_rows = existing_rows[:]

    for idx, row in enumerate(english_rows, start=1):
        if row["question_id"] in completed:
            continue

        print(
            f"[{idx}/{len(english_rows)}] Translating qid={row['question_id']} "
            f"{row['category']} -> {target_language}",
            flush=True,
        )
        messages = build_messages(row, target_language)
        raw_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        payload = extract_json_object(raw_text)

        translated_row = dict(row)
        translated_row["language"] = target_language
        translated_row["question"] = payload["question"]
        translated_row["expected"] = payload["expected"]
        translated_row["unique_id"] = build_target_unique_id(
            row.get("unique_id", ""), target_meta["unique_id_suffix"]
        )

        translated_rows.append(translated_row)
        write_jsonl(output_path, translated_rows)
        time.sleep(0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate the English seed subset into Kannada or Tamil."
    )
    parser.add_argument(
        "--input",
        default="dataset/final_4lang_seed/source_subset_en_hi.jsonl",
        help="Input English/Hindi source subset JSONL.",
    )
    parser.add_argument(
        "--target-language",
        required=True,
        choices=sorted(TARGET_CONFIG.keys()),
        help="Target language to generate.",
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
        help="Model quantization mode for Colab.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generation tokens per translation.",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path. Defaults to dataset/final_4lang_seed/<language>.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    rows = load_rows(input_path)

    output_path = (
        Path(args.output)
        if args.output
        else Path("dataset/final_4lang_seed")
        / f"translated_{TARGET_CONFIG[args.target_language]['code']}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    translate_rows(
        rows=rows,
        model_name=args.model,
        target_language=args.target_language,
        output_path=output_path,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Saved translations to {output_path}")


if __name__ == "__main__":
    main()
