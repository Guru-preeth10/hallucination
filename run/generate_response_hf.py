#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate BHRAM-IL model responses with Hugging Face Transformers.

This script is intended for Google Colab or other GPU environments where
running Ollama is inconvenient. It reads the benchmark JSONL files directly
and produces evaluator-compatible JSONL output.
"""

import argparse
import json
import os
import platform
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

###############################################################################
# Constants

PROMPTS_DIR = Path("prompts")
DEFAULT_DATASET = Path("../dataset/BHRAM_IL_10K/dataset_10k.jsonl")
DEFAULT_OUTPUT_DIR = Path("output")

SYSTEM_PROMPT_PREFIX = "system"
OUTPUT_FORMAT_PROMPT_PREFIX = "output_format"

LANGUAGES = ["english", "gujarati", "hindi", "marathi", "odia", "kannada", "tamil"]

LANGUAGE_ID_MAP = {
    "english": 1,
    "hindi": 2,
    "gujarati": 3,
    "marathi": 4,
    "odia": 5,
    "kannada": 6,
    "tamil": 7,
}

CATEGORY_ID_MAP = {
    "factual_questions": 1,
    "indian_questions": 2,
    "maths_questions": 3,
    "ner_questions": 4,
    "reasoning_questions": 5,
    "semantically_incorrect_questions": 6,
    "chrono_questions": 7,
    "true_false_questions": 8,
    "word_ordering_questions": 9,
    "summarization_questions": 10,
}

###############################################################################


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "model"


def split_model_name(model_name: str) -> Dict[str, str]:
    leaf_name = model_name.split("/")[-1]
    if "-" in leaf_name:
        base, variant = leaf_name.rsplit("-", 1)
    else:
        base, variant = leaf_name, ""
    return {
        "model": model_name,
        "model_name": base,
        "model_variant": variant,
    }


def normalize_row(raw_item: Dict) -> Dict:
    question_id = raw_item.get("question_id", raw_item.get("ID"))
    language = raw_item.get("language", raw_item.get("Language"))
    category = raw_item.get("category", raw_item.get("Category"))
    question = raw_item.get("question", raw_item.get("Question"))
    expected = raw_item.get("expected", raw_item.get("Answer"))
    domain = raw_item.get("domain", raw_item.get("Domain"))
    unique_id = raw_item.get("unique_id", "")

    return {
        "question_id": question_id,
        "language": language,
        "category": category,
        "question": question,
        "expected": expected,
        "domain": domain,
        "unique_id": unique_id,
    }


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(normalize_row(json.loads(line)))
    return rows


def read_prompt_templates(categories: Iterable[str]) -> Dict:
    templates = defaultdict(dict)

    for language in LANGUAGES:
        prompts_dir = PROMPTS_DIR / language
        system_prompt_file = prompts_dir / f"{SYSTEM_PROMPT_PREFIX}.txt"
        output_format_file = prompts_dir / f"{OUTPUT_FORMAT_PROMPT_PREFIX}.txt"

        templates[language]["system"] = (
            system_prompt_file.read_text(encoding="utf-8")
            if system_prompt_file.is_file()
            else ""
        )
        templates[language]["output"] = (
            output_format_file.read_text(encoding="utf-8")
            if output_format_file.is_file()
            else ""
        )

        templates[language]["user"] = {}
        for category in categories:
            template_file = prompts_dir / f"{category}.prompt.txt"
            if template_file.is_file():
                templates[language]["user"][category] = template_file.read_text(
                    encoding="utf-8"
                )

    # Fill missing templates for Kannada/Tamil with English fallbacks
    for language in LANGUAGES:
        if language == "english":
            continue
        if not templates[language]["system"]:
            templates[language]["system"] = templates["english"]["system"]
        if not templates[language]["output"]:
            templates[language]["output"] = templates["english"]["output"]
        for category in categories:
            if category not in templates[language]["user"]:
                templates[language]["user"][category] = templates["english"]["user"].get(
                    category, ""
                )

    return templates


def choose_prompt_language(question_language: str, prompt_mode: str) -> str:
    normalized = str(question_language).strip().lower()
    if prompt_mode == "english":
        return "english"
    if normalized not in LANGUAGES:
        print(
            f"Warning: Unsupported native prompt language {question_language}. Falling back to English."
        )
        return "english"
    return normalized


def format_user_prompt(
    question: str,
    output_format: str,
    user_prompt_template: str,
) -> str:
    return user_prompt_template.format(
        question=str(question),
        output_format=output_format,
    )


def supports_chat_template(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return bool(template)


def build_model_inputs(tokenizer, system_prompt: str, prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if supports_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    plain_prompt = (
        f"{system_prompt}\n\n{prompt}".strip()
        if system_prompt else prompt
    )
    return tokenizer(plain_prompt, return_tensors="pt")


def load_model_and_tokenizer(
    model_name: str,
    quantization: str,
    torch_dtype: str,
    trust_remote_code: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }

    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
        )
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_kwargs["torch_dtype"] = dtype_map[torch_dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    output_format: str,
    user_prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict:
    prompt = format_user_prompt(question, output_format, user_prompt_template)
    model_inputs = build_model_inputs(tokenizer, system_prompt, prompt)
    input_device = next(model.parameters()).device
    model_inputs = {key: value.to(input_device) for key, value in model_inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    start_time = time.perf_counter_ns()
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            **generation_kwargs,
        )
    total_duration = time.perf_counter_ns() - start_time

    prompt_tokens = model_inputs["input_ids"].shape[-1]
    output_tokens = generated.shape[-1] - prompt_tokens
    response_tokens = generated[0][prompt_tokens:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    return {
        "content": response_text,
        "metadata": {
            "done": "True",
            "done_reason": "stop",
            "total_duration": str(total_duration),
            "load_duration": "0",
            "prompt_eval_count": str(prompt_tokens),
            "prompt_eval_duration": "0",
            "eval_count": str(max(output_tokens, 0)),
            "eval_duration": str(total_duration),
        },
    }


def build_output_record(
    item: Dict,
    response: Dict,
    model_name: str,
    prompt_type: str,
) -> Dict:
    language_normalized = str(item["language"]).strip().lower()
    metadata = split_model_name(model_name)

    return {
        "question_id": item["question_id"],
        "language": item["language"],
        "category": item["category"],
        "question": item["question"],
        "expected": str(item["expected"]),
        "response": response,
        "domain": item.get("domain"),
        "unique_id": item.get("unique_id", ""),
        "category_id": CATEGORY_ID_MAP.get(item["category"], -1),
        "domain_id": -1,
        "language_id": LANGUAGE_ID_MAP.get(language_normalized, -1),
        "local_question_id": item["question_id"],
        "machine_name": platform.node() or "colab",
        "prompt_type": prompt_type,
        **metadata,
    }


def load_existing_records(output_file: Path) -> set:
    if not output_file.exists():
        return set()

    completed = set()
    with open(output_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add((row.get("question_id"), row.get("language")))
    return completed


def filter_rows(
    rows: List[Dict],
    categories: Optional[List[str]],
    languages: Optional[List[str]],
    limit: Optional[int],
) -> List[Dict]:
    selected = []
    wanted_categories = set(categories or [])
    wanted_languages = {language.lower() for language in (languages or [])}

    for row in rows:
        if wanted_categories and row["category"] not in wanted_categories:
            continue
        if wanted_languages and str(row["language"]).lower() not in wanted_languages:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break

    return selected


def default_output_file(
    output_dir: Path,
    model_name: str,
    dataset_path: Path,
    prompt_mode: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = slugify(model_name.split("/")[-1])
    dataset_slug = slugify(dataset_path.stem)
    return output_dir / f"output.{dataset_slug}.{model_slug}.{prompt_mode}.jsonl"


def run_experiment(
    dataset_path: Path,
    output_file: Path,
    model_name: str,
    prompt_mode: str,
    categories: Optional[List[str]],
    languages: Optional[List[str]],
    limit: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    quantization: str,
    torch_dtype: str,
    trust_remote_code: bool,
):
    rows = read_jsonl(dataset_path)
    selected_rows = filter_rows(rows, categories, languages, limit)
    category_names = sorted({row["category"] for row in selected_rows})
    templates = read_prompt_templates(category_names)
    existing = load_existing_records(output_file)

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        quantization=quantization,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    print(f"Loaded {len(selected_rows)} benchmark rows from {dataset_path}")
    print(f"Writing outputs to {output_file}")

    with open(output_file, "a", encoding="utf-8") as handle:
        for index, item in enumerate(selected_rows, start=1):
            row_key = (item["question_id"], item["language"])
            if row_key in existing:
                continue

            prompt_language = choose_prompt_language(item["language"], prompt_mode)
            category = item["category"]
            if category not in templates[prompt_language]["user"]:
                raise KeyError(
                    f"Missing prompt template for {prompt_language}/{category}"
                )

            print(
                f"[{index}/{len(selected_rows)}] "
                f"{item['category']} | {item['language']} | qid={item['question_id']}",
                flush=True,
            )

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                question=item["question"],
                system_prompt=templates[prompt_language]["system"],
                output_format=templates[prompt_language]["output"],
                user_prompt_template=templates[prompt_language]["user"][category],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            output = build_output_record(
                item=item,
                response=response,
                model_name=model_name,
                prompt_type=prompt_mode,
            )
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate BHRAM-IL responses with Hugging Face models"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Hugging Face model id, e.g. google/gemma-2-2b-it",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSONL file. Defaults to run/output/output.<dataset>.<model>.<prompt>.jsonl",
    )
    parser.add_argument(
        "-c",
        "--category",
        nargs="+",
        help="One or more categories to run",
    )
    parser.add_argument(
        "-l",
        "--language",
        nargs="+",
        help="One or more languages to run, e.g. English Hindi",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["english", "native"],
        default="english",
        help="Use English prompts for all rows or prompts in each row's language",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of rows to run after filtering",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per answer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p used when temperature > 0",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="4bit",
        help="Quantization mode for Colab GPUs",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype when quantization is disabled",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when the chosen model requires it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    dataset_path = Path(options.dataset)
    output_file = (
        Path(options.output)
        if options.output
        else default_output_file(
            output_dir=DEFAULT_OUTPUT_DIR,
            model_name=options.model,
            dataset_path=dataset_path,
            prompt_mode=options.prompt_mode,
        )
    )

    run_experiment(
        dataset_path=dataset_path,
        output_file=output_file,
        model_name=options.model,
        prompt_mode=options.prompt_mode,
        categories=options.category,
        languages=options.language,
        limit=options.limit,
        max_new_tokens=options.max_new_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        quantization=options.quantization,
        torch_dtype=options.torch_dtype,
        trust_remote_code=options.trust_remote_code,
    )
