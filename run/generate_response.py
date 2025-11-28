#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:33:16 2025

@author: Hrishikesh Terdalkar
"""

###############################################################################

import os
import json
import time
from pathlib import Path
from collections import defaultdict

import ollama
import pandas as pd

###############################################################################
# Names of Excel Fields

KEY_QUESTION_ID = "ID"
KEY_CATEGORY = "Category"
KEY_LANGUAGE = "Language"
KEY_QUESTION = "Question"
KEY_EXPECTED = "Answer"

###############################################################################

PROMPTS_DIR = Path("prompts")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

SYSTEM_PROMPT_PREFIX = "system"
OUTPUT_FORMAT_PROMPT_PREFIX = "output_format"

LANGUAGES = ["english", "gujarati", "hindi", "marathi", "odia"]

###############################################################################


def query_ollama(model, question, system_prompt, output_format, user_prompt_template):
    try:
        prompt = user_prompt_template.format(question=question, output_format=output_format)
        print(prompt)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "seed": 42,
                "temperature": 0.4,
                "num_predict": 1000  # max 10000 tokens to be generated
            },
        )
        response_object = {
            "content": response["message"]["content"],
            "metadata": {
                metadata_key: str(response.get(metadata_key))
                for metadata_key in [
                    "done", "done_reason",
                    "total_duration", "load_duration",
                    "prompt_eval_count", "prompt_eval_duration",
                    "eval_count", "eval_duration",
                ]
            }
        }
        return response_object

    except Exception as e:
        print(f"Error for question '{question[:30]}...': {e}", flush=True)
        return None


def read_prompt_templates(categories):
    templates = defaultdict(dict)

    for language in LANGUAGES:
        prompts_dir = PROMPTS_DIR / language

        system_prompt_file = prompts_dir / f"{SYSTEM_PROMPT_PREFIX}.txt"
        output_format_file = prompts_dir / f"{OUTPUT_FORMAT_PROMPT_PREFIX}.txt"

        if system_prompt_file.is_file():
            templates[language]["system"] = system_prompt_file.read_text(encoding="utf-8")

        if output_format_file.is_file():
            templates[language]["output"] = output_format_file.read_text(encoding="utf-8")

        templates[language]["user"] = {}
        for category in categories:
            user_prompt_template_file = prompts_dir / f"{category}.prompt.txt"
            if user_prompt_template_file.is_file():
                templates[language]["user"][category] = user_prompt_template_file.read_text(encoding="utf-8")

    return templates


def run_experiment(
    model: str,
    question_category: str,
    use_native_prompts: bool,
    templates: dict,
    use_text_completion_mode: bool,
):
    print(f"Processing {question_category} ...")
    # Input File
    input_file = DATA_DIR / f"{question_category}.xlsx"
    df = pd.read_excel(input_file, parse_dates=False, na_filter=None)

    # Output File
    native = ".native" if use_native_prompts else ""
    output_file = OUTPUT_DIR / f"output.{question_category}.{model}{native}.jsonl"

    if KEY_QUESTION not in df.columns:
        raise ValueError(f"Excel must have a '{KEY_QUESTION}' column")

    # If OUTPUT_FILE exists, load previous progress
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            content = [json.loads(line) for line in f.readlines()]
            existing = set((r["question_id"], r["language"]) for r in content)
    else:
        existing = set()

    for i, (idx, row) in enumerate(df.iterrows(), start=0):
        question_id = row[KEY_QUESTION_ID]
        language = row[KEY_LANGUAGE]
        if (question_id, language) in existing:
            continue

        _language = language.lower()

        if use_native_prompts:
            if _language == "english":
                continue
        else:
            _language = "english"

        if _language not in LANGUAGES:
            print(row)
        system_prompt = templates[_language]["system"]
        output_format = templates[_language]["output"]
        user_prompt_template = templates[_language]["user"][question_category]

        if use_text_completion_mode:
            user_prompt_template = (
                user_prompt_template.replace(
                    "Question: {question}", ""
                ).replace(
                    "Sentence: {question}", ""
                ).replace(
                    "Output Format:", ""
                ).replace(
                    "{output_format}", ""
                )
            )
            user_prompt_template = (
                "\n### Instruction:\n\n"
                f"{system_prompt}\n\n"
                "Response Format: {output_format}\n"
                f"{user_prompt_template}\n"
                "### Input:\n\n"
                "{question}\n\n"
                "### Response:\n"
            )
            system_prompt = None

        response = query_ollama(
            model=model,
            question=str(row[KEY_QUESTION]),
            system_prompt=system_prompt,
            output_format=output_format,
            user_prompt_template=user_prompt_template,
        )
        if response is not None:
            output = {
                "question_id": row[KEY_QUESTION_ID],
                "language": row.get(KEY_LANGUAGE),
                "category": row.get(KEY_CATEGORY),
                "question": row.get(KEY_QUESTION),
                "expected": str(row.get(KEY_EXPECTED)),
                "response": response
            }

            # save after every question
            output_json = json.dumps(output, ensure_ascii=False)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{output_json}\n")

    print(f"Output saved to {output_file}", flush=True)

###############################################################################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Run Experiment")
    parser.add_argument("-m", "--model", required=True, help="Ollama Model Identifier")
    parser.add_argument("-c", "--category", required=True, nargs="+", help="Categories")
    parser.add_argument("-n", "--native", action="store_true", help="Use native prompts")
    parser.add_argument("-t", "--text-completion", action="store_true", help="Use text-completion prompt")

    options = parser.parse_args()

    templates = read_prompt_templates(options.category)
    for category in options.category:
        run_experiment(
            options.model,
            category,
            options.native,
            templates=templates,
            use_text_completion_mode=options.text_completion,
        )

###############################################################################
