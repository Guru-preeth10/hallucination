#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Framework for BHRAM-IL

Created on Thu Oct 10 20:43:17 2025

Author: Hrishikesh Terdalkar
"""

###############################################################################

import os
import re
import json
import string
import logging
import unicodedata
import argparse
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Third-party
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
import fasttext
from thefuzz import fuzz
import torch
import bert_score
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import scipy.stats

# from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

from indic_transliteration import sanscript
from indic_transliteration.detect import detect

###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Helper Utilities


class FastText:
    MODEL_PATH = "fasttext"
    DIMENSION = 100
    MODELS = {}

    def __init__(self):
        for language in ["en", "gu", "hi", "mr", "or"]:
            try:
                self.MODELS[language] = fasttext.load_model(
                    os.path.join(self.MODEL_PATH, f"cc.{language}.{self.DIMENSION}.bin")
                )
            except Exception:
                print(f"Model could not be loaded: {language}")

    def similarity(self, s1: str, s2: str, lang: str = "en"):
        v1 = self.MODELS.get(lang, self.MODELS["en"]).get_sentence_vector(s1)
        v2 = self.MODELS.get(lang, self.MODELS["en"]).get_sentence_vector(s2)

        if np.all(v1 == 0) or np.all(v2 == 0):
            # Assign 0 similarity if either vector is zero (e.g., from empty string)
            similarity = 0.0
        else:
            similarity = 1 - cosine(v1, v2)

        return float(similarity)

# --------------------------------------------------------------------------- #

FASTTEXT = FastText()

###############################################################################


class MetricHelper:
    """Complete metric computation with all original metrics."""

    @staticmethod
    def exact_match(a: str, b: str) -> bool:
        return a == b

    @staticmethod
    def fuzzy_score(a: str, b: str) -> float:
        """Raw fuzzy similarity score (0-100)."""
        return fuzz.ratio(a.strip().lower(), b.strip().lower())

    @staticmethod
    def fuzzy_match(pred: str, ref: str, threshold: int = 85) -> bool:
        """Fuzzy string match using similarity threshold."""
        return MetricHelper.fuzzy_score(pred, ref) >= threshold

    @staticmethod
    def semantic_similarity(s1: str, s2: str, lang: str) -> float:
        """Return semantic similarity."""
        return FASTTEXT.similarity(s1, s2, lang)


    @staticmethod
    def word_level_metrics(s1: str, s2: str) -> Dict[str, float]:
        """Basic word-level precision, recall, F1."""
        pred_norm = BaseEvaluator.full_normalize(s1)
        ref_norm = BaseEvaluator.full_normalize(s2)

        pred_words = set(pred_norm.split())
        ref_words = set(ref_norm.split())

        if not pred_words and not ref_words:
            return {"word_precision": 0.0, "word_recall": 0.0, "word_f1": 0.0}

        if not pred_words:
            return {"word_precision": 0.0, "word_recall": 0.0, "word_f1": 0.0}

        if not ref_words:
            return {"word_precision": 0.0, "word_recall": 0.0, "word_f1": 0.0}

        common = pred_words & ref_words
        precision = len(common) / len(pred_words)
        recall = len(common) / len(ref_words)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "word_precision": precision,
            "word_recall": recall,
            "word_f1": f1
        }

    @staticmethod
    def char_level_metrics(s1: str, s2: str) -> Dict[str, float]:
        """Basic character-level precision, recall, F1 (ignoring spaces)."""
        pred_norm = BaseEvaluator.full_normalize(s1).replace(" ", "")
        ref_norm = BaseEvaluator.full_normalize(s2).replace(" ", "")

        if not pred_norm or not ref_norm:
            return {
                "char_precision": 0.0,
                "char_recall": 0.0,
                "char_f1": 0.0
            }

        pred_chars = set(pred_norm)
        ref_chars = set(ref_norm)

        common = pred_chars & ref_chars
        precision = len(common) / len(pred_chars) if pred_chars else 0.0
        recall = len(common) / len(ref_chars) if ref_chars else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "char_precision": precision,
            "char_recall": recall,
            "char_f1": f1
        }

    @staticmethod
    def rouge_scores(pred: str, ref: str) -> Dict[str, float]:
        """ROUGE scores for single sentence pairs."""
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        scores = scorer.score(ref, pred)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    @staticmethod
    def bleu_score(pred: str, ref: str) -> float:
        """BLEU for single sentence pairs."""
        # TODO: May be need GLEU here
        try:
            if not pred.strip() or not ref.strip():
                return 0.0

            score = corpus_bleu(
                [pred],
                [[ref]],
                tokenize="intl",
                lowercase=True,
                force=True,
            )
            return score.score
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return 0.0

    @staticmethod
    def multi_metric_similarity(pred: str, ref: str, pred_lang: str, ref_lang: str) -> float:
        """Compute multiple similarity metrics between two texts, handling cross-lingual cases."""

        fuzzy = MetricHelper.fuzzy_score(pred, ref) / 100.0
        word_metrics = MetricHelper.word_level_metrics(pred, ref)
        char_metrics = MetricHelper.char_level_metrics(pred, ref)
        semantic_score = MetricHelper.semantic_similarity(pred, ref, pred_lang)

        # Weighted average of multiple metrics
        semantic_score = (
            fuzzy * 0.1 +
            word_metrics["word_f1"] * 0.1 +
            char_metrics["char_f1"] * 0.1 +
            semantic_score * 0.7
        )
        return semantic_score

    @staticmethod
    def kendalls_tau(pred_sequence: List[str], ref_sequence: List[str]) -> Dict[str, float]:
        """Kendall's Tau for sequence similarity."""
        if len(pred_sequence) != len(ref_sequence) or len(pred_sequence) < 2:
            return {"kendall_tau": 0.0, "p_value": 1.0}

        try:
            # Create mapping from reference to predicted order
            ref_to_index = {item: idx for idx, item in enumerate(ref_sequence)}
            pred_indices = [ref_to_index.get(item, -1) for item in pred_sequence]

            # Remove items not found in reference
            valid_indices = [idx for idx in pred_indices if idx != -1]
            if len(valid_indices) < 2:
                return {"kendall_tau": 0.0, "p_value": 1.0}

            # Calculate Kendall's Tau
            tau_result = scipy.stats.kendalltau(range(len(valid_indices)), valid_indices)

            if tau_result and not np.isnan(tau_result.correlation):
                return {"kendall_tau": tau_result.correlation, "p_value": tau_result.pvalue}
            else:
                return {"kendall_tau": 0.0, "p_value": 1.0}

        except Exception as e:
            logger.warning(f"Kendall's Tau computation failed: {e}")
            return {"kendall_tau": 0.0, "p_value": 1.0}

###############################################################################
# Data structures

@dataclass
class EvaluationResult:

    question_id: str
    language: str
    category: str
    predicted: str
    expected: str
    match: Dict[str, bool]
    scores: Dict[str, float]
    metrics: Dict[str, Any]
    language_hallucination: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkipItem(Exception):
    """Raised when a particular item should be ignored."""

    pass


###############################################################################
# Base Evaluator


class BaseEvaluator(ABC):
    """Common utilities shared by every concrete evaluator."""

    # Configuration constants
    LANGUAGE_MAP = {
        "english": "en",
        "hindi": "hi",
        "gujarati": "gu",
        "marathi": "mr",
        "odia": "or",
        "kannada": "kn",
        "tamil": "ta",
    }
    VALID_LANGUAGES = set(LANGUAGE_MAP.values())

    NUMERAL_MAP = {
        "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
        "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
        "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
        "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
        "୦": "0", "୧": "1", "୨": "2", "୩": "3", "୪": "4",
        "୫": "5", "୬": "6", "୭": "7", "୮": "8", "୯": "9",
    }

    def __init__(self, category: str, primary_threshold: float = 1.0, fuzzy_threshold: int = 85):
        self.category = category
        self.primary_threshold = primary_threshold
        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def strip_html_tags(text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode and remove special characters."""
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\u200c', '').replace('\u200d', '')
        return text

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation + '।॥'))

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace."""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def full_normalize(text: str) -> str:
        """Apply all normalization steps."""
        text = BaseEvaluator.strip_html_tags(text)
        text = BaseEvaluator.normalize_unicode(text)
        text = BaseEvaluator.remove_punctuation(text)
        text = BaseEvaluator.normalize_whitespace(text)
        return text.lower()

    def extract_answer(self, response_text: str) -> str:
        """Extract answer from response text with robust patterns."""
        response_text = response_text.strip()

        patterns = [
            r"<answer>(.*?)</answer>",
            r"answer:\s*(.*?)(?:\n|$)",
            # r"उत्तर:\s*(.*?)(?:\n|$)",
            # r"जवाब:\s*(.*?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted

        return self.strip_html_tags(response_text).strip()

    def normalize_text(self, text: str) -> str:
        """Complete text normalization."""
        return self.full_normalize(text)

    def extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text with Indian numeral conversion."""
        for ind, ascii_ in self.NUMERAL_MAP.items():
            text = text.replace(ind, ascii_)

        if "=" in text:
            text = text.split("=")[-1]

        numbers = re.findall(r"-?[\d,]*\.?\d+", text)
        return [float(num.replace(",", "")) for num in numbers]

    def compare_numbers(self, pred: str, ref: str) -> bool:
        """Compare number sets for equality."""
        pred_nums = self.extract_numbers(pred)
        ref_nums = self.extract_numbers(ref)
        if not ref_nums:
            return True
        return sorted(pred_nums) == sorted(ref_nums)

    def detect_language(self, text: str, ref_lang: str) -> str:
        """Improved language detection using script detection."""
        # TODO: can use fasttext language detector too
        try:
            script = detect(text)
            script_to_lang = {
                "devanagari": ref_lang,  # Use reference language for Devanagari scripts
                "gujarati": "gu",
                "oriya": "or",
                "slp1": "en",
                "itrans": "en",
                "bengali": "bn",
                "tamil": "ta",
                "telugu": "te",
                "kannada": "kn",
                "malayalam": "ml",
            }
            detected_lang = script_to_lang.get(script, ref_lang)
            if script == "devanagari":
                # simple heuristics to distinguish Hindi vs Marathi
                text_lower = text.lower()
                hindi_indicators = [
                    "है",
                    "था",
                    "होग",
                    "नही",
                    "नहीं",
                    "मैं",
                    "मै",
                    "तुम",
                    "वह",
                    "यह",
                    "और",
                    "फिर",
                    "क्या",
                    "किधर",
                    "कितने",
                    "कौन",
                    "किस",
                    "के",
                    "से",
                    "लिए",
                    "लिये",
                ]
                marathi_indicators = [
                    "आहे",
                    "आहोत",
                    "होत",
                    "होता",
                    "असेल",
                    "नाही",
                    "मी",
                    "आम्ही",
                    "तुम्ही",
                    "आणि",
                    "कोठे",
                    "किती",
                    "कुठे",
                    "कोण",
                    "काय",
                ]

                hindi_count = sum(
                    text_lower.count(word) for word in hindi_indicators
                )
                marathi_count = sum(
                    text_lower.count(word) for word in marathi_indicators
                )

                if marathi_count > hindi_count:
                    detected_lang = "mr"
                else:
                    detected_lang = "hi"

            return detected_lang
        except Exception as e:
            logger.debug(f"Script detection failed: {e}")

        # Fallback: Simple Latin vs non-Latin detection
        latin_chars = sum(
            (0x0041 <= ord(c) <= 0x005A) or (0x0061 <= ord(c) <= 0x007A)
            for c in text
        )
        non_latin_chars = sum(
            c.isalpha()
            and not (
                (0x0041 <= ord(c) <= 0x005A) or (0x0061 <= ord(c) <= 0x007A)
            )
            for c in text
        )

        return "en" if latin_chars > non_latin_chars else ref_lang


    def detect_language_hallucination(self, pred_lang: str, ref_lang: str) -> Dict[str, Any]:
        """Complete language hallucination detection."""
        is_hallucination = pred_lang != ref_lang

        # English hallucination (most common)
        if pred_lang == "en" and ref_lang != "en":
            # Check if English response is actually correct content-wise
            hallucination_type = "english_response"
            severity = "medium"
        elif (pred_lang in ["hi", "mr", "gu", "or", "bn", "ta", "te", "kn", "ml"] and
            ref_lang in ["hi", "mr", "gu", "or", "bn", "ta", "te", "kn", "ml"]):

            # Determine if it's a related language (same script family)
            pred_script = self._get_script_family(pred_lang)
            ref_script = self._get_script_family(ref_lang)
            is_related_script = pred_script == ref_script

            hallucination_type = "indian_language_response"
            severity = "low" if is_related_script else "medium"
        else:
            hallucination_type = "unexpected_language"
            severity = "high"

        return {
            "is_hallucination": is_hallucination,
            "predicted_language": pred_lang,
            "reference_language": ref_lang,
            "hallucination_type": hallucination_type,
            "severity": severity,
        }

    def _get_script_family(self, lang: str) -> str:
        """Map language to script family."""
        script_map = {
            "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
            "gu": "gujarati", "or": "oriya", "bn": "bengali",
            "ta": "tamil", "te": "telugu", "kn": "kannada", "ml": "malayalam"
        }
        return script_map.get(lang, "unknown")

    def preprocess_response(self, response: str, pred_lang: str = None) -> Any:
        """Override in child classes for response-specific preprocessing."""
        return self.normalize_text(response)

    @abstractmethod
    def calculate_primary_metric(self, processed_pred: Any, processed_ref: Any,  pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Calculate primary metric (exact/match-based)."""
        pass

    @abstractmethod
    def calculate_fuzzy_metric(self, processed_pred: Any, processed_ref: Any, pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Calculate fuzzy metric (similarity-based)."""
        pass

    def evaluate(self, item: Dict, answer_lookup: Dict) -> EvaluationResult:
        """Complete evaluation logic for all evaluators."""
        # Extract basic information
        question_id = item["question_id"]
        ref = str(item.get("expected", ""))
        pred_raw = str(item.get("response", {}).get("content", ""))
        pred = self.extract_answer(pred_raw)
        ref_lang = self.LANGUAGE_MAP.get(str(item.get("language", "unknown")).lower(), "unknown")

        # Language detection
        pred_lang = self.detect_language(pred, ref_lang)
        language_hallucination = self.detect_language_hallucination(pred_lang, ref_lang)

        # Response preprocessing (category-specific)
        processed_pred = self.preprocess_response(pred, pred_lang)
        processed_ref = self.preprocess_response(ref, pred_lang)

        # Calculate primary and fuzzy metrics
        primary_match, primary_score = self.calculate_primary_metric(processed_pred, processed_ref, pred_lang, ref_lang)
        fuzzy_match, fuzzy_score = self.calculate_fuzzy_metric(processed_pred, processed_ref, pred_lang, ref_lang)

        # Basic metrics (available for all categories)
        metrics = {
            "exact_match": MetricHelper.exact_match(pred, ref),
            "fuzzy_similarity": MetricHelper.fuzzy_score(pred, ref),
            **MetricHelper.word_level_metrics(pred, ref),
            **MetricHelper.char_level_metrics(pred, ref),
        }

        # Calculate corrected primary and fuzzy metrics
        corrected_primary_match, corrected_primary_score =  primary_match, primary_score
        corrected_fuzzy_match, corrected_fuzzy_score = fuzzy_match, fuzzy_score
        corrected_metrics = metrics.copy()
        if language_hallucination["is_hallucination"]:
            matching_ref = answer_lookup.get(question_id, {}).get(pred_lang)
            if matching_ref:
                processed_matching_ref = self.preprocess_response(matching_ref, pred_lang)
                corrected_primary_match, corrected_primary_score = self.calculate_primary_metric(processed_pred, processed_matching_ref, pred_lang, pred_lang)
                corrected_fuzzy_match, corrected_fuzzy_score = self.calculate_fuzzy_metric(processed_pred, processed_matching_ref, pred_lang, pred_lang)
                corrected_metrics = {
                    "exact_match": MetricHelper.exact_match(pred, matching_ref),
                    "fuzzy_text_similarity": MetricHelper.fuzzy_score(pred, matching_ref),
                    **MetricHelper.word_level_metrics(pred, matching_ref),
                    **MetricHelper.char_level_metrics(pred, matching_ref),
                }
        corrected_metrics = {
            f"corrected_{k}": v for k, v in corrected_metrics.items()
        }
        metrics.update(corrected_metrics)

        match = {
            "primary": primary_match,
            "fuzzy": fuzzy_match,
            "corrected_primary": corrected_primary_match,
            "corrected_fuzzy": corrected_fuzzy_match
        }
        scores = {
            "primary": primary_score,
            "fuzzy": fuzzy_score,
            "corrected_primary": corrected_primary_score,
            "corrected_fuzzy": corrected_fuzzy_score
        }
        metadata = {
            "response_metadata": item["response"]["metadata"],
            "model": item["model"],
            "model_name": item["model_name"],
            "model_variant": item["model_variant"],
            "prompt_type": item["prompt_type"],
            "machine_name": item["machine_name"],
            "local_question_id": item["local_question_id"],
            "language_id": item["language_id"],
            "domain_id": item["domain_id"],
            "category_id": item["category_id"],
            "unique_id": item["unique_id"],
            "domain": item.get("domain")
        }
        return EvaluationResult(
            question_id=item["question_id"],
            language=ref_lang,
            category=self.category,
            predicted=pred,
            expected=ref,
            match=match,
            scores=scores,
            metrics=metrics,
            language_hallucination=language_hallucination,
            metadata=metadata
        )


###############################################################################
# Category Specific Evaluators


class FactualEvaluator(BaseEvaluator):
    """Plain factual QA evaluation."""

    def calculate_primary_metric(self, processed_pred: str, processed_ref: str, pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Calculate exact match as primary metric."""
        exact_match = processed_pred == processed_ref
        score = 1.0 if exact_match else 0.0
        return exact_match, score

    def calculate_fuzzy_metric(self, processed_pred: str, processed_ref: str, pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Calculate fuzzy similarity as secondary metric."""
        fuzzy_score_val = MetricHelper.fuzzy_score(processed_pred, processed_ref)
        fuzzy_match = fuzzy_score_val >= self.fuzzy_threshold
        return fuzzy_match, fuzzy_score_val / 100.0


# --------------------------------------------------------------------------- #


class MathEvaluator(BaseEvaluator):
    """Mathematical question evaluation."""

    def preprocess_response(self, response: str, pred_lang: str = None) -> List[float]:
        """Extract numbers from response."""
        return self.extract_numbers(response)

    def calculate_primary_metric(self, processed_pred: List[float], processed_ref: List[float], pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Compare number sets for equality."""
        if not processed_ref:  # If no numbers expected, consider correct (or incorrect!?)
            return True, 1.0

        is_match = sorted(processed_pred) == sorted(processed_ref)
        score = 1.0 if is_match else 0.0
        return is_match, score

    def calculate_fuzzy_metric(self, processed_pred: List[float], processed_ref: List[float], pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Use numerical comparison for fuzzy metric as well."""
        return self.calculate_primary_metric(processed_pred, processed_ref)


# --------------------------------------------------------------------------- #


class ChronologicalEvaluator(BaseEvaluator):
    """Order-sensitive evaluation using Kendall's Tau."""

    def parse_events(self, text: str) -> List[str]:
        """Parse events from text."""
        cleaned = self.extract_answer(text)
        events = []
        for event in cleaned.split(","):
            event = event.strip()
            if event:
                event = re.sub(r'^\d+[\.\)]\s*', '', event)  # Remove numbering
                events.append(event)
        return events

    def preprocess_response(self, response: str, pred_lang: str = None) -> Tuple[List[str], List[str]]:
        """Parse and return both raw and normalized events."""
        events = self.parse_events(response)
        normalized_events = [self.normalize_text(event) for event in events]
        return events, normalized_events

    def _fuzzy_map_events(self, pred_events: List[str], ref_events: List[str]) -> Tuple[List[str], List[str]]:
        """Map predicted events to reference events using fuzzy matching."""
        remaining_ref = ref_events[:]
        mapped = []
        common = []

        for pred_event in pred_events:
            best_match = None
            best_score = 0

            for ref_event in remaining_ref:
                score = MetricHelper.fuzzy_score(pred_event, ref_event)
                if score > best_score:
                    best_match, best_score = ref_event, score

            if best_match and best_score >= self.fuzzy_threshold:
                mapped.append(best_match)
                common.append(best_match)
                remaining_ref.remove(best_match)
            else:
                mapped.append(pred_event)

        return mapped, common

    def calculate_primary_metric(self, processed_pred: Tuple[List[str], List[str]],
                            processed_ref: Tuple[List[str], List[str]], pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Kendall's Tau with EXACT event matching."""
        pred_events, pred_norm = processed_pred
        ref_events, ref_norm = processed_ref

        if len(ref_events) < 2:
            # Single event - ordering doesn't apply
            if pred_events and ref_events:
                exact_match = pred_norm[0] == ref_norm[0]
                return exact_match, 1.0 if exact_match else 0.0
            return False, 0.0

        # Find common events using EXACT matching
        common_events = set(pred_norm) & set(ref_norm)

        if len(common_events) < 2:
            return False, 0.0

        # Get the order of exactly matching events in both sequences
        ref_order = [event for event in ref_norm if event in common_events]
        pred_order = [event for event in pred_norm if event in common_events]

        # Calculate Kendall's Tau on exact matches
        kendall_result = MetricHelper.kendalls_tau(pred_order, ref_order)
        tau = kendall_result["kendall_tau"]

        match = tau >= 0.5
        return match, (tau + 1) / 2

    def calculate_fuzzy_metric(self, processed_pred: Tuple[List[str], List[str]],
                            processed_ref: Tuple[List[str], List[str]], pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Kendall's Tau with FUZZY event matching."""
        pred_events, pred_norm = processed_pred
        ref_events, ref_norm = processed_ref

        if len(ref_events) < 2:
            # Single event - use fuzzy matching
            if pred_events and ref_events:
                fuzzy_match = MetricHelper.fuzzy_score(pred_events[0], ref_events[0]) >= self.fuzzy_threshold
                return fuzzy_match, 1.0 if fuzzy_match else 0.0
            return False, 0.0

        # Map events using FUZZY matching
        mapped_pred, common_events = self._fuzzy_map_events(pred_norm, ref_norm)

        if len(common_events) < 2:
            return False, 0.0

        # Calculate Kendall's Tau on fuzzy-mapped events
        kendall_result = MetricHelper.kendalls_tau(mapped_pred, ref_norm)
        tau = kendall_result["kendall_tau"]

        match = tau >= 0.5
        return match, (tau + 1) / 2


# --------------------------------------------------------------------------- #


class TrueFalseEvaluator(BaseEvaluator):
    """True/False question evaluation - simple text comparison."""

    def preprocess_response(self, response: str, pred_lang: str = None) -> str:
        """Normalize true/false response to simple true/false."""
        normalized = self.normalize_text(response)

        # Map common true/false patterns to standardized form
        true_patterns = ["true", "yes", "correct", "सत्य", "સાચું", "ସତ୍ୟ"]
        false_patterns = ["false", "no", "incorrect", "असत्य", "ખોટું", "ମିଥ୍ୟା"]

        if any(pattern in normalized for pattern in true_patterns):
            return "true"
        elif any(pattern in normalized for pattern in false_patterns):
            return "false"
        else:
            return normalized

    def calculate_primary_metric(self, processed_pred: str, processed_ref: str, pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Exact match for true/false answers."""
        if processed_ref not in ("true", "false"):
            raise SkipItem(f"Invalid reference value for true/false: {processed_ref}")

        exact_match = processed_pred == processed_ref
        return exact_match, 1.0 if exact_match else 0.0

    def calculate_fuzzy_metric(self, processed_pred: str, processed_ref: str, pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Fuzzy match for true/false answers."""
        if processed_ref not in ("true", "false"):
            raise SkipItem(f"Invalid reference value for true/false: {processed_ref}")

        # For true/false, fuzzy matching is the same as exact matching
        # since the answer space is binary
        return self.calculate_primary_metric(processed_pred, processed_ref)


# --------------------------------------------------------------------------- #



class SemanticallyIncorrectEvaluator(FactualEvaluator):
    """Detect invalid questions and evaluate valid ones."""

    def calculate_primary_metric(self, processed_pred: str, processed_ref: str, pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Classify as valid/invalid question."""
        is_ref_invalid = "invalid" in processed_ref.lower()
        is_pred_invalid = "invalid" in processed_pred.lower()

        if is_ref_invalid and is_pred_invalid:
            return True, 1.0  # True Positive
        elif not is_ref_invalid and not is_pred_invalid:
            return super().calculate_primary_metric(processed_pred, processed_ref)
        else:
            return False, 0.0  # False Positive

    def calculate_fuzzy_metric(self, processed_pred: str, processed_ref: str, pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Use semantic similarity for valid questions."""
        is_pred_invalid = MetricHelper.fuzzy_match("invalid", processed_ref.lower())
        is_ref_invalid = "invalid" in processed_pred.lower()

        if is_ref_invalid and is_pred_invalid:
            return True, 1.0  # True Positive
        elif not is_ref_invalid and not is_pred_invalid:
            return super().calculate_fuzzy_metric(processed_pred, processed_ref)
        else:
            return False, 0.0  # False Positive


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class ReasoningEvaluator(BaseEvaluator):
    """Multiple-choice reasoning evaluation with basic metrics (Exact/Fuzzy)."""

    
    primary_threshold = 1.0  # Primary match is now a strict Exact Match (score must be 1.0)
    fuzzy_threshold = 85   

    def calculate_primary_metric(self, processed_pred: str, processed_ref: str, pred_lang: str, ref_lang: str = None) -> Tuple[bool, float]:
        """Primary metric: Exact Match of normalized text."""
        exact_match = processed_pred == processed_ref
        score = 1.0 if exact_match else 0.0
        return exact_match, score

    def calculate_fuzzy_metric(self, processed_pred: str, processed_ref: str, pred_lang: str, ref_lang: str = None) -> Tuple[bool, float]:
        """Fuzzy metric: Simple fuzzy score above threshold."""
        if not processed_pred or not processed_ref:
            return False, 0.0
            
        fuzzy_score_val = MetricHelper.fuzzy_score(processed_pred, processed_ref)
        fuzzy_match = fuzzy_score_val >= self.fuzzy_threshold
        
        return fuzzy_match, fuzzy_score_val / 100.0

# --------------------------------------------------------------------------- #


class SummarizationEvaluator(ReasoningEvaluator):
    """Long-form generation evaluation."""

    # def calculate_primary_metric(self, processed_pred: str, processed_ref: str) -> Tuple[bool, float]:
    #     """Use ROUGE-L as primary metric."""
    #     rouge_scores = MetricHelper.rouge_scores(processed_pred, processed_ref)
    #     rouge_l = rouge_scores["rougeL"]
    #     match = rouge_l >= 0.5  # Adjust threshold as needed
    #     return match, rouge_l

    # def calculate_fuzzy_metric(self, processed_pred: str, processed_ref: str) -> Tuple[bool, float]:
    #     """Use semantic similarity as fuzzy metric."""
    #     # For summarization, we need the original text for semantic similarity
    #     similarity = MetricHelper.multi_metric_similarity(
    #         processed_pred, processed_ref, "en", "en"  # Language will be handled by caller
    #     )
    #     match = similarity >= 0.7  # Adjust threshold as needed
    #     return match, similarity


# --------------------------------------------------------------------------- #


class WordOrderingEvaluator(BaseEvaluator):
    """Word ordering evaluation using Kendall's Tau for sequence accuracy."""

    primary_threshold = 0.7
    fuzzy_threshold = 80

    def preprocess_response(self, response: str, pred_lang: str = None) -> List[str]:
        """Split response into words for sequence analysis."""
        normalized = self.normalize_text(response)
        # Split into words, filtering out empty strings
        words = [word for word in normalized.split() if word]
        return words

    def calculate_primary_metric(self, processed_pred: List[str], processed_ref: List[str], pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Primary metric: Kendall's Tau for exact word sequence matching."""
        if not processed_pred or not processed_ref:
            return False, 0.0

        # For word ordering, we want to compare the exact sequence of words
        if len(processed_pred) != len(processed_ref):
            return False, 0.0  # Different number of words = wrong order

        # Calculate Kendall's Tau on the exact word sequences
        kendall_result = MetricHelper.kendalls_tau(processed_pred, processed_ref)
        tau = kendall_result["kendall_tau"]

        # Primary match based on strong ordering correlation
        primary_match = tau >= self.primary_threshold
        return primary_match, (tau + 1) / 2  # Normalize to [0,1]

    def calculate_fuzzy_metric(self, processed_pred: List[str], processed_ref: List[str], pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Fuzzy metric: Kendall's Tau with fuzzy word matching."""
        if not processed_pred or not processed_ref:
            return False, 0.0

        # For fuzzy matching, we need to map similar words first
        # Simple approach: use the first method from ChronologicalEvaluator
        mapped_pred, common_words = self._fuzzy_map_words(processed_pred, processed_ref)

        if len(common_words) < 2:
            return False, 0.0  # Need at least 2 common words for ordering

        # Calculate Kendall's Tau on fuzzy-mapped words
        kendall_result = MetricHelper.kendalls_tau(mapped_pred, processed_ref)
        tau = kendall_result["kendall_tau"]

        fuzzy_match = tau >= (self.fuzzy_threshold / 100.0)
        return fuzzy_match, (tau + 1) / 2

    def _fuzzy_map_words(self, pred_words: List[str], ref_words: List[str]) -> Tuple[List[str], List[str]]:
        """Map predicted words to reference words using fuzzy matching."""
        remaining_ref = ref_words[:]
        mapped = []
        common = []

        for pred_word in pred_words:
            best_match = None
            best_score = 0

            for ref_word in remaining_ref:
                score = MetricHelper.fuzzy_score(pred_word, ref_word)
                if score > best_score:
                    best_match, best_score = ref_word, score

            if best_match and best_score >= self.fuzzy_threshold:
                mapped.append(best_match)
                common.append(best_match)
                remaining_ref.remove(best_match)
            else:
                mapped.append(pred_word)  # Keep unmatched words

        return mapped, common

# --------------------------------------------------------------------------- #


class NEREvaluator(BaseEvaluator):
    """Named Entity Recognition evaluation."""

    f1_threshold = 0.7
    fuzzy_entity_threshold = 80

    @staticmethod
    def _parse_ner(text: str) -> List[Tuple[str, str]]:
        """Parse NER tags from text."""
        parts = text.strip().split()
        if len(parts) % 2 != 0:
            return []  # Invalid format
        return [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]

    @staticmethod
    def _extract_entities(tag_pairs: List[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        """Convert BIO tags to entity set."""
        entities = set()
        current_entity = []
        current_type = None

        for word, tag in tag_pairs:
            if tag.startswith("B-"):
                if current_entity:
                    entities.add((" ".join(current_entity), current_type))
                current_entity = [word]
                current_type = tag[2:]
            elif tag.startswith("I-") and current_type == tag[2:]:
                current_entity.append(word)
            else:
                if current_entity:
                    entities.add((" ".join(current_entity), current_type))
                current_entity = []
                current_type = None

        if current_entity:
            entities.add((" ".join(current_entity), current_type))

        return entities

    def preprocess_response(self, response: str, pred_lang: str = None) -> Set[Tuple[str, str]]:
        """Extract entities from NER response."""
        tag_pairs = self._parse_ner(response)
        return self._extract_entities(tag_pairs)

    def calculate_primary_metric(self, processed_pred: Set[Tuple[str, str]],
                               processed_ref: Set[Tuple[str, str]], pred_lang: str = None, ref_lang: str = None) -> Tuple[bool, float]:
        """Calculate F1 score as primary metric."""
        if not processed_ref:
            return True, 1.0  # No entities expected

        tp = len(processed_pred & processed_ref)
        fp = len(processed_pred - processed_ref)
        fn = len(processed_ref - processed_pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        match = f1 >= self.f1_threshold
        return match, f1

    def _calculate_fuzzy_tp(self, pred_entities: Set[Tuple[str, str]], ref_entities: Set[Tuple[str, str]]) -> float:
        """Calculate fuzzy true positives using entity similarity."""
        if not ref_entities:
            return 0.0

        total_similarity = 0.0
        matched_refs = set()

        for pred_entity, pred_type in pred_entities:
            best_similarity = 0.0
            best_match = None

            for ref_entity, ref_type in ref_entities:
                if ref_type == pred_type and ref_entity not in matched_refs:
                    # Calculate entity text similarity
                    entity_similarity = MetricHelper.fuzzy_score(pred_entity, ref_entity) / 100.0
                    if entity_similarity > best_similarity:
                        best_similarity = entity_similarity
                        best_match = ref_entity

            if best_match and best_similarity >= (self.fuzzy_entity_threshold / 100.0):
                total_similarity += best_similarity
                matched_refs.add(best_match)

        return total_similarity

    def calculate_fuzzy_metric(self, processed_pred: Set[Tuple[str, str]],
                             processed_ref: Set[Tuple[str, str]], pred_lang: str, ref_lang: str) -> Tuple[bool, float]:
        """Fuzzy F1 score with entity similarity."""
        if not processed_ref:
            return True, 1.0

        # Calculate fuzzy true positives (weighted by similarity)
        fuzzy_tp = self._calculate_fuzzy_tp(processed_pred, processed_ref)
        fuzzy_fp = max(0, len(processed_pred) - fuzzy_tp)
        fuzzy_fn = max(0, len(processed_ref) - fuzzy_tp)

        precision = fuzzy_tp / (fuzzy_tp + fuzzy_fp) if (fuzzy_tp + fuzzy_fp) > 0 else 0.0
        recall = fuzzy_tp / (fuzzy_tp + fuzzy_fn) if (fuzzy_tp + fuzzy_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        match = f1 >= self.f1_threshold
        return match, f1

# --------------------------------------------------------------------------- #


class IndianQuestionsEvaluator(FactualEvaluator):
    """Indian-specific questions (inherits from FactualEvaluator)."""

    pass


###############################################################################
# Main Evaluation Function


def build_answer_lookup(data: List[Dict]) -> Dict:
    """Build answer lookup table."""
    answer_lookup = defaultdict(dict)
    for item in data:
        ref = str(item.get("expected", ""))
        lang_full = str(item.get("language", "")).lower()

        if ref and lang_full in BaseEvaluator.LANGUAGE_MAP:
            lang_code = BaseEvaluator.LANGUAGE_MAP[lang_full]
            answer_lookup[item["question_id"]][lang_code] = ref
            answer_lookup[item["question_id"]]["_domain"] = item.get(
                "domain", ""
            )
            answer_lookup[item["question_id"]]["_category"] = item.get(
                "category", ""
            )

    return answer_lookup

def evaluate_jsonl_file(input_file: str, output_file: str) -> pd.DataFrame:
    """Evaluate JSONL file with complete metrics."""
    # Load data
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} items from {input_file}")

    # Build answer lookup
    answer_lookup = build_answer_lookup(data)

    # Define all evaluators
    evaluators = {
        "factual_questions": FactualEvaluator("factual_questions"),
        "maths_questions": MathEvaluator("maths_questions"),
        "chrono_questions": ChronologicalEvaluator("chrono_questions"),
        "true_false_questions": TrueFalseEvaluator("true_false_questions"),
        "semantically_incorrect_questions": SemanticallyIncorrectEvaluator("semantically_incorrect_questions"),
        "reasoning_questions": ReasoningEvaluator("reasoning_questions"),
        "summarization_questions": SummarizationEvaluator("summarization_questions"),
        "word_ordering_questions": WordOrderingEvaluator("word_ordering_questions"),
        "ner_questions": NEREvaluator("ner_questions"),
        "indian_questions": IndianQuestionsEvaluator("indian_questions"),
    }

    # Evaluate each item
    results = []
    for item in tqdm(data):
        category = item.get("category")
        if category not in evaluators:
            logger.warning(f"No evaluator for category: {category}")
            continue

        # try:
        evaluator: BaseEvaluator = evaluators[category]
        result: EvaluationResult = evaluator.evaluate(item, answer_lookup)

        # Convert to dict for DataFrame
        result_dict = {
            "question_id": result.question_id,
            "language": result.language,
            "category": result.category,
            "predicted": result.predicted,
            "expected": result.expected,
            "match": result.match,
            "scores": result.scores,
            "metrics": result.metrics,
            "language_hallucination": result.language_hallucination,
            **result.metadata
        }
        results.append(result_dict)

        # except SkipItem as e:
        #     logger.debug(f"Skipping item: {e}")
        # except Exception as e:
        #     logger.error(f"Failed to evaluate item in {category}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    logger.info(f"Saved results to {output_file}")

    return df


###############################################################################


def main():
    """Complete CLI interface."""
    parser = argparse.ArgumentParser(description="Complete BHRAM-IL Evaluation")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run evaluation
    df = evaluate_jsonl_file(args.input, args.output)
    logger.info(f"Evaluation completed. Processed {len(df)} items.")

if __name__ == "__main__":
    main()
