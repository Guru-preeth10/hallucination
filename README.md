# BHRAM-IL: A Benchmark for Hallucination Recognition and Assessment in Multiple Indian Languages

Full Paper accepted at [1st Workshop on BHASHA: Benchmarks, Harmonization, Annotation, and Standardization for Human-Centric AI in Indian Languages](https://bhasha-workshop.github.io) at [IJCNLP-AACL 2025](https://2025.aaclnet.org).

## About

Large language models (LLMs) are increasingly deployed in multilingual applications
but often generate plausible yet incorrect or misleading outputs, known as hallucinations.
While hallucination detection has been studied extensively in English, under-resourced
Indian languages remain largely unexplored. We present BHRAM-IL, a benchmark for
hallucination recognition and assessment in multiple Indian languages, covering Hindi,
Gujarati, Marathi, Odia, along with English. The benchmark comprises 36,047 curated
questions across nine categories spanning factual, numerical, reasoning, and linguistic
tasks. We evaluate 14 state-of-the-art multilingual LLMs on a benchmark subset of
10,265 questions, analyzing cross-lingual and factual hallucinations across languages,
models, scales, categories, and domains using category-specific metrics normalized to (0,1)
range. Aggregation over all categories and models yields a primary score of 0.23 and
a language-corrected fuzzy score of 0.385, demonstrating the usefulness of BHRAM-IL
for hallucination-focused evaluation. 

Also available on [HuggingFace](https://huggingface.co/datasets/sambhashana/BHRAM-IL/).

## Structure

* `dataset` - dataset in 10K (benchmarked) and 40K (full) versions
* `collect` - data colleciton scripts
* `run` - code for generating model response on the benchmark
* `evaluate` - category specific evaluation
* `output` - response produced by LLMs and evaluated responses

## Cite
