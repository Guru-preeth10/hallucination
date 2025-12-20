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

```
@inproceedings{terdalkar2025bhramil,
    title = {{BHRAM-IL}: A Benchmark for Hallucination Recognition and Assessment in Multiple Indian Languages},
    author = "Terdalkar, Hrishikesh  and
      Bhojani, Kirtan  and
      Dongare, Aryan  and
      Behera, Omm Aditya",
    editor = "",
    booktitle = "Proceedings of the 1st Workshop on BHASHA: Benchmarks, Harmonization, Annotation, and Standardization for Human-Centric AI in Indian Languages at the 14th International Joint Conference on NLP \&amp; Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bhasha-1.27/",
    doi = "",
    pages = "102--116",
    ISBN = "979-8-89176-313-5",
    abstract = "Large language models (LLMs) are increasingly deployed in multilingual applications but often generate plausible yet incorrect or misleading outputs, known as hallucinations. While hallucination detection has been studied extensively in English, under-resourced Indian languages remain largely unexplored. We present BHRAM-IL, a benchmark for hallucination recognition and assessment in multiple Indian languages, covering Hindi, Gujarati, Marathi, Odia, along with English. The benchmark comprises 36,047 curated questions across nine categories spanning factual, numerical, reasoning, and linguistic tasks. We evaluate 14 state-of-the-art multilingual LLMs on a benchmark subset of 10,265 questions, analyzing cross-lingual and factual hallucinations across languages, models, scales, categories, and domains using category-specific metrics normalized to (0,1) range. Aggregation over all categories and models yields a primary score of 0.23 and a language-corrected fuzzy score of 0.385, demonstrating the usefulness of BHRAM-IL for hallucination-focused evaluation. The dataset, and the code for generation and evaluation are available on GitHub (https://github.com/sambhashana/BHRAM-IL/) and HuggingFace (https://huggingface.co/datasets/sambhashana/BHRAM-IL/) to support future research in multilingual hallucination detection and mitigation."
}

@misc{terdalkar2025bhramilbenchmarkhallucinationrecognition,
      title={{BHRAM-IL}: A {B}enchmark for {H}allucination {R}ecognition and {A}ssessment in {M}ultiple {I}ndian {L}anguages},
      author={Hrishikesh Terdalkar and Kirtan Bhojani and Aryan Dongare and Omm Aditya Behera},
      year={2025},
      eprint={2512.01852},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.01852},
}
```
