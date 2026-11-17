# Cross-Lingual Evaluation of Hallucination in Multilingual LLMs

This project **compares pretrained large language models (LLMs) to evaluate multilingual hallucination**, with a focus on Indian languages. We measure how often each model produces plausible but incorrect answers, and how that changes across languages, prompt styles, question categories and models.

> [!IMPORTANT]
> **We did not train any model.** All models used here are **pretrained, open-source models taken from other authors' repositories** (mainly the Hugging Face Hub, e.g. `Qwen/Qwen2.5-3B-Instruct`). We only run them for inference and compare their outputs.
>
> The benchmark framework (dataset, prompts, evaluation metrics) builds on the **BHRAM-IL** project by Terdalkar et al.: https://github.com/sambhashana/BHRAM-IL/ — see [Credits](#credits-and-attribution).

## Project Goal

- Compare multiple pretrained LLMs on the same multilingual hallucination benchmark.
- Evaluate **factual hallucinations** across languages (English, Hindi, Kannada, Tamil).
- Compare **English-prompt vs native-language-prompt** behaviour.
- Analyse results per model, per language and per question category.

## What We Did

| Step | Description |
|------|-------------|
| 1. Benchmark | Reused the BHRAM-IL benchmark (10K benchmarked / 40K full subsets of Hindi, Gujarati, Marathi, Odia and English questions). |
| 2. Extension | Built an aligned **4-language subset (English, Hindi, Kannada, Tamil)** by selecting translation-safe categories and translating English/Hindi rows into Kannada and Tamil (`collect/`). |
| 3. Generation | Ran pretrained models from the Hugging Face Hub on the benchmark using `run/generate_response_hf.py` (Colab-friendly, supports 4-bit/8-bit quantization and resuming). |
| 4. Evaluation | Scored the responses with the category-specific metrics in `evaluate/evaluation.py`. |
| 5. Comparison | Compared models, languages, prompt modes and categories using the evaluated outputs. |

### Models

All models are pretrained and used as-is from their original repositories (no fine-tuning):

- `Qwen/Qwen2.5-3B-Instruct` (used in our runs so far)
- Other small instruction-tuned models suitable for a free Colab GPU, e.g. `google/gemma-2-2b-it`, `microsoft/Phi-3.5-mini-instruct`

Any Hugging Face causal LM can be compared by changing the `--model` argument. Some gated models need a Hugging Face login and access approval.

### Question Categories

`factual`, `chrono` (chronological ordering), `indian` (India-specific knowledge), `maths`, `reasoning`, `true_false`, plus `ner`, `word_ordering` and `semantically_incorrect` in the original benchmark.

The final 4-language set uses only the translation-safe categories, because NER, word-ordering and semantically-incorrect questions can change meaning or structure when machine-translated.

## Repository Structure

```
.
├── collect/                      # Data collection, subset building and translation scripts
├── dataset/
│   ├── BHRAM_IL_10K/             # Benchmarked subset (10K)
│   ├── BHRAM_IL_40K/             # Full dataset (40K)
│   └── final_4lang_seed/         # Our English/Hindi/Kannada/Tamil aligned subset
├── run/
│   ├── generate_response_hf.py   # Generate responses with Hugging Face models (Colab)
│   ├── generate_response.py      # Original Ollama-based generator
│   ├── prompts/                  # Prompt templates per language
│   ├── colab_requirements.txt
│   └── README.md                 # Detailed Colab workflow
├── evaluate/
│   └── evaluation.py             # Category-specific evaluation
├── output/                       # Model responses and evaluated responses
├── BHRAM_IL_colab_pipeline.ipynb # End-to-end Colab notebook (generate, evaluate, plot)
└── HANDOFF_FRIEND_CODEX.md       # Detailed project notes and continuation guide
```

## How to Run

The workflow is: edit code locally, push to GitHub, then run generation and evaluation on a Google Colab GPU and save results to Google Drive.

```bash
git clone https://github.com/Guru-preeth10/hallucination.git
cd hallucination/run
pip install -r colab_requirements.txt
pip install torch fasttext-wheel bert-score nltk
```

**1. Generate responses** (example: one model, one category, English prompts):

```bash
python generate_response_hf.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset ../dataset/BHRAM_IL_10K/dataset_10k.jsonl \
  --category chrono_questions \
  --limit 100 \
  --prompt-mode english \
  --quantization 4bit \
  --max-new-tokens 192 \
  --output output.chrono.qwen.english.100.jsonl
```

**2. Evaluate:**

```bash
cd ../evaluate
python evaluation.py --input ../run/output.chrono.qwen.english.100.jsonl --output evaluated.chrono.qwen.english.100.jsonl
```

**3. Compare models:** repeat step 1 with a different `--model` and compare the evaluated JSONL files.

Notes:

- `--prompt-mode english` uses the English prompt for every row (all languages still run); `--prompt-mode native` uses the prompt in each row's own language.
- `--limit` counts rows, not unique questions (each question appears once per language).
- Interrupted runs resume automatically if the same output file is reused.
- fastText models are not included; the evaluator still runs without them (with warnings).

See [run/README.md](run/README.md) for the full Colab guide.

## Known Limitations

- Chronology `primary` scores are lenient: a partial answer in the correct order can still score well, so inspect outputs manually.
- Small models can truncate answers or produce malformed `<answer>` tags and corrupted text in lower-resource scripts (e.g. Odia); we use `--max-new-tokens 192` to reduce truncation.
- Kannada and Tamil rows are machine-translated and may contain translation noise.

## Credits and Attribution

- **Benchmark, prompts and evaluation framework:** [BHRAM-IL](https://github.com/sambhashana/BHRAM-IL/) ([dataset on HuggingFace](https://huggingface.co/datasets/sambhashana/BHRAM-IL/)). *BHRAM-IL: A Benchmark for Hallucination Recognition and Assessment in Multiple Indian Languages*, 1st Workshop on BHASHA at IJCNLP-AACL 2025.
- **Pretrained models:** created and released by their respective authors (e.g. Alibaba Qwen, Google Gemma, Microsoft Phi) and loaded from the Hugging Face Hub. We do not claim ownership of any model.
- **Our contribution:** the Hugging Face/Colab generation pipeline, the English/Hindi/Kannada/Tamil aligned subset and translation scripts, and the multilingual model comparison and analysis.

Please cite the original work if you use the benchmark:

```bibtex
@inproceedings{terdalkar2025bhramil,
    title = {{BHRAM-IL}: A Benchmark for Hallucination Recognition and Assessment in Multiple Indian Languages},
    author = "Terdalkar, Hrishikesh  and
      Bhojani, Kirtan  and
      Dongare, Aryan  and
      Behera, Omm Aditya",
    booktitle = "Proceedings of the 1st Workshop on BHASHA: Benchmarks, Harmonization, Annotation, and Standardization for Human-Centric AI in Indian Languages at the 14th International Joint Conference on NLP & Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bhasha-1.27/",
    pages = "102--116",
    ISBN = "979-8-89176-313-5",
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
