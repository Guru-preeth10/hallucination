# Colab Workflow for BHRAM-IL

This repository can be run on Google Colab without Ollama.

The recommended path is:

1. Use your laptop only for code changes and Git.
2. Push the repository to GitHub.
3. Clone the repository inside Colab.
4. Run generation with `run/generate_response_hf.py`.
5. Evaluate the output JSONL with `evaluate/evaluation.py`.
6. Save the generated and evaluated JSONL files to Google Drive.

## What changed

The original script, `run/generate_response.py`, is tied to Ollama and to Excel files in `run/data/`.

The new script, `run/generate_response_hf.py`, is Colab-oriented:

- reads benchmark JSONL directly from `dataset/`
- uses Hugging Face `transformers`
- supports `4bit`, `8bit`, or full precision loading
- writes evaluator-compatible JSONL output
- resumes from partially completed output files

## Colab setup

### 1. Start a GPU runtime

In Colab:

- `Runtime -> Change runtime type`
- choose `T4 GPU` if available

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 3. Clone your repository

```bash
%cd /content
!git clone https://github.com/<your-username>/BHRAM-IL.git
%cd /content/BHRAM-IL/run
```

If you already cloned it once:

```bash
%cd /content/BHRAM-IL
!git pull
%cd /content/BHRAM-IL/run
```

### 4. Install dependencies

```bash
!pip install -q -r colab_requirements.txt
```

If you want to evaluate responses too, also install:

```bash
!pip install -q torch fasttext-wheel bert-score nltk
```

Note:

- `bert-score` is imported by `evaluate/evaluation.py`
- `fasttext` models are not included in the repo, so corrected semantic similarity features may need additional setup later

## Recommended first test

Do not start with the full benchmark.

Use a pilot run first:

- one small model
- one category
- 20 to 100 rows

Example model choices for free Colab:

- `google/gemma-2-2b-it`
- `Qwen/Qwen2.5-3B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`

Some gated models may require Hugging Face login and access approval.

## Generation command

Run from `/content/BHRAM-IL/run`.

Example: test only chronology questions with 25 rows using English prompts.

```bash
!python generate_response_hf.py \
  --model google/gemma-2-2b-it \
  --dataset ../dataset/BHRAM_IL_10K/dataset_10k.jsonl \
  --category chrono_questions \
  --limit 25 \
  --prompt-mode english \
  --quantization 4bit \
  --max-new-tokens 96 \
  --output /content/drive/MyDrive/BHRAM-Results/output.chrono.gemma2-2b-it.25.jsonl
```

Example: run two categories with native-language prompts.

```bash
!python generate_response_hf.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset ../dataset/BHRAM_IL_10K/dataset_10k.jsonl \
  --category factual_questions chrono_questions \
  --limit 100 \
  --prompt-mode native \
  --quantization 4bit \
  --output /content/drive/MyDrive/BHRAM-Results/output.qwen2.5-3b.native.100.jsonl
```

## Evaluation command

Run from `/content/BHRAM-IL/evaluate`.

```bash
%cd /content/BHRAM-IL/evaluate
!python evaluation.py \
  --input /content/drive/MyDrive/BHRAM-Results/output.chrono.gemma2-2b-it.25.jsonl \
  --output /content/drive/MyDrive/BHRAM-Results/evaluated.chrono.gemma2-2b-it.25.jsonl
```

## Suggested execution plan

### Stage 1: smoke test

- `limit 10`
- one category
- one model

Goal:

- confirm model loads
- confirm output JSONL is written
- confirm evaluator runs

### Stage 2: pilot benchmark

- `limit 100 to 300`
- two or three categories
- one model

Goal:

- estimate runtime
- estimate Colab memory usage
- inspect answer formatting quality

### Stage 3: actual experiment

- full `dataset_10k.jsonl`
- selected categories or all categories
- one model at a time

Goal:

- produce publishable baseline outputs

## Important practical notes

### Prompt mode

`--prompt-mode english`

- uses the English prompt template for every language row
- mirrors the cross-lingual setup from the original script

`--prompt-mode native`

- uses the prompt template matching the row language
- useful when you want to compare English-prompt versus native-prompt behavior

### Resuming interrupted runs

If the output file already exists, the script skips rows already written.

That means if Colab disconnects, you can rerun the same command and continue.

### Output compatibility

The new script writes metadata fields expected by `evaluate/evaluation.py`, including:

- `question_id`
- `language`
- `category`
- `question`
- `expected`
- `response`
- `model`
- `model_name`
- `model_variant`
- `prompt_type`
- `language_id`
- `category_id`
- `unique_id`

### Quantization guidance

Use these defaults first:

- free Colab T4: `--quantization 4bit`
- if 4-bit gives trouble for a specific model: try `--quantization 8bit`
- if the model is small and you want simplicity: `--quantization none --torch-dtype float16`

## What you should do next

1. Push this repository to your GitHub account.
2. Open Colab and mount Drive.
3. Clone the repo.
4. Run a 10-row smoke test on `chrono_questions`.
5. If generation works, run evaluation.
6. Only then scale to 100 rows, then 1000, then the full 10K.
