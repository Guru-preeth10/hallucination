# BHRAM-IL Research Handoff for Friend / Next Codex

Last updated: July 18, 2026

## 1. Purpose of This Document

This document is a full handoff for continuing the project without needing to reconstruct prior discussion.

It is written for:

- my friend
- my friend's laptop / Colab workflow
- another Codex session that needs to continue from the current state

The goal is to explain:

- what this repository is doing
- what has already been changed
- what has already been tested
- what problems were found
- what is left to do next
- how to continue the experiments safely
- how to extend the benchmark to Kannada and Tamil

This is not a generic README. It is the project continuation document.

## 2. Research Context

This is a research internship project under CARA / RV University.

Approved topic:

`Cross-Lingual Evaluation of Factual Hallucinations in Open-Source Large Language Models`

Important clarification:

- The primary approved baseline paper is the Nature 2024 paper on semantic entropy for hallucination detection.
- We are **not** using that paper's codebase directly because it is too heavy for undergraduate hardware.
- We are using `BHRAM-IL` only as the practical implementation framework:
  - dataset
  - prompting setup
  - response generation
  - evaluation scripts

So:

- approved scientific framing comes from the Nature paper
- practical benchmark framework comes from BHRAM-IL

## 3. Repository Reality Check

This repo is not a perfectly clean reproducibility package.

What it contains:

- benchmark datasets in JSONL
- an older Ollama-based generator
- a more complete evaluator
- some prototype data-collection scripts

Important mismatch discovered earlier:

- checked-in benchmark is in `dataset/*.jsonl`
- old generation script expected Excel files in `run/data/*.xlsx`
- evaluator expects richer metadata than the old generator produced

Because of that, a Colab-friendly Hugging Face generation path was added.

## 4. What Was Added / Changed

These files were created or updated during this work:

### Added

- [run/generate_response_hf.py](/home/guru/Documents/hallucination/BHRAM-IL/run/generate_response_hf.py:1)
- [run/colab_requirements.txt](/home/guru/Documents/hallucination/BHRAM-IL/run/colab_requirements.txt:1)
- [HANDOFF_FRIEND_CODEX.md](/home/guru/Documents/hallucination/BHRAM-IL/HANDOFF_FRIEND_CODEX.md:1)

### Updated

- [run/README.md](/home/guru/Documents/hallucination/BHRAM-IL/run/README.md:1)

### Purpose of the new generator

The new file [run/generate_response_hf.py](/home/guru/Documents/hallucination/BHRAM-IL/run/generate_response_hf.py:1):

- reads benchmark JSONL directly from `dataset/`
- uses Hugging Face `transformers` instead of Ollama
- works naturally in Google Colab
- supports `4bit`, `8bit`, or full precision model loading
- resumes from partially completed output files
- writes evaluator-compatible JSONL

This is the main reason the project is now runnable on Colab without local Ollama.

## 5. Why Colab Was Used Instead of Local Laptop

The original user laptop is low-spec and cannot practically run Ollama + benchmark experiments at scale.

So the architecture was changed to:

`Laptop for code + GitHub + Colab for model execution + Google Drive for outputs`

This is the intended workflow now.

## 6. Current Working Execution Pattern

The working experimental flow is:

1. Edit code locally.
2. Push code to GitHub.
3. Open Colab.
4. Mount Google Drive.
5. Clone or pull the repo from GitHub.
6. Install dependencies from `run/colab_requirements.txt`.
7. Run generation using `run/generate_response_hf.py`.
8. Save partial and final JSONL outputs to Google Drive.
9. Run evaluation using `evaluate/evaluation.py`.
10. Save evaluated JSONL to Google Drive.

This flow has already been validated on Colab.

## 7. Important Colab Usage Rule

There was confusion earlier between:

- notebook code cells
- terminal shell commands

Reminder:

- `%cd` and `!python` are for notebook code cells
- plain `cd`, `python`, `git` are for terminal

Because Colab terminal state disappears easily, notebook cells are preferred for long runs and reproducibility.

## 8. What Has Been Successfully Tested

### Confirmed working

The following was successfully tested end-to-end in Colab:

- model loading with Hugging Face
- generation on `chrono_questions`
- writing output JSONL to Google Drive
- evaluation with `evaluate/evaluation.py`
- writing evaluated JSONL to Google Drive

### Model used

- `Qwen/Qwen2.5-3B-Instruct`

### Pilot tested

- `chrono_questions`
- small smoke test
- 100-row pilot
- later scaling to larger runs

### Confirmed evaluation behavior

Evaluation completed successfully even though fastText warnings appeared.

Those fastText warnings are non-fatal because the required fastText binary models are not present under `evaluate/fasttext/`.

So:

- evaluation script runs
- optional semantic language correction resources are missing
- this does not block current benchmark execution

## 9. Paths Used in Colab

These are the standard paths assumed during runs:

### Repo

`/content/hallucination`

### Run directory

`/content/hallucination/run`

### Evaluate directory

`/content/hallucination/evaluate`

### Drive results folder

`/content/drive/MyDrive/BHRAM-Results`

## 10. Important Observation About Prompt Mode

This caused confusion and must be explicit.

When running:

`--prompt-mode english`

it does **not** mean:

- only English dataset rows are processed

It means:

- all selected multilingual dataset rows are processed
- but the **English prompt template** is used for every row

So if `chrono_questions` contains English, Hindi, Gujarati, Marathi, Odia rows, all of them still run.

`--prompt-mode native` means:

- each row uses the prompt template matching its own language

This matters for cross-lingual comparison.

## 11. Important Observation About `--limit`

Another major confusion point:

`--limit` applies to **rows**, not unique question IDs.

In the BHRAM-IL multilingual benchmark:

- one base question appears once per language

So:

- `1000` rows is **not** `1000` unique questions
- for 5-language chronology, `1000` rows is roughly `200` base question IDs

For `chrono_questions`, the checked-in 10K file actually contains:

- `980` rows total
- `196` unique question IDs

So if `--limit 1000` is given for `chrono_questions`, it only runs the available `980` rows.

## 12. Category Counts Found in the 10K Benchmark

Using `dataset/BHRAM_IL_10K/dataset_10k.jsonl`, these counts were observed:

- `chrono_questions`: `980` rows, `196` unique qids
- `factual_questions`: `1950` rows, `390` unique qids
- `indian_questions`: `1135` rows, `227` unique qids
- `maths_questions`: `875` rows, `175` unique qids
- `ner_questions`: `805` rows, `805` unique qids
- `reasoning_questions`: `705` rows, `141` unique qids
- `semantically_incorrect_questions`: `1825` rows, `366` unique qids
- `true_false_questions`: `985` rows, `197` unique qids
- `word_ordering_questions`: `1005` rows, `201` unique qids

## 13. Meaning of the Evaluation Fields

This is especially important for `chrono_questions`.

The main evaluated fields are:

- `match.primary`
- `match.fuzzy`
- `scores.primary`
- `scores.fuzzy`

### For chronology

The evaluator uses `ChronologicalEvaluator` in [evaluate/evaluation.py](/home/guru/Documents/hallucination/BHRAM-IL/evaluate/evaluation.py:657).

It is **not** standard classification F1.

### `Primary`

For chronology, `primary` means:

- exact event overlap is found between predicted and reference events
- only the common exact events are considered
- their order is compared using Kendall's Tau

### `Fuzzy`

For chronology, `fuzzy` means:

- predicted events are fuzzily mapped to reference events
- order is then compared again

### Important consequence

The chronology metrics are somewhat lenient:

- partial answers can still score well if the matched subset is in the correct order

That became a major issue during manual inspection.

## 14. Critical Evaluation Problem Found in Chronology

This is the single most important scientific caution discovered so far.

In [evaluate/evaluation.py](/home/guru/Documents/hallucination/BHRAM-IL/evaluate/evaluation.py:701), chronology `primary` scoring:

- only evaluates the ordering of exact-matching overlapping events
- does **not** require all expected events to be present

So a predicted answer can:

- omit one or more events
- still receive `primary = True`
- still receive `primary_score = 1.0`

This was observed in the sample inspected from the pilot.

So before reporting final chronology results:

- do not blindly trust `primary match rate`
- manually inspect outputs
- interpret chronology scores carefully

## 15. Additional Chronology Metric Issue

The fuzzy chronology metric also has an interpretability issue.

Because the code in [evaluate/evaluation.py](/home/guru/Documents/hallucination/BHRAM-IL/evaluate/evaluation.py:731) eventually calls Kendall Tau, and Kendall Tau returns `0.0` when lengths differ in [evaluate/evaluation.py](/home/guru/Documents/hallucination/BHRAM-IL/evaluate/evaluation.py:216), partial outputs often collapse to:

- `fuzzy_match = False`
- `fuzzy_score = 0.5`

That `0.5` is often just a normalized neutral value after failed ordering comparison, not an intuitive “half correct.”

So fuzzy chronology scores should also be interpreted carefully.

## 16. Formatting / Output Problems Observed in the Pilot

During manual inspection of the 100-row chronology pilot, several systematic issues appeared:

- answers were sometimes truncated
- answers sometimes stopped before listing all events
- some answers contained malformed or incomplete `<answer>` tags
- multilingual outputs, especially Odia, sometimes degraded into corrupted transliteration-like text

This was one reason the token budget was increased from `96` to `192`.

## 17. Token Budget Change Already Recommended

Earlier generation used:

- `--max-new-tokens 96`

This was found to be too small for multilingual chronology answers.

Recommended baseline moving forward:

- `--max-new-tokens 192`

This is especially important for:

- chronology
- long event names
- multilingual outputs

## 18. Baseline Experiments That Were Planned

The intended baseline sequence became:

1. `Qwen` + `english prompt mode` + around `1000` rows for chronology
2. `Qwen` + `native prompt mode` + matched run size for chronology
3. compare the two evaluated JSONL files

This is not the final paper benchmark.

This is just the baseline experiment for the existing BHRAM-IL setup.

## 19. What Is Reported as Already Done

From the user's report:

- the `english` prompt `1k` chronology run completed and was evaluated
- the `native` prompt `1k` chronology run was started
- the `native` run got interrupted because of Colab resource/session limits

This interruption is not a disaster because the generator supports resume from partial output JSONL.

## 20. Resume Behavior for Interrupted Runs

The resume logic is implemented in:

- [run/generate_response_hf.py](/home/guru/Documents/hallucination/BHRAM-IL/run/generate_response_hf.py:321)
- [run/generate_response_hf.py](/home/guru/Documents/hallucination/BHRAM-IL/run/generate_response_hf.py:400)

It works by:

- reading existing output JSONL
- collecting existing `(question_id, language)` pairs
- skipping already completed rows

This means a friend can continue the interrupted native run as long as:

- the partial output JSONL is available in Drive
- the same repo version is used
- the same command is rerun

## 21. Best Way for Friend to Continue Interrupted Colab Runs

Do **not** commit large JSONL outputs to GitHub.

Instead:

- share the Google Drive folder or copy the JSONL into the friend's Drive
- clone the repo from GitHub
- rerun the same generation command with the same output path

This is safe and intended.

## 22. Colab Commands Used Earlier

### Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Clone repo

```python
%cd /content
!git clone https://github.com/Guru-preeth10/hallucination.git
```

### Or pull if already cloned

```python
%cd /content/hallucination
!git pull
```

### Go to run folder

```python
%cd /content/hallucination/run
```

### Install packages

```python
!pip install -q -r colab_requirements.txt
!pip install -q torch fasttext-wheel bert-score nltk huggingface_hub
```

### Create results folder

```python
!mkdir -p /content/drive/MyDrive/BHRAM-Results
```

## 23. Example Generation Commands Used

### English prompt chronology baseline

```python
%cd /content/hallucination/run
!python generate_response_hf.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset ../dataset/BHRAM_IL_10K/dataset_10k.jsonl \
  --category chrono_questions \
  --limit 1000 \
  --prompt-mode english \
  --quantization 4bit \
  --max-new-tokens 192 \
  --output /content/drive/MyDrive/BHRAM-Results/output.chrono.qwen.english.1000.jsonl
```

### Evaluate English prompt chronology baseline

```python
%cd /content/hallucination/evaluate
!python evaluation.py \
  --input /content/drive/MyDrive/BHRAM-Results/output.chrono.qwen.english.1000.jsonl \
  --output /content/drive/MyDrive/BHRAM-Results/evaluated.chrono.qwen.english.1000.jsonl
```

### Native prompt chronology baseline

```python
%cd /content/hallucination/run
!python generate_response_hf.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset ../dataset/BHRAM_IL_10K/dataset_10k.jsonl \
  --category chrono_questions \
  --limit 1000 \
  --prompt-mode native \
  --quantization 4bit \
  --max-new-tokens 192 \
  --output /content/drive/MyDrive/BHRAM-Results/output.chrono.qwen.native.1000.jsonl
```

### Evaluate native prompt chronology baseline

```python
%cd /content/hallucination/evaluate
!python evaluation.py \
  --input /content/drive/MyDrive/BHRAM-Results/output.chrono.qwen.native.1000.jsonl \
  --output /content/drive/MyDrive/BHRAM-Results/evaluated.chrono.qwen.native.1000.jsonl
```

## 24. Meaning of the Final 5K Benchmark Decision

This was clarified later and is important.

The final extension benchmark is **not**:

- 5,000 unique question IDs
- 7 languages

The final extension benchmark **is**:

- `5,000 total multilingual rows`
- `1,250 unique question IDs`
- `4 languages` only:
  - English
  - Hindi
  - Kannada
  - Tamil

That means:

- `1250 x 4 = 5000 total rows`

This is the intended final benchmark design.

## 25. Alignment Rule for the Final 5K Benchmark

For the final benchmark:

- the same `question_id` must exist in all four languages
- the benchmark must be directly aligned across languages

For each aligned question:

- preserve `question_id`
- preserve `category`
- preserve shared metadata
- preserve the semantic meaning

Translate only the language-dependent content:

- `question`
- `expected`, where appropriate

Dates, numbers, named entities, and chronological order must remain unchanged in meaning.

## 26. Important Clarification About What “Preserve Metadata” Means

Not every field can be copied literally.

What should be preserved:

- `question_id`
- `category`
- `domain`
- benchmark identity

What necessarily changes:

- `language`
- `question`
- translated `expected`
- language-specific IDs such as `language_id`
- language-specific `unique_id` if final schema uses one

So this is:

- preserve aligned benchmark identity
- rebuild language-specific fields systematically

## 27. Recommendation About Which Categories to Use for Final Kannada/Tamil Extension

Do **not** use every category.

Some categories are structurally unsafe for naive translation.

### Recommended safe categories

- `factual_questions`
- `chrono_questions`
- `indian_questions`
- `maths_questions`
- `reasoning_questions`
- `true_false_questions`

### Not recommended initially

- `ner_questions`
- `word_ordering_questions`
- `semantically_incorrect_questions`

### Why unsafe categories are risky

`ner_questions`

- token-level labeling
- translation changes tokenization and label boundaries

`word_ordering_questions`

- question structure itself depends on word order scrambling
- machine translation can destroy or normalize that structure

`semantically_incorrect_questions`

- validity/invalidity may change during translation
- a translated bad question may accidentally become valid

## 28. Recommended Exact Composition for 1250 Unique Question IDs

A clean recommended final composition is:

- all `390` factual qids
- all `196` chrono qids
- all `227` indian qids
- all `175` maths qids
- all `141` reasoning qids
- `121` true_false qids

Total:

- `390 + 196 + 227 + 175 + 141 + 121 = 1250 unique qids`

This is a strong recommendation because:

- it reaches exactly the needed size
- it avoids structurally dangerous categories
- it keeps the benchmark aligned and interpretable

## 29. Where Subset Building Should Happen

Best split decided earlier:

### Local / repo side

Use local code for:

- selecting the `1250` aligned qids
- writing subset JSONL files
- maintaining benchmark schema
- adding Kannada/Tamil support in code

### Colab side

Use Colab for:

- actual translation into Kannada and Tamil
- later model generation and evaluation

Reason:

- subset construction is lightweight
- translation is the heavy part
- Colab is better suited for translation at scale

## 30. Should Translation Happen Here or in Colab?

Answer:

- subset preparation: here
- translation execution: Colab

That remains the recommended approach.

## 31. Code Changes Needed for Kannada and Tamil Support

### In generation

[run/generate_response_hf.py](/home/guru/Documents/hallucination/BHRAM-IL/run/generate_response_hf.py:38) currently has:

- English
- Gujarati
- Hindi
- Marathi
- Odia

For the final 4-language benchmark, it will need support for:

- English
- Hindi
- Kannada
- Tamil

So update:

- `LANGUAGES`
- `LANGUAGE_ID_MAP`

and create prompt folders:

- `run/prompts/kannada/`
- `run/prompts/tamil/`

### In evaluation

[evaluate/evaluation.py](/home/guru/Documents/hallucination/BHRAM-IL/evaluate/evaluation.py:275) currently lacks Kannada/Tamil in `LANGUAGE_MAP`.

Add:

- `"kannada": "kn"`
- `"tamil": "ta"`

Good news:

- script detection already knows Kannada and Tamil in [evaluate/evaluation.py](/home/guru/Documents/hallucination/BHRAM-IL/evaluate/evaluation.py:377)
- script-family mapping already includes `kn` and `ta` in [evaluate/evaluation.py](/home/guru/Documents/hallucination/BHRAM-IL/evaluate/evaluation.py:499)

So evaluation support is partly there already.

## 32. What the Friend Should Not Do

Do not:

- rerun everything from scratch unnecessarily
- commit large JSONL benchmark outputs to GitHub
- trust chronology primary scores blindly
- translate unsafe categories first
- assume `limit 1000` means `1000 unique questions`
- assume `prompt-mode english` means English-only rows

## 33. What the Friend Should Do Next

This is the recommended exact continuation order.

### Phase A: Finish existing baseline work

1. Confirm the English prompt `1000` chronology run and evaluated file exist in Drive.
2. Resume and finish the interrupted native prompt `1000` chronology run.
3. Evaluate the native run.
4. Compare English-prompt vs native-prompt results on chronology.
5. Manually inspect failures, especially Hindi and Odia-like issues seen earlier in the original 5-language baseline.

### Phase B: Prepare final 4-language benchmark

6. Build the `1250` unique question ID subset from safe categories only.
7. Create aligned English and Hindi source subset rows.
8. Translate those aligned rows into Kannada and Tamil.
9. Merge all four languages into one final `5000`-row JSONL benchmark.
10. Add Kannada/Tamil prompt templates and code support.

### Phase C: Final evaluation runs

11. Run final model experiments on the 4-language 5K benchmark.
12. Evaluate outputs.
13. Produce cross-lingual comparisons.

## 34. Comparison Logic for Baseline Runs

When comparing:

- English prompt mode
- native prompt mode

for chronology, compare these:

- primary match rate
- fuzzy match rate
- average primary score
- average fuzzy score
- per-language breakdown

But always remember:

- chronology primary is lenient for partial ordered answers

So manual failure inspection is mandatory.

## 35. Manual Quality Checks That Should Be Repeated

Before trusting a larger run, inspect:

- answer truncation rate
- malformed `<answer>` tag rate
- predicted event count vs expected event count
- language corruption cases
- per-language failure examples

Especially inspect:

- Hindi
- Kannada
- Tamil

later in the 4-language extension benchmark.

## 36. Most Likely Scientific Interpretation Already Emerging

From the earlier pilot behavior, it appears likely that:

- English prompts may work acceptably for English and some languages
- native prompts may improve some language-specific behavior
- chronology scoring may overestimate correctness when answers are partial but ordered

This means the research conclusions should talk about:

- answer completeness
- formatting reliability
- language fidelity
- not only the benchmark score

## 37. Practical Note About Sharing the Partial Native Run

If the native run was interrupted and the user wants a friend to continue it:

best options are:

- share the Drive folder with the partial output JSONL
- or copy the partial JSONL into the friend's Drive

Then the friend can rerun the same command and resume automatically.

## 38. Suggested Immediate To-Do for the Next Codex

If another Codex session continues from this file, the best next concrete tasks are:

1. Ask for or locate the current Drive output filenames for:
   - English 1000 chronology output
   - English 1000 chronology evaluated output
   - Native 1000 chronology partial or final output
   - Native 1000 chronology evaluated output if already created
2. Verify whether the native run is complete or still partial.
3. If partial, provide resume cell(s) only.
4. Once both are complete, generate a comparison summary.
5. Then build the `1250` unique-qid subset plan in code.

## 39. Bottom-Line Summary

What is already achieved:

- Colab-friendly generator created
- Colab workflow validated
- generation and evaluation work
- baseline chronology runs started successfully
- chronology evaluator weaknesses identified
- final benchmark design clarified

What remains:

- finish the interrupted native baseline run
- compare English vs native prompt baselines
- build final `1250` aligned unique-qid subset
- translate to Kannada and Tamil
- add Kannada/Tamil code and prompts
- run final 4-language 5K benchmark

## 40. If You Need a One-Sentence Mission

Continue the existing Colab baseline carefully, do not lose the interrupted native run, and then build a clean aligned 4-language `1250 qid / 5000 row` benchmark using only translation-safe categories before final experiments.
