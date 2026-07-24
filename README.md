# LLM Fine-Tuning for Structured Data Extraction

Fine-tuned Google's [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it) model to extract structured food and drink data from raw text, built end-to-end with Hugging Face Transformers, TRL, and PyTorch.

## Overview

Given raw text (e.g. an image caption or product description), the model classifies whether it's food/drink related, tags it by type, and extracts structured food and drink items — turning unstructured text into structured output.

### Example

**Input:**

    British Breakfast with baked beans, fried eggs, black pudding, sausages, bacon, mushrooms, a cup of tea and toast and fried tomatoes

**Output:**

    food_or_drink: 1
    tags: fi, di
    foods: British Breakfast, baked beans, fried eggs, black pudding, sausages, bacon, mushrooms, toast, fried tomatoes
    drinks: tea

## Pipeline

1. **Data preparation** — loaded and explored the [FoodExtract-1k](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k) dataset (1,420 samples), formatted into prompt/completion pairs, 80/20 train/test split.
2. **Tokenization** — used the Gemma 3 chat template via `AutoTokenizer`.
3. **Training** — fully fine-tuned `google/gemma-3-270m-it` using Hugging Face `trl.SFTTrainer` (Supervised Fine-Tuning), 3 epochs, on a Google Colab T4 GPU.
4. **Evaluation** — achieved ~98.3% mean token accuracy on the held-out test set. Manual inspection of predictions surfaced a real limitation (see below).
5. **Inference** — built a reusable prediction pipeline with the Hugging Face `pipeline` API.
6. **Deployment** — packaged as an interactive Gradio demo.

## Results

| Metric | Value |
|---|---|
| Mean token accuracy (test set) | 98.3% |
| Training epochs | 3 |
| Base model params | 268,098,176 |
| Training hardware | NVIDIA T4 (Google Colab) |

## Known limitation

Evaluation revealed the model under-predicts `drink_items` relative to `food_items`. Investigating the training data showed only **13% of samples (185/1,420)** contain any drink items — a class imbalance that explains the bias toward misclassifying ambiguous items (e.g. "banana juice") as food. This was found through manual inspection, not just the aggregate accuracy metric, which masked the issue. A natural next step would be addressing this imbalance (oversampling, or a larger dataset such as `FoodExtract-135k`).

## Project structure

- `src/config.py` — model/dataset config
- `src/device_check.py` — GPU/CPU detection
- `src/load_model.py` — base model + tokenizer loading
- `src/explore_data.py` — dataset inspection
- `src/prepare_data.py` — prompt/completion formatting + train/test split
- `src/train.py` — SFTConfig + SFTTrainer training script
- `src/train_dryrun.py` — small-scale sanity check before full training
- `demos/FoodExtract/app.py` — Gradio demo app
- `demos/FoodExtract/requirements.txt` — demo dependencies
- `demos/FoodExtract/README.md` — demo description

## Links

- **Fine-tuned model**: https://huggingface.co/OmarShamakh/FoodExtract-gemma-3-270m-fine-tune-v1
- **Dataset**: https://huggingface.co/datasets/mrdbourke/FoodExtract-1k
- **Base model**: https://huggingface.co/google/gemma-3-270m-it
- **Tutorial reference**: https://www.learnhuggingface.com/notebooks/hugging_face_llm_full_fine_tune_tutorial

## Running locally

```bash
uv sync
uv run python src/prepare_data.py
uv run python demos/FoodExtract/app.py
```