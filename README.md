# Cross-Axis Capping

An exploratory study on separating jailbreak detection from correction in LLM activation space.

## What is this?

When a large language model (LLM) generates text, its internal "hidden states" carry directional signals that indicate whether it's about to comply with or refuse a request. Safety researchers have found that you can nudge a model toward refusal by **capping** these signals along a known direction (the "assistant axis").

The standard approach uses **one axis** for both detecting a jailbreak and correcting the model's behavior. This project asks: **what if we use two separate axes instead?**

- **Assistant-axis capping** (the standard): uses the same direction to spot a jailbreak and to push the model toward refusal.
- **Cross-axis capping** (the new idea): spots the jailbreak on the assistant axis, but corrects using a different "compliance axis" derived from PCA on the model's own refusing vs. compliant activations.

We test both methods on **Qwen/Qwen3-32B** across 100 jailbreak prompts and 50 benign prompts, then use Claude Sonnet as an automated judge to classify each output.

## What did we find?

| Metric | Assistant-Axis | Cross-Axis |
|---|---|---|
| Jailbreak refusal rate | 28% | **55%** |
| Jailbreak compliance rate | 59% | **29%** |
| Benign output preserved | 62% | 62% |
| Benign output degraded | 6% | 6% |

Cross-axis capping nearly doubles the refusal rate while halving compliance -- and it does this with **no additional cost** to benign output quality.

## Repository Structure

```
crosscap_experiment.py      Core library (model loading, capping hooks, axis math, generation)
run_crosscap.py             Main script that orchestrates the experiment
reclassify_refusals.py      Post-hoc LLM judge (sends outputs to Claude Sonnet for labeling)
run_parallel.sh             Convenience script for multi-GPU runs
analyze_csvs.py             Prints summary stats from the reclassified CSVs
analyze_results.sh          Shell wrapper around the analysis
check_original_csvs.py      Quick sanity check on raw CSVs before reclassification
requirements.txt            Python dependencies
Final Results/              CSVs and metadata from the completed experiment
final_analysis_summary.txt  Plain-text summary of all results
```

## Prerequisites

- **Python 3.10+**
- **A CUDA GPU** with enough memory for Qwen3-32B (80 GB+ VRAM recommended)
- **An Anthropic API key** (only needed for the LLM-judge reclassification step)

## Setup

Follow these steps in order to avoid dependency errors.

### 1. Clone the repository

```bash
git clone <repo-url> && cd cross_capping
```

### 2. Install the HuggingFace CLI

The experiment downloads models, datasets, and pre-computed axes from HuggingFace, so you need the CLI installed and authenticated first.

**Linux / macOS (bash):**
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

**Windows (PowerShell):**
```powershell
-ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

### 3. Log in to HuggingFace

```bash
hf auth login
```

This will prompt you for an access token. You can create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 4. Install Flash Attention

Flash Attention needs to be installed separately before the rest of the dependencies, otherwise the build can fail:

```bash
pip install flash-attn --no-build-isolation
```

### 5. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Set your Anthropic API key (optional)

Only needed if you plan to run the LLM-judge reclassification step:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Running the Experiment

### 1. Quick smoke test (5 prompts, fastest way to verify everything works)

```bash
python run_crosscap.py --preset sanity
```

### 2. Full run on a single GPU

```bash
python run_crosscap.py --preset full
```

### 3. Full run across multiple GPUs (recommended)

The easiest way:

```bash
./run_parallel.sh full 4    # uses 4 GPUs
```

Or do it manually in three steps:

```bash
# Step 1 -- warmup: downloads everything, computes axes and thresholds (run once)
python run_crosscap.py --preset full --warmup

# Step 2 -- generate: run one chunk per GPU in parallel
CUDA_VISIBLE_DEVICES=0 python run_crosscap.py --preset full --chunk 0/4 &
CUDA_VISIBLE_DEVICES=1 python run_crosscap.py --preset full --chunk 1/4 &
CUDA_VISIBLE_DEVICES=2 python run_crosscap.py --preset full --chunk 2/4 &
CUDA_VISIBLE_DEVICES=3 python run_crosscap.py --preset full --chunk 3/4 &
wait

# Step 3 -- merge: combine chunk CSVs into the four final output files
python run_crosscap.py --preset full --merge
```

### 4. Classify outputs with the LLM judge

After generation, have Claude Sonnet label each output:

```bash
python reclassify_refusals.py            # full run
python reclassify_refusals.py --resume   # pick up where you left off
python reclassify_refusals.py --summary-only  # just print stats from existing labels
```

### 5. View summary statistics

```bash
python analyze_csvs.py
```

## How the Pipeline Fits Together

```
 WARMUP
 ──────
   Download model, datasets, and assistant axis from HuggingFace
   Collect activations from 50 "refusing" + 50 "compliant" runs
   Compute PCA compliance axis and per-layer detection thresholds
   Save everything to warmup.pt
          │
          ▼
 GENERATION
 ──────────
   For each of the 150 prompts (100 jailbreak + 50 benign):
     1. Generate a baseline response (no capping)
     2. Generate an assistant-capped response
     3. Generate a cross-capped response
   Record which layers fired and the per-layer projection values
          │
          ▼
 MERGE
 ─────
   Combine per-chunk CSVs into four final files:
     assistant_cap_jailbreak.csv    assistant_cap_benign.csv
     cross_cap_jailbreak.csv        cross_cap_benign.csv
          │
          ▼
 RECLASSIFICATION (LLM JUDGE)
 ────────────────────────────
   Send each capped output to Claude Sonnet for labeling:
     Jailbreak outputs  →  refusal / compliance / partial_refusal / degraded
     Benign outputs     →  unchanged / false_refusal / degraded
   Produces *_reclassified.csv files with llm_label column
```

## Presets

| Preset | Jailbreak prompts | Benign prompts | Max tokens | When to use |
|---|---|---|---|---|
| `sanity` | 5 | 10 | 64 | Smoke test -- does it run at all? |
| `small` | 20 | 20 | 128 | Development and debugging |
| `full` | 100 | 50 | 256 | The real experiment |
| `full_meandiff` | 100 | 50 | 256 | Variant using mean-difference axis instead of PCA |

## Key Configuration

These live at the top of `run_crosscap.py`:

| Parameter | Default | What it controls |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-32B` | Which model to cap |
| `CAP_LAYERS` | L46--L53 (8 layers) | Where in the network capping is applied (72--84% depth) |
| `SEED` | 42 | Random seed for reproducibility |
| `AXIS_METHOD` | `pca` | How the compliance axis is built (`pca` or `mean_diff`) |

## How Capping Works (briefly)

During text generation the model builds up a hidden state **h** at each layer. Capping modifies **h** in-place:

**Single-axis (assistant) capping:**
> If **h** projects below a threshold on the assistant axis **v**, push it back:
> `h = h - v * min(dot(h, v) - threshold, 0)`

**Cross-axis capping:**
> **Detect** on the assistant axis (same threshold check), but **correct** along a separate compliance axis **c**:
> `h = h - c * min(dot(h, c) - threshold_c, 0)`

The compliance axis is built by running PCA on hidden states the model produces when it refuses harmful requests vs. when it complies.

## Datasets

All downloaded automatically from HuggingFace:

| Dataset | Role in the experiment |
|---|---|
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | Bare harmful goals -- used to collect "refusing" activations |
| [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) | Adversarial jailbreak prompts for evaluation, plus "compliant" activations from the train split |
| [assistant-axis-vectors](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors) | Pre-computed assistant axis at layer 53 |

## Output CSV Format

Each row is one prompt. Key columns:

| Column | Description |
|---|---|
| `prompt_text` | The input prompt |
| `baseline_text` | What the model said with no intervention |
| `{method}_cap_applied` | Did the capping fire? (`Yes` / `No`) |
| `{method}_cap_layers` | Which layers fired (e.g. `L46,L47,L48`) |
| `{method}_cap_text` | What the model said with capping active |
| `llm_label` | Judge's classification (added during reclassification) |

## License

Research use. See repository for details.
