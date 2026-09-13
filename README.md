# Explainable Fault Classification for the Tennessee Eastman Process

A fault detection and explanation pipeline built on the Tennessee Eastman Process (TEP) simulation dataset. A Temporal Fusion Transformer (TFT) classifies process faults from sensor and actuator time series, and the surrounding code turns the model's internal attention into a structured, human readable explanation of *why* it flagged a given fault, using a locally run LLM.

## Why this exists

Fault classifiers in process control are only useful if an operator can trust and act on their output. This project pairs a TFT, which is interpretable by design through its attention mechanism, with a pipeline that:

1. Extracts which input variables and which time steps the model weighted most heavily for a given prediction.
2. Summarizes the trend behavior (slope, volatility, acceleration, and so on) of those variables during the fault window.
3. Packages that summary into a compact JSON report.
4. Hands the report to a small language model running fully on CPU, which turns it into a plain language explanation, optionally grounded against reference material through retrieval.

## The Tennessee Eastman Process

TEP is a widely used chemical process simulation benchmark for fault detection research, covering a reactor, condenser, separator, stripper, and compressor loop. It defines 41 measured variables (`XMEAS_1` to `XMEAS_41`) and 11 manipulated variables (`XMV_1` to `XMV_11`), along with 21 predefined fault types plus normal operation. `inference.py` includes a lookup table (`X_dict`) mapping the raw tag names to readable variable names such as `Reactor_temp` and `Stripper_pressure_valve`, used throughout the explanation layer.

## Pipeline overview

```
TEP simulation data
        │
        ▼
demo_sequences.py  →  onset / steady state / aftermath sequences per fault, saved as parquet
        │
        ▼
inference.py       →  loads the trained TFT checkpoint and runs a forward pass
                       extracts temporal attention, encoder variable attention,
                       and decoder variable attention
        │
        ▼
causality/         →  runs PCMCI over the attended variables to discover
                       significant driver -> target edges, then greedily
                       walks that edge table to find a likely root cause
                       and its propagation path
        │
        ▼
cer_draft.py       →  computes trend metrics (slope, normalized slope, relative
                       change, volatility, acceleration, R²) for the top
                       attended variables and assembles a causal effect report
                       (CER) as JSON: predicted fault, confidence, top
                       contributing variables, top time lags, and their trends
        │
        ▼
model-qa.py + common.py  →  ONNX Runtime GenAI chat loop that consumes the CER
                             (and optionally retrieved context, see `rag/`) to
                             generate a natural language explanation, running
                             entirely on CPU
```

`causality/` holds the causal discovery work that identifies which variable is most likely the root cause of a fault, and traces how it propagates to the others. `helper_code/` holds supporting utilities used across the pipeline.

## Causal root cause analysis

Attention tells you which variables the TFT weighted heavily, but not which one is the actual driver versus a downstream symptom. `causality/` addresses that with two pieces:

**PCMCI discovery** (`causality/pcmci.py`): runs the PCMCI algorithm (via Tigramite, with a partial correlation independence test) over the variables in `df_causal_analysis` up to a maximum lag, extracts the statistically significant driver → target edges (dropping self loops, keeping each pair's strongest lag), and returns them both as a graph plot and as a summary table of `Driver`, `Target`, `Lag`, and `Strength`.

**Greedy path search** (`causality/greedy.py`): takes that edge table and finds the most likely root cause and how the fault propagates from it, on the hypothesis that the true driver is the variable that shows up most often as a `Driver` across the discovered edges.

- Each edge is scored with `weight = |strength| * exp(lam * lag)`, so that stronger and longer lag relationships score higher.
- The variable(s) that appear most often as a driver become root cause candidates; ties are broken by picking the candidate whose edge has the highest weight.
- Starting from that root cause, the algorithm greedily walks forward: at each step it follows the highest weight outgoing edge to a variable not yet visited, and only allows lags less than or equal to the previous step's lag, so the path moves through time in a consistent direction rather than jumping around. The walk ends when it reaches a variable with no valid next step or would revisit one already on the path.

The result is a single causal chain, for example `Reactor_cooling_water_outlet_temp -> Reactor_temp -> Separator_temp`, that reads as a story of how the fault propagated, rather than an unordered list of important variables. This chain is a natural candidate to fold into the CER alongside the trend metrics, so the LLM can explain not just what changed but what most likely caused it.

## Repository layout

| Path | Purpose |
|---|---|
| `inference.py` | Loads the TFT checkpoint, runs inference on a sample fault sequence, and extracts attention based interpretability (temporal, encoder, decoder). |
| `cer_draft.py` | Builds the causal effect report (CER): trend statistics for the top attended variables, packaged as JSON for prompting the LLM. |
| `demo_sequences.py` | Slices the full TEP dataset into per fault demo sequences (onset, steady state, aftermath) for inference and demos. |
| `common.py` | ONNX Runtime GenAI helper functions (config, chat templating, guidance/grammar for structured output, generation options). Adapted from Microsoft's `onnxruntime-genai` examples. |
| `model-qa.py` | Interactive chat loop for running a local, CPU based LLM through ONNX Runtime GenAI, used to turn the CER into an explanation. Adapted from Microsoft's `onnxruntime-genai` examples. |
| `causality/` | Causal discovery: PCMCI based driver/target edge extraction, plus a self implemented greedy path search that picks a likely root cause and traces its propagation path. |
| `rag/` | Retrieval component for grounding explanations in reference material. |
| `helper_code/` | Shared utilities used across the pipeline. |
| `test.py` | Tests. |
| `cpu_compat_checkpoint.ckpt` | TFT checkpoint saved in a CPU loadable format. |

## Status

This is an active work in progress rather than a finished, polished release. `cer_draft.py` is a draft, and several sections of `inference.py` are marked for refactoring (duplicate attention handling, a temporal loop that should be a function). Expect the pipeline to run end to end for a single sample sequence, but treat batch processing, error handling, and the `rag/` and `causality/` integrations as evolving.

## Setup

```bash
git clone https://github.com/lloydakresi/explainable_fault_classification_tep.git
cd explainable_fault_classification_tep
```

Core dependencies (no pinned `requirements.txt` yet, so install recent compatible versions):

- `torch`
- `lightning` (`pytorch-lightning`)
- `pytorch-forecasting`
- `pandas`, `numpy`, `scipy`
- `matplotlib`, `seaborn`
- `onnxruntime-genai`
- `tigramite` (PCMCI)
- `networkx`

You will also need:
- A TEP dataset (the standard Rieth et al. simulation dataset is a common source) exported as the `df_demo.feather` file `demo_sequences.py` expects.
- The `dataset/dataset_parameters.pt` file produced when the `TimeSeriesDataSet` was originally built during training.
- A local ONNX model folder (containing `genai_config.json` and `model.onnx`) for `model-qa.py`.

## Usage

Generate demo fault sequences from the full dataset:

```bash
python demo_sequences.py
```

Run inference and extract attention for a sample fault sequence:

```bash
python inference.py
```

Run PCMCI and the greedy root cause search over the attended variables:

```bash
python causality/pcmci.py
python causality/greedy.py
```

Build the causal effect report for that prediction:

```bash
python cer_draft.py
```

Run the local LLM chat loop to generate an explanation from the CER:

```bash
python model-qa.py -m <path-to-onnx-model-folder> --non_interactive -up "Explain the predicted fault using the attached report."
```

## Acknowledgments

- `common.py` and `model-qa.py` are adapted from Microsoft's `onnxruntime-genai` example scripts (MIT licensed).
- The Tennessee Eastman Process simulation is a long standing benchmark originally developed at Eastman Chemical Company for process control and fault detection research.
