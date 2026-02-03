# ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response

<p align="center">
  <img src="pic/teaser.png" width="800" alt="ProAct Teaser" />
</p>

## Abstract

While passive agents merely follow instructions, proactive agents align with higher-level objectives, such as assistance and safety by continuously monitoring the environment to determine when and how to act. However, developing proactive agents is hindered by the lack of specialized resources. To address this, we introduce **ProAct-75**, a benchmark designed to train and evaluate proactive agents across diverse domains, including assistance, maintenance, and safety monitoring. Spanning 75 tasks, our dataset features 91,581 step-level annotations enriched with explicit task graphs. These graphs encode step dependencies and parallel execution possibilities, providing the structural grounding necessary for complex decision-making. Building on this benchmark, we propose **ProAct-Helper**, a reference baseline powered by a MLLM that grounds decision-making in state detection, and leveraging task graphs to enable entropy-driven heuristic search for action selection, allowing agents to execute parallel threads independently rather than mirroring the human's next step.

Extensive experiments demonstrate that ProAct-Helper outperforms strong closed-source models, improving trigger detection mF1 by 6.21%, saving 0.25 more steps in online one-step decision, and increasing the rate of parallel actions by 15.58%.

---

Datasets will be made available in a future release.

## Repository Layout

- `train/`: SFT training code and the single runnable training script.
- `test/onestep_planning/`: One-step planning evaluation code and scripts.
- `test/detection_prediction/`: Detection / prediction evaluation helpers and test runner.
- `module_tools/TFace/recognition/README.md`: Third-party face recognition module README (with a note that data will be released in the future).
- `data/`: Placeholder directory for future data release.

## Quickstart

### Environment

Create a Python environment (3.10+ recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

### Data / Paths (required)

Most training scripts are **data-agnostic** and use environment variables for paths:

- `MODEL_NAME_OR_PATH`: base model directory (e.g. a local HuggingFace model folder)
- `TRAIN_JSON`: training JSONL
- `VAL_JSON`: validation JSONL
- `FRAME_ROOT`: frame directory root
- `ANNOTATION_JSON`: annotation file (if needed by your run)
- `PRIORITY_SCORES_JSON`: priority score file (if needed by your run)

Example:

```bash
export MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-3B-Instruct
export TRAIN_JSON=/path/to/train.jsonl
export VAL_JSON=/path/to/val.jsonl
export FRAME_ROOT=/path/to/frames
export ANNOTATION_JSON=/path/to/all_annotations.json
export PRIORITY_SCORES_JSON=/path/to/priority_score.json
```

### Train (SFT)

Run a standard training job:

```bash
bash train/run_train.sh
```

### Detection / Prediction test

```bash
bash test/detection_prediction/run_l2_sft_test.sh
```

### Evaluate (One-Step Planning)

The main evaluator is `test/onestep_planning/eval_onestep_end2end.py`.

For cached prediction folders (entropy selector):

```bash
bash test/onestep_planning/run_eval_onestep_ours.sh
```

For LLM-based evaluation via OpenAI-compatible endpoints or Gemini, use the provided scripts under `test/onestep_planning/`.

## Notes

- **No secrets in repo**: API keys are not hardcoded; provide them via environment variables.
- **Offline mode**: Some scripts set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` by default. Override them if you want online downloads.

# ProAct
