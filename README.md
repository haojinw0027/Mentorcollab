# Mentorcollab

> This is the official repo for **MENTORCOLLAB: Selective Large-to-Small Inference-Time Guidance for Efficient Reasoning**

A double-model collaborative inference system that dynamically routes between base and expert models for efficient and accurate multi-task language understanding and reasoning.

## Overview

Mentorcollab implements an innovative inference framework that dynamically selects models based on task difficulty:
- **Base Model**: Handles straightforward tasks efficiently
- **Expert Model**: Tackles complex problems for better accuracy
- **Decision Mechanism**: When models disagree, uses entropy analysis or MLP prediction to select the optimal choice

## Architecture

```
Input Prompt
    │
    ▼
Base Model generates token
    │
    ▼
Expert Model generates token (if needed)
    │
    ▼
Decision Point:
  - Tokens match → use base model
  - Tokens differ → consult decision mechanism
    │
    ▼
Generate completion sequence
    │
    ▼
Output answer + statistics
```

## Main Components

### Core Files

| File | Description |
|------|-------------|
| `src/mentorcollab_free.py` | Entropy-based decision version using information entropy to judge model confidence |
| `src/mentorcollab_mlp.py` | MLP-based decision version using a trained neural network for routing |

### Core Class: TripleModelGenerator

- `__init__()`: Initialize models, tokenizers, and MLP (optional)
- `_test_connections()`: Verify vLLM endpoint connections
- `generate_sequence()`: Call vLLM API to generate token sequences
- `get_next_token_with_logits()`: Get next token and probability distribution
- `generate_with_triple_model()`: Main inference loop orchestrating routing logic

## Supported Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| MMLU-Pro | Multiple Choice | Cross-domain knowledge Q&A |
| MATH | Mathematical Reasoning | Math problem solving |
| Minerva | Advanced Math | Complex mathematical reasoning |
| SuperGPQA | Science Q&A | Expert-level science questions |
| Com2 Hard | Legal/Commonsense | Legal case analysis and commonsense reasoning |
| ARC Challenge | Factual Reasoning | Science fact Q&A (free version only) |

## Tech Stack

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **vLLM**: High-performance inference server
- **Datasets**: Hugging Face datasets library
- **NumPy**: Numerical computation

## Usage

### Prerequisites

1. Start vLLM services:
   - Base model: port 8000
   - Expert model: port 8001

### Running Inference

**Entropy-based version:**
```bash
python src/mentorcollab_free.py \
    --base_model_path <base_model_path> \
    --expert_model_path <expert_model_path> \
    --dataset mmlu_pro \
    --output_dir results/
```

**MLP-based version:**

You can download the MLP model from [Hugging Face](https://huggingface.co/SeanWang0027/MentorCollab-MLP).

```bash
python src/mentorcollab_mlp.py \
    --base_model_path <base_model_path> \
    --expert_model_path <expert_model_path> \
    --mlp_path <mlp_model_path> \
    --dataset mmlu_pro \
    --output_dir results/
```


### Key Arguments

| Argument | Description |
|----------|-------------|
| `--base_model_path` | Path to base model |
| `--expert_model_path` | Path to expert model |
| `--dataset` | Evaluation dataset name |
| `--output_dir` | Results output directory |
| `--max_tokens` | Maximum tokens to generate |
| `--mlp_path` | MLP model path (MLP version only) |
| `--mlp_threshold` | MLP decision threshold (MLP version only) |

## Decision Strategies

### Entropy-based Version (Free)
1. Calculate entropy of base model's next token distribution
2. If entropy exceeds threshold, use base model directly
3. Otherwise, compare base and expert model outputs
4. When outputs differ, use entropy analysis to select optimal model

### MLP-based Version
1. Random drop sampling (default 5%)
2. For non-dropped samples: compare base vs expert outputs
3. When outputs differ: extract hidden state features
4. Use trained MLP to predict optimal branch

## Output

Inference results are saved as JSON files containing:
- Generated answer
- Base/expert model token usage statistics
- Confidence scores
- Decision statistics
- Detailed token choice logs

## Features

- **Checkpoint Support**: Resume processing from last saved result
- **Parallel Processing**: Free version supports multi-threaded execution
- **Detailed Logging**: Saves token-level decisions and probabilities
- **Flexible Configuration**: Extensive command-line arguments

## License

[To be added]