# The Latent Truth
### Investigating the Knowledge-Behavior Gap in Language Models via Probing and Activation Steering

This project studies a simple question:

> Can language models internally represent correct logical information even when they generate incorrect outputs?

Across Gemma-2-2B, Phi-2, and Mistral-7B, I observe a consistent pattern where logical information becomes highly linearly separable in intermediate transformer representations long before behavioral reasoning accuracy improves.

The project combines:
- hidden-state probing
- residual stream analysis
- activation steering
- causal intervention experiments
- behavioral evaluation across prompting conditions

The associated paper is currently under review at an ICLR workshop.

---

# Core Idea

Language models often fail logical reasoning tasks behaviorally even when they appear to contain the relevant information internally.

This repository investigates that discrepancy through what I call the:

## Knowledge-Behavior Gap

The central observation is:

- latent logical representations saturate early
- behavioral outputs remain substantially worse

In multiple models, probing accuracy reaches near 100% while generation accuracy remains around ~65%.

This suggests that reasoning failures may not always arise from missing representations, but from failures in translating those representations into autoregressive behavior.

---

# Experimental Pipeline

The workflow consists of five main stages.

## 1. Synthetic Logic Generation

Synthetic logical deduction clusters are generated to reduce memorization and linguistic priors.

Tasks include:
- deductive reasoning
- abductive reasoning
- solver-augmented settings
- chain-of-thought prompting

---

## 2. Behavioral Evaluation

Models are evaluated under:
- Direct prompting
- Chain-of-Thought (CoT)
- Solver-Augmented prompting

Behavioral outputs are parsed and evaluated against ground-truth logical targets.

---

## 3. Hidden-State Extraction

Residual stream activations are extracted across multiple transformer layers for:
- Gemma-2B
- Phi-2
- Mistral-7B

Intermediate activations are saved for probing analysis.

---

## 4. Linear Probing

Logistic regression probes are trained over hidden representations to measure:
- latent logical separability
- layer-wise representation dynamics
- emergence of logical information

---

## 5. Activation Steering

Steering vectors are extracted from probe directions and injected back into the residual stream during generation.

This tests whether latent representations are causally implicated in downstream behavior.

---

# Main Findings

## Behavioral Accuracy

| Method | Accuracy |
|---|---|
| Direct | 62.4% |
| Solver-Augmented | 64.0% |
| Chain-of-Thought | 65.6% |

---

## Internal Probe Accuracy (Phi-2)

| Layer | Probe Accuracy |
|---|---|
| 0 | 50% |
| 8 | 94-98% |
| 16 | 100% |
| 24 | 100% |

Latent logical information becomes highly linearly separable substantially before behavioral reasoning saturates.

---

## Activation Steering

Moderate steering interventions partially recover failed generations across models.

Matched random-vector controls fail to reproduce these gains, suggesting causal specificity rather than generic perturbation effects.

---

# Repository Structure

```text
project/
│
├── notebooks/
│   └── run3.ipynb
│
├── data/
│   ├── clusters.json
│   ├── phase2_direct.json
│   ├── phase2_cot.json
│   └── phase2_solver_augmented.json
│
├── tensors/
│   └── hidden-state activations
│
├── paper/
│   └── manuscript draft
│
└── README.md
```

---

# Models

Experiments were run on:
- microsoft/phi-2
- google/gemma-2-2b-it
- mistralai/Mistral-7B

using Hugging Face Transformers.

---

# Requirements

```bash
pip install transformers accelerate bitsandbytes torch scikit-learn pandas numpy tqdm z3-solver
```

---

# Running the Experiments

## Generate synthetic logic clusters

```python
gen = ProfessionalClusterGenerator()
gen.build_dataset(250)
```

---

## Run behavioral evaluation

```python
run_phase_2_production(clusters, model, tokenizer, "direct")
```

Options:
- `"direct"`
- `"cot"`
- `"solver_augmented"`

---

## Extract hidden states

```python
run_phase_3_individual(clusters, model, tokenizer, ["direct"])
```

---

## Train probes

```python
train_balanced_probe("direct", 16)
```

---

## Run steering interventions

```python
run_steered_test(model, tokenizer, failures, alpha=1.0)
```

---

# Important Notes

This repository contains active research code and is still evolving alongside the paper submission.

The experiments here should be interpreted carefully:
- probing does not imply direct accessibility during generation
- activation steering demonstrates causal influence within this experimental setup
- results may not generalize beyond controlled logical reasoning environments

The goal of this project is not to claim that models "know truth" universally, but to study the relationship between latent representations and behavioral outputs.

---

Harika Mahapatra  
harikamahapatra@gmail.com
