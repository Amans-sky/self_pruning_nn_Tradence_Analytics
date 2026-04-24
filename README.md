# Self-Pruning Neural Network

> A feed-forward network that **learns to prune itself during training** — no post-training pruning step, no handcrafted masks. Weights decide their own fate.

```
Epoch  1: acc=10.2%  sparsity= 0.0%  [training begins]
Epoch 10: acc=48.3%  sparsity=12.4%  [gates start committing]
Epoch 20: acc=51.1%  sparsity=31.7%  [pruning in full swing]  
Epoch 30: acc=49.2%  sparsity=38.4%  [converged — 38% of weights removed, -3% accuracy]
```

---

## How It Works

Every weight `w_ij` in the network has a twin: a **gate_score** `s_ij` (same shape). During the forward pass:

```
gate   = σ(s / τ)             ← sigmoid, temperature-scaled  
output = (weight ⊙ gate) · x  ← pruned weight matrix applied to input
```

Training minimises:

```
Loss = CrossEntropy(logits, labels)  +  λ · Σ(gates) / N_gates
            ↑                                    ↑
    want accurate predictions          want most gates = 0
```

**The competition:** useful weights get their gate_score pushed up by the CE gradient; useless weights only feel the L1 pressure pushing them to zero. λ sets the threshold.

## Architecture

```
Input(3072) → PrunableLinear(3072,256) → BN → GELU → Dropout
            → PrunableLinear(256,128)  → BN → GELU → Dropout
            → PrunableLinear(128,64)   → BN → GELU → Dropout
            → PrunableLinear(64,10)    → logits
```

`PrunableLinear` is a complete from-scratch re-implementation (no `nn.Linear` used). Gradients flow through both `weight` and `gate_scores` via standard autograd.

## Results (CIFAR-10, 30 epochs)

| λ       | Test Accuracy | Sparsity | Active Weights      |
|---------|:-------------:|:--------:|:-------------------:|
| `0.0`   | 52.48%        | 0.00%    | 828,032 / 828,032   |
| `1e-5`  | 51.83%        | 12.10%   | 727,840 / 828,032   |
| `1e-4`  | 49.17%        | 38.40%   | 510,067 / 828,032   |
| `5e-4`  | 45.23%        | 67.20%   | 271,594 / 828,032   |
| `1e-3`  | 38.12%        | 88.70%   | 93,567 / 828,032    |

**Sweet spot: λ = 1e-4** → 38% compression for only a ~3% accuracy drop.

> *Baseline accuracy of ~52% is expected for a fully-connected MLP on CIFAR-10 (no convolutions). CNNs achieve 90%+ by exploiting spatial structure; this project uses only linear layers per the task specification.*

## Quick Start

```bash
pip install torch torchvision matplotlib numpy seaborn

# Full λ sweep (5 values, ~30 min on CPU)
python train.py

# Single λ
python train.py --lambda_val 1e-4 --epochs 30

# Generate all figures
python visualise.py
```

## Files

```
self_pruning_nn/
├── prunable_net.py   ← PrunableLinear + SelfPruningNet (core implementation)
├── train.py          ← Training pipeline with argument parsing & metrics logging
├── visualise.py      ← 5-figure publication-quality visualisation suite
├── REPORT.md         ← Full technical report with theory + analysis
└── results/
    ├── all_results.json            ← Serialised training history
    └── figures/
        ├── gate_distributions.png           ← Required: bimodal gate histograms
        ├── sparsity_accuracy_tradeoff.png   ← Pareto frontier curve
        ├── training_curves.png              ← Per-λ accuracy + sparsity dynamics
        ├── layer_sparsity_heatmap.png       ← Per-layer sparsity breakdown
        └── temperature_annealing.png        ← τ annealing vs sparsity growth
```

## Key Engineering Decisions

| Decision | Why |
|----------|-----|
| `gate_scores` initialised to 0 | `σ(0) = 0.5` gives maximum gradient sensitivity — equal footing at t=0 |
| Temperature annealing τ: 1.0→0.1 | Soft gates early (stable), hard binary decisions late |
| Separate gate learning rate (5× higher) | CE gradients dominate weight updates; gates need a boost |
| Normalised sparsity loss (`/ N_gates`) | Makes λ scale-invariant across architectures |
| Gradient clipping (norm=1.0) | Prevents catastrophic gate destruction in early instability |
| BatchNorm after each layer | Stable training even as gate sparsity disrupts activation scale |

## Why L1 on Sigmoid Gates → Sparsity

The L1 penalty `Σ σ(s_ij)` has a gradient `σ(s) · (1−σ(s)) / τ` that **does not vanish as s → −∞**.  
Unlike L2 (whose gradient → 0 near 0), L1 maintains constant pressure on negative scores — pushing them further negative, making sigmoid(s) → 0 exactly.

This is the same reason LASSO produces exact zeros while Ridge regression produces only small values.

---

*Submitted for Tredence AI Engineering Internship — 2025 Cohort*
