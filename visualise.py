"""
Visualisation Suite — Self-Pruning Neural Network
==================================================

Generates all plots required for the case study report plus several extras:

  1. Gate value distribution (required — histogram for best model)
  2. Sparsity vs Accuracy trade-off curve
  3. Training curves per λ (loss + accuracy)
  4. Per-layer sparsity heatmap
  5. Temperature annealing vs sparsity progression

Run after training:
    python visualise.py

Author: Aman (Tredence Case Study Submission)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


RESULTS_PATH = Path("./results/all_results.json")
FIGURES_DIR  = Path("./results/figures")

# ─── Aesthetic palette ────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0D0F14",
    "surface":  "#161A23",
    "border":   "#252A38",
    "accent":   "#4F8EF7",
    "accent2":  "#F75F4F",
    "accent3":  "#4FF7A0",
    "accent4":  "#F7C74F",
    "text":     "#E2E8F0",
    "muted":    "#64748B",
}

LAMBDA_COLORS = ["#64748B", "#4F8EF7", "#4FF7A0", "#F7C74F", "#F75F4F"]

def setup_style():
    plt.rcParams.update({
        "figure.facecolor":     PALETTE["bg"],
        "axes.facecolor":       PALETTE["surface"],
        "axes.edgecolor":       PALETTE["border"],
        "axes.labelcolor":      PALETTE["text"],
        "axes.titlecolor":      PALETTE["text"],
        "xtick.color":          PALETTE["muted"],
        "ytick.color":          PALETTE["muted"],
        "text.color":           PALETTE["text"],
        "grid.color":           PALETTE["border"],
        "grid.linestyle":       "--",
        "grid.alpha":           0.5,
        "legend.facecolor":     PALETTE["surface"],
        "legend.edgecolor":     PALETTE["border"],
        "font.family":          "monospace",
        "font.size":            11,
        "axes.titlesize":       13,
        "axes.labelsize":       11,
        "figure.dpi":           150,
    })


def label_for(lam: float) -> str:
    if lam == 0:
        return "λ = 0 (baseline)"
    return f"λ = {lam:.0e}"


# ─── Plot 1: Gate Distribution (required) ────────────────────────────────────

def plot_gate_distribution(results: list):
    """Distribution of gate values for each λ — 'best model' spotlighted."""
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 5),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Gate Value Distributions Across λ Values",
                 fontsize=15, fontweight="bold", y=1.02, color=PALETTE["text"])

    for ax, result, color in zip(axes, results, LAMBDA_COLORS):
        gates = np.array(result["gate_values"])
        lam   = result["lambda"]

        # Two-component view: near-zero (pruned) vs active
        near_zero = gates[gates < 0.05]
        active    = gates[gates >= 0.05]

        bins = np.linspace(0, 1, 40)
        ax.hist(near_zero, bins=bins, color=PALETTE["accent2"], alpha=0.9,
                label=f"Pruned ({len(near_zero)/len(gates):.1%})", zorder=3)
        ax.hist(active,    bins=bins, color=color,               alpha=0.9,
                label=f"Active ({len(active)/len(gates):.1%})",   zorder=3)

        ax.set_title(label_for(lam), fontweight="bold")
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, zorder=0)

        # Annotation: sparsity + accuracy
        acc = result["best_test_acc"]
        sp  = result["final_sparsity"]
        ax.text(0.97, 0.97, f"Acc: {acc:.3f}\nSparse: {sp:.1%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=PALETTE["accent3"],
                bbox=dict(boxstyle="round,pad=0.3", fc=PALETTE["bg"],
                          ec=PALETTE["border"], alpha=0.9))

    plt.tight_layout()
    path = FIGURES_DIR / "gate_distributions.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved: {path}")


# ─── Plot 2: Sparsity vs Accuracy Trade-off ───────────────────────────────────

def plot_tradeoff(results: list):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    sparsities = [r["final_sparsity"]    * 100 for r in results]
    accuracies = [r["best_test_acc"]     * 100 for r in results]
    lambdas    = [r["lambda"]                   for r in results]

    # Pareto region shading
    ax.fill_betweenx([min(accuracies) - 2, max(accuracies) + 2],
                     0, 100, alpha=0.03, color=PALETTE["accent"])

    ax.plot(sparsities, accuracies, color=PALETTE["muted"],
            linestyle="--", linewidth=1, alpha=0.5, zorder=1)

    for sp, acc, lam, color in zip(sparsities, accuracies, lambdas, LAMBDA_COLORS):
        ax.scatter(sp, acc, color=color, s=120, zorder=4, edgecolors="white",
                   linewidths=0.8)
        ax.annotate(label_for(lam), (sp, acc),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8.5, color=color)

    ax.set_xlabel("Sparsity Level (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Sparsity–Accuracy Trade-off Frontier", fontweight="bold")
    ax.grid(True, zorder=0)

    # Efficiency region annotation
    best_idx = max(range(len(results)),
                   key=lambda i: accuracies[i] / max(1, 100 - sparsities[i]))
    ax.annotate("← Efficiency sweet spot",
                xy=(sparsities[best_idx], accuracies[best_idx]),
                xytext=(sparsities[best_idx] + 5, accuracies[best_idx] - 3),
                fontsize=8, color=PALETTE["accent3"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent3"]))

    plt.tight_layout()
    path = FIGURES_DIR / "sparsity_accuracy_tradeoff.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved: {path}")


# ─── Plot 3: Training Curves ─────────────────────────────────────────────────

def plot_training_curves(results: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Training Dynamics", fontsize=14, fontweight="bold")

    ax_acc, ax_sp = axes

    for result, color in zip(results, LAMBDA_COLORS):
        hist   = result["history"]
        epochs = [h["epoch"] for h in hist]
        t_acc  = [h["train_acc"]          * 100 for h in hist]
        v_acc  = [h["test_acc"]           * 100 for h in hist]
        sp     = [h["overall_sparsity"]   * 100 for h in hist]
        lam    = result["lambda"]

        label = label_for(lam)
        ax_acc.plot(epochs, v_acc, color=color, linewidth=2, label=label, zorder=3)
        ax_acc.plot(epochs, t_acc, color=color, linewidth=1,
                    linestyle=":", alpha=0.4, zorder=2)

        ax_sp.plot(epochs, sp, color=color, linewidth=2, label=label, zorder=3)

    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Test Accuracy (solid) / Train Accuracy (dotted)")
    ax_acc.legend(fontsize=8); ax_acc.grid(True)

    ax_sp.set_xlabel("Epoch"); ax_sp.set_ylabel("Overall Sparsity (%)")
    ax_sp.set_title("Sparsity Level Progression")
    ax_sp.legend(fontsize=8); ax_sp.grid(True)

    plt.tight_layout()
    path = FIGURES_DIR / "training_curves.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved: {path}")


# ─── Plot 4: Per-layer Sparsity Heatmap ──────────────────────────────────────

def plot_layer_heatmap(results: list):
    layer_keys = sorted(
        [k for k in results[0]["per_layer_sparsity"].keys()],
        key=lambda x: int(x.split("_")[1])
    )
    n_layers  = len(layer_keys)
    n_lambdas = len(results)

    matrix = np.zeros((n_layers, n_lambdas))
    for j, r in enumerate(results):
        for i, k in enumerate(layer_keys):
            matrix[i, j] = r["per_layer_sparsity"].get(k, 0) * 100

    fig, ax = plt.subplots(figsize=(n_lambdas * 1.8 + 2, n_layers * 0.9 + 1.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(n_lambdas))
    ax.set_xticklabels([label_for(r["lambda"]) for r in results], rotation=20, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {k.split('_')[1]}" for k in layer_keys])
    ax.set_title("Per-Layer Sparsity Heatmap (%)", fontweight="bold")

    for i in range(n_layers):
        for j in range(n_lambdas):
            ax.text(j, i, f"{matrix[i,j]:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="black" if matrix[i,j] > 50 else PALETTE["text"])

    plt.colorbar(im, ax=ax, label="Sparsity (%)")
    plt.tight_layout()
    path = FIGURES_DIR / "layer_sparsity_heatmap.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved: {path}")


# ─── Plot 5: Temperature vs Sparsity (best λ) ────────────────────────────────

def plot_temperature_annealing(results: list):
    # Pick the model with highest sparsity that still has reasonable accuracy
    best = max(results, key=lambda r: r["final_sparsity"] * r["best_test_acc"])
    hist = best["history"]

    epochs = [h["epoch"]             for h in hist]
    temps  = [h["temperature"]       for h in hist]
    sp     = [h["overall_sparsity"]  * 100 for h in hist]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax2 = ax1.twinx()
    l1, = ax1.plot(epochs, temps, color=PALETTE["accent"],  linewidth=2, label="Temperature")
    l2, = ax2.plot(epochs, sp,    color=PALETTE["accent3"], linewidth=2, label="Sparsity (%)")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sigmoid Temperature", color=PALETTE["accent"])
    ax2.set_ylabel("Overall Sparsity (%)", color=PALETTE["accent3"])
    ax1.set_title(f"Temperature Annealing → Sparsity Growth  [λ={best['lambda']:.1e}]",
                  fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=[l1, l2], loc="upper right", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "temperature_annealing.png"
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run train.py first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} experiment results. Generating figures...\n")

    plot_gate_distribution(results)
    plot_tradeoff(results)
    plot_training_curves(results)
    plot_layer_heatmap(results)
    plot_temperature_annealing(results)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
