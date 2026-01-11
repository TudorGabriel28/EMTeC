#!/usr/bin/env python
"""
Extended Bayesian Regression Analyses for Eye-Tracking Reading Measures
========================================================================

This script runs two types of analyses:

1. Baseline models with effect sizes in milliseconds:
   - Converts log-scale coefficients to interpretable millisecond effects
   - Creates coefficient plots with ms effect sizes on y-axis

2. Surprisal-focused models with continuous prediction plots:
   - Shows the relationship between surprisal and reading time
   - Creates smooth prediction curves with 95% credible intervals

Usage on Kaggle:
    python extended_baseline_analyses.py --analysis baseline -r FPRT
    python extended_baseline_analyses.py --analysis baseline -r TFT
    python extended_baseline_analyses.py --analysis surprisal -r FPRT
    python extended_baseline_analyses.py --analysis surprisal -r TFT
"""

import argparse
import warnings
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Plot settings
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# Model parameters
WARMUP_ITERATIONS = 1000
DEFAULT_ITERATIONS = 3000
N_CHAINS = 4
RANDOM_SEED = 42

# Original surprisal column names from the data (with hyphens)
SURPRISAL_COLS_ORIGINAL = [
    "surprisal_gpt2",
    "surprisal_opt-1.3b",
    "surprisal_mistral-base",
    "surprisal_phi2",
    "surprisal_llama2-13b",
    "surprisal_pythia-12b",
]

# Renamed columns (underscores instead of hyphens for formula compatibility)
SURPRISAL_COLS = [
    "surprisal_gpt2",
    "surprisal_opt_1_3b",
    "surprisal_mistral_base",
    "surprisal_phi2",
    "surprisal_llama2_13b",
    "surprisal_pythia_12b",
]

# Mapping from original to renamed columns
SURPRISAL_RENAME_MAP = dict(zip(SURPRISAL_COLS_ORIGINAL, SURPRISAL_COLS))

# Short labels for plotting
SURPRISAL_LABELS = {
    "surprisal_gpt2": "GPT-2",
    "surprisal_opt_1_3b": "OPT-1.3B",
    "surprisal_mistral_base": "Mistral",
    "surprisal_phi2": "Phi-2",
    "surprisal_llama2_13b": "LLaMA2-13B",
    "surprisal_pythia_12b": "Pythia-12B",
}

# Reading measure labels
MEASURE_LABELS = {
    "FPRT": "First-Pass Reading Time",
    "TFT": "Total Fixation Time",
    "FFD": "First Fixation Duration",
    "FRT": "First Reading Time",
}


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess the reading measures data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep="\t")

    print(f"  Loaded {len(df):,} observations")
    print(f"  Number of subjects: {df['subject_id'].nunique()}")
    print(f"  Number of items: {df['item_id'].nunique()}")

    df["subject_id"] = df["subject_id"].astype("category")

    # Store original values for back-transformation
    df["zipf_freq_raw"] = df["zipf_freq"].copy()
    df["word_length_raw"] = df["word_length_with_punct"].copy()
    df["surprisal_raw"] = df["surprisal_gpt2"].copy()

    # Scale predictors (z-score standardization)
    df["log_lex_freq"] = stats.zscore(df["zipf_freq"])
    df["word_length"] = stats.zscore(df["word_length_with_punct"])
    df["surprisal"] = stats.zscore(df["surprisal_gpt2"])
    df["last_in_line"] = df["last_in_line"].astype(int)

    # Rename surprisal columns with hyphens to use underscores (for formula compatibility)
    for orig_col, new_col in SURPRISAL_RENAME_MAP.items():
        if orig_col in df.columns and orig_col != new_col:
            df[new_col] = df[orig_col]
            print(f"  Renamed {orig_col} -> {new_col}")

    # Z-score all surprisal columns from different LLMs
    for col in SURPRISAL_COLS:
        if col in df.columns:
            # Store raw values
            df[f"{col}_raw"] = df[col].copy()
            # Z-score standardize
            df[col] = stats.zscore(df[col])
            # Store scaling parameters
            df.attrs[f"{col}_mean"] = df[f"{col}_raw"].mean()
            df.attrs[f"{col}_std"] = df[f"{col}_raw"].std()
        else:
            print(f"  Warning: {col} not found in data")

    # Store scaling parameters for back-transformation
    df.attrs["surprisal_mean"] = df["surprisal_gpt2_raw"].mean()
    df.attrs["surprisal_std"] = df["surprisal_gpt2_raw"].std()
    df.attrs["word_length_mean"] = df["word_length_with_punct"].mean()
    df.attrs["word_length_std"] = df["word_length_with_punct"].std()
    df.attrs["zipf_freq_mean"] = df["zipf_freq"].mean()
    df.attrs["zipf_freq_std"] = df["zipf_freq"].std()

    print("  Preprocessing complete.")
    return df


# =============================================================================
# PART 1: Baseline Analysis with Millisecond Effect Sizes
# =============================================================================


def run_baseline_model(
    response_var: str, data: pd.DataFrame, iterations: int = DEFAULT_ITERATIONS
):
    """
    Fit baseline lognormal model and return results.
    """
    print(f"\nFitting baseline model for {response_var}...")

    formula = f"{response_var} ~ (1 | subject_id) + word_length + log_lex_freq + surprisal + last_in_line"

    # Filter to non-zero values and log-transform
    data_subset = data[data[response_var] > 0].copy()
    print(f"  Using {len(data_subset):,} non-zero observations")

    # Store the median RT for back-transformation
    median_rt = data_subset[response_var].median()
    mean_rt = data_subset[response_var].mean()

    data_subset[response_var] = np.log(data_subset[response_var])

    priors = {
        "Intercept": bmb.Prior("Normal", mu=6, sigma=1.5),
        "word_length": bmb.Prior("Normal", mu=0, sigma=1),
        "log_lex_freq": bmb.Prior("Normal", mu=0, sigma=1),
        "surprisal": bmb.Prior("Normal", mu=0, sigma=1),
        "last_in_line": bmb.Prior("Normal", mu=0, sigma=1),
        "sigma": bmb.Prior("HalfNormal", sigma=1),
    }

    model = bmb.Model(formula, data_subset, family="gaussian", priors=priors)

    results = model.fit(
        draws=max(iterations - WARMUP_ITERATIONS, 1000),
        tune=min(WARMUP_ITERATIONS, iterations // 2),
        chains=N_CHAINS,
        random_seed=RANDOM_SEED,
        progressbar=True,
    )

    return model, results, median_rt, mean_rt


def compute_ms_effects(results, median_rt: float, data: pd.DataFrame):
    """
    Convert log-scale coefficients to millisecond effect sizes.

    For a lognormal model, a coefficient β represents the multiplicative
    effect on the median: exp(β).

    The effect in ms for a 1-SD change is:
        effect_ms = median_rt * (exp(β) - 1)

    For more intuitive interpretation, we can also compute the effect
    for a 1-unit change in the original scale.
    """
    effects = {}

    predictors = ["word_length", "log_lex_freq", "surprisal", "last_in_line"]

    for pred in predictors:
        # Get posterior samples for the coefficient
        samples = results.posterior[pred].values.flatten()

        # Convert to multiplicative effect
        mult_effect = np.exp(samples)

        # Convert to ms effect (for 1-SD change in predictor)
        ms_effect_samples = median_rt * (mult_effect - 1)

        # Compute summary statistics
        effects[pred] = {
            "mean_ms": np.mean(ms_effect_samples),
            "sd_ms": np.std(ms_effect_samples),
            "hdi_2.5%_ms": np.percentile(ms_effect_samples, 2.5),
            "hdi_97.5%_ms": np.percentile(ms_effect_samples, 97.5),
            "mean_log": np.mean(samples),
            "hdi_2.5%_log": np.percentile(samples, 2.5),
            "hdi_97.5%_log": np.percentile(samples, 97.5),
            "samples_ms": ms_effect_samples,
        }

        # Compute effect for 1-unit change in original scale
        if pred == "word_length":
            # 1 character increase
            std = data.attrs.get("word_length_std", 1)
            unit_effect = median_rt * (np.exp(samples / std) - 1)
            effects[pred]["per_char_ms"] = np.mean(unit_effect)
        elif pred == "surprisal":
            # 1 bit increase in surprisal
            std = data.attrs.get("surprisal_std", 1)
            unit_effect = median_rt * (np.exp(samples / std) - 1)
            effects[pred]["per_bit_ms"] = np.mean(unit_effect)
        elif pred == "log_lex_freq":
            # 1 unit increase in Zipf frequency
            std = data.attrs.get("zipf_freq_std", 1)
            unit_effect = median_rt * (np.exp(samples / std) - 1)
            effects[pred]["per_zipf_ms"] = np.mean(unit_effect)

    return effects


def plot_baseline_ms_effects(
    response_var: str,
    effects: dict,
    output_dir: str = "plots",
    figsize: tuple = (10, 7),
):
    """
    Create coefficient plot with effect sizes in milliseconds.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Select predictors to plot (excluding last_in_line for cleaner visualization)
    predictors = ["word_length", "log_lex_freq", "surprisal"]
    labels = [
        "Word Length\n(+1 SD)",
        "Lexical Frequency\n(+1 SD)",
        "Surprisal\n(+1 SD)",
    ]

    x_pos = np.arange(len(predictors))

    means = [effects[p]["mean_ms"] for p in predictors]
    lowers = [effects[p]["hdi_2.5%_ms"] for p in predictors]
    uppers = [effects[p]["hdi_97.5%_ms"] for p in predictors]

    # Color by direction and significance
    colors = []
    for i, p in enumerate(predictors):
        if lowers[i] > 0:
            colors.append("#d62728")  # Red for increase
        elif uppers[i] < 0:
            colors.append("#2ca02c")  # Green for decrease (facilitation)
        else:
            colors.append("#7f7f7f")  # Gray for non-significant

    # Plot error bars
    for i, (x, mean, lower, upper, color) in enumerate(
        zip(x_pos, means, lowers, uppers, colors)
    ):
        ax.errorbar(
            x,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="o",
            markersize=12,
            capsize=8,
            capthick=2.5,
            elinewidth=2.5,
            color=color,
            markeredgecolor="black",
            markeredgewidth=1,
        )

    # Reference line at zero
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # Add value annotations
    for i, (x, mean, lower, upper) in enumerate(zip(x_pos, means, lowers, uppers)):
        offset = 5 if mean > 0 else -5
        va = "bottom" if mean > 0 else "top"
        ax.annotate(
            f"{mean:.1f} ms\n[{lower:.1f}, {upper:.1f}]",
            xy=(x, mean),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Effect Size (milliseconds)", fontsize=14)
    ax.set_xlabel("Predictor", fontsize=14)

    measure_label = MEASURE_LABELS.get(response_var, response_var)
    ax.set_title(
        f"Effect Sizes on {measure_label}\n(95% Credible Intervals)",
        fontsize=16,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, axis="y")

    # Add interpretation note
    note = "Positive = longer reading times; Negative = shorter reading times (facilitation)"
    ax.text(
        0.5,
        -0.12,
        note,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        style="italic",
        color="gray",
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"baseline_{response_var}_ms_effects.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    plt.show()
    return fig


# =============================================================================
# PART 2: Surprisal Prediction Model with Continuous Plot
# =============================================================================


def run_surprisal_model(
    response_var: str, data: pd.DataFrame, iterations: int = DEFAULT_ITERATIONS
):
    """
    Fit a model comparing surprisal from multiple LLMs.

    This model includes ONLY:
    - Surprisal predictors from each LLM (GPT-2, OPT, Mistral, Phi-2, LLaMA2, Pythia)
    - Random intercepts for subjects

    No control variables (word length, frequency) - pure surprisal comparison.
    """
    print(f"\nFitting multi-LLM surprisal model for {response_var}...")

    # Check which surprisal columns are available
    available_surp = [col for col in SURPRISAL_COLS if col in data.columns]
    print(f"  Available surprisal columns: {available_surp}")

    # Build formula with only surprisal predictors
    surp_terms = " + ".join(available_surp)
    formula = f"{response_var} ~ (1 | subject_id) + {surp_terms}"
    print(f"  Formula: {formula}")

    # Filter to non-zero values
    data_subset = data[data[response_var] > 0].copy()
    print(f"  Using {len(data_subset):,} non-zero observations")

    # Store original RT values for reference
    original_rt = data_subset[response_var].copy()
    median_rt = original_rt.median()

    # Log-transform response
    data_subset[response_var] = np.log(data_subset[response_var])

    # Build priors - only for surprisal predictors
    priors = {
        "Intercept": bmb.Prior("Normal", mu=6, sigma=1.5),
        "sigma": bmb.Prior("HalfNormal", sigma=1),
    }
    for col in available_surp:
        priors[col] = bmb.Prior("Normal", mu=0, sigma=1)

    model = bmb.Model(formula, data_subset, family="gaussian", priors=priors)

    results = model.fit(
        draws=max(iterations - WARMUP_ITERATIONS, 1000),
        tune=min(WARMUP_ITERATIONS, iterations // 2),
        chains=N_CHAINS,
        random_seed=RANDOM_SEED,
        progressbar=True,
    )

    return model, results, data_subset, median_rt, available_surp


def generate_surprisal_predictions(
    results, data: pd.DataFrame, available_surp: list, n_points: int = 100
):
    """
    Generate predictions for each LLM's surprisal separately.

    For each LLM, varies that surprisal while holding others at mean (0).
    """
    predictions = {}

    # Get posterior samples for intercept
    intercept = results.posterior["Intercept"].values.flatten()
    n_samples = len(intercept)

    # Get beta samples for each surprisal predictor
    betas = {}
    for col in available_surp:
        betas[col] = results.posterior[col].values.flatten()

    # Generate predictions for each LLM's surprisal
    for surp_col in available_surp:
        # Get surprisal range (in standardized units)
        surp_min = data[surp_col].min()
        surp_max = data[surp_col].max()
        surprisal_range = np.linspace(surp_min, surp_max, n_points)

        # Get original surprisal range for plotting
        raw_col = f"{surp_col}_raw"
        if raw_col in data.columns:
            surp_raw_min = data[raw_col].min()
            surp_raw_max = data[raw_col].max()
        else:
            # Approximate from standardized values
            surp_raw_min = surp_min
            surp_raw_max = surp_max
        surprisal_raw_range = np.linspace(surp_raw_min, surp_raw_max, n_points)

        # Generate predictions varying this surprisal, holding others at 0
        preds = np.zeros((n_samples, n_points))

        for i, surp in enumerate(surprisal_range):
            # Linear predictor: intercept + this surprisal's effect
            # Other surprisals held at 0 (their mean)
            log_rt = intercept + betas[surp_col] * surp
            # Back-transform to ms
            preds[:, i] = np.exp(log_rt)

        # Compute summary statistics
        predictions[surp_col] = {
            "surprisal_std": surprisal_range,
            "surprisal_raw": surprisal_raw_range,
            "mean": np.mean(preds, axis=0),
            "lower_95": np.percentile(preds, 2.5, axis=0),
            "upper_95": np.percentile(preds, 97.5, axis=0),
            "lower_50": np.percentile(preds, 25, axis=0),
            "upper_50": np.percentile(preds, 75, axis=0),
            "samples": preds,
        }

    return predictions


def compute_surprisal_ms_effects(results, median_rt: float, available_surp: list):
    """
    Compute effect sizes in milliseconds for each LLM's surprisal.
    """
    effects = {}

    for surp_col in available_surp:
        samples = results.posterior[surp_col].values.flatten()

        # Convert to multiplicative effect
        mult_effect = np.exp(samples)

        # Convert to ms effect (for 1-SD change in predictor)
        ms_effect_samples = median_rt * (mult_effect - 1)

        effects[surp_col] = {
            "mean_ms": np.mean(ms_effect_samples),
            "sd_ms": np.std(ms_effect_samples),
            "hdi_2.5%_ms": np.percentile(ms_effect_samples, 2.5),
            "hdi_97.5%_ms": np.percentile(ms_effect_samples, 97.5),
            "mean_log": np.mean(samples),
            "hdi_2.5%_log": np.percentile(samples, 2.5),
            "hdi_97.5%_log": np.percentile(samples, 97.5),
            "samples_ms": ms_effect_samples,
        }

    return effects


def plot_surprisal_comparison(
    response_var: str,
    effects: dict,
    output_dir: str = "plots",
    figsize: tuple = (12, 7),
):
    """
    Create a coefficient comparison plot for all LLM surprisals.

    Shows effect sizes in milliseconds with 95% CIs for each LLM.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get LLMs in order
    llms = list(effects.keys())
    n_llms = len(llms)

    x_pos = np.arange(n_llms)

    means = [effects[llm]["mean_ms"] for llm in llms]
    lowers = [effects[llm]["hdi_2.5%_ms"] for llm in llms]
    uppers = [effects[llm]["hdi_97.5%_ms"] for llm in llms]

    # Color palette for different LLMs
    colors = plt.cm.Set2(np.linspace(0, 1, n_llms))

    # Plot error bars
    for i, (x, mean, lower, upper, color) in enumerate(
        zip(x_pos, means, lowers, uppers, colors)
    ):
        ax.errorbar(
            x,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="o",
            markersize=14,
            capsize=10,
            capthick=2.5,
            elinewidth=2.5,
            color=color,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )

    # Reference line at zero
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # Add value annotations
    for i, (x, mean, lower, upper) in enumerate(zip(x_pos, means, lowers, uppers)):
        offset = 8 if mean > 0 else -8
        va = "bottom" if mean > 0 else "top"
        ax.annotate(
            f"{mean:.1f}",
            xy=(x, mean),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
        )

    # Labels
    labels = [SURPRISAL_LABELS.get(llm, llm) for llm in llms]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12, rotation=15, ha="right")
    ax.set_ylabel("Effect Size (milliseconds per SD)", fontsize=14)
    ax.set_xlabel("Language Model", fontsize=14)

    measure_label = MEASURE_LABELS.get(response_var, response_var)
    ax.set_title(
        f"Surprisal Effects on {measure_label} by LLM\n(95% Credible Intervals)",
        fontsize=16,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"surprisal_comparison_{response_var}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {plot_path}")

    plt.show()
    return fig


def plot_surprisal_effect(
    response_var: str,
    predictions: dict,
    data: pd.DataFrame,
    output_dir: str = "plots",
    figsize: tuple = (14, 10),
    show_data: bool = False,
):
    """
    Create prediction plots showing each LLM's surprisal effect on reading time.

    One subplot per LLM, showing predicted RT across surprisal range.
    """
    n_llms = len(predictions)
    n_cols = 3
    n_rows = (n_llms + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_llms > 1 else [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_llms))

    for idx, (surp_col, pred) in enumerate(predictions.items()):
        ax = axes[idx]

        surprisal = pred["surprisal_raw"]
        mean_rt = pred["mean"]
        lower_95 = pred["lower_95"]
        upper_95 = pred["upper_95"]

        # Plot 95% credible interval
        ax.fill_between(surprisal, lower_95, upper_95, alpha=0.3, color=colors[idx])

        # Plot mean prediction
        ax.plot(surprisal, mean_rt, color=colors[idx], linewidth=2.5)

        # Labels
        llm_label = SURPRISAL_LABELS.get(surp_col, surp_col)
        ax.set_title(llm_label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Surprisal (bits)", fontsize=11)
        ax.set_ylabel("RT (ms)", fontsize=11)

        # Add effect annotation
        rt_change = mean_rt[-1] - mean_rt[0]
        ax.text(
            0.05,
            0.95,
            f"Δ = {rt_change:+.0f} ms",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_llms, len(axes)):
        axes[idx].set_visible(False)

    measure_label = MEASURE_LABELS.get(response_var, response_var)
    fig.suptitle(
        f"Predicted {measure_label} by Surprisal (each LLM)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"surprisal_effects_all_llms_{response_var}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Multi-LLM plot saved to {plot_path}")

    plt.show()
    return fig


def plot_surprisal_overlay(
    response_var: str,
    predictions: dict,
    output_dir: str = "plots",
    figsize: tuple = (12, 8),
):
    """
    Create an overlay plot with all LLM surprisal predictions on one axis.
    Clean publication-ready version.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))

    for idx, (surp_col, pred) in enumerate(predictions.items()):
        surprisal = pred["surprisal_raw"]
        mean_rt = pred["mean"]
        lower_95 = pred["lower_95"]
        upper_95 = pred["upper_95"]

        llm_label = SURPRISAL_LABELS.get(surp_col, surp_col)

        # Plot mean line
        ax.plot(surprisal, mean_rt, color=colors[idx], linewidth=2.5, label=llm_label)

        # Plot CI as thin shading
        ax.fill_between(surprisal, lower_95, upper_95, alpha=0.1, color=colors[idx])

    # Labels and title
    ax.set_xlabel("Surprisal (bits)", fontsize=14, fontweight="bold")
    measure_label = MEASURE_LABELS.get(response_var, response_var)
    ax.set_ylabel(f"{measure_label} (ms)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Predicted {measure_label} by Surprisal\n(Comparison Across LLMs)",
        fontsize=16,
        fontweight="bold",
    )

    # Legend
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9, ncol=2)

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"surprisal_overlay_{response_var}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Overlay plot saved to {plot_path}")

    pdf_path = output_path / f"surprisal_overlay_{response_var}.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.show()
    return fig


# =============================================================================
# Main Function
# =============================================================================


def save_results(
    model,
    results,
    analysis_type: str,
    response_var: str,
    output_dir: str = "model_fits",
    **kwargs,
):
    """Save model results and additional data."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save posterior samples
    nc_path = output_path / f"{analysis_type}_{response_var}.nc"
    results.to_netcdf(str(nc_path))
    print(f"\nPosterior samples saved to {nc_path}")

    # Save summary
    summary = az.summary(results, hdi_prob=0.95)
    csv_path = output_path / f"{analysis_type}_{response_var}_summary.csv"
    summary.to_csv(csv_path)
    print(f"Summary saved to {csv_path}")

    # Save additional data if provided
    if "effects" in kwargs:
        effects_df = pd.DataFrame(kwargs["effects"]).T
        effects_path = output_path / f"{analysis_type}_{response_var}_ms_effects.csv"
        effects_df.to_csv(effects_path)
        print(f"MS effects saved to {effects_path}")

    if "predictions" in kwargs:
        pred_df = pd.DataFrame(
            {
                "surprisal_bits": kwargs["predictions"]["surprisal_raw"],
                "predicted_rt_ms": kwargs["predictions"]["mean"],
                "ci_lower_95": kwargs["predictions"]["lower_95"],
                "ci_upper_95": kwargs["predictions"]["upper_95"],
            }
        )
        pred_path = output_path / f"{analysis_type}_{response_var}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to {pred_path}")


def main(
    response="FPRT",
    analysis="baseline",
    iterations=3000,  # Replaced DEFAULT_ITERATIONS with explicit value or keep variable if defined globally
    data_path="/kaggle/input/emtec-ds/reading_measures_corrected.csv",
    output_dir="model_fits",
    plot_dir="plots",
    n_subjects=None,
):
    """
    Main function refactored for notebook execution.
    Parameters correspond to the original command line arguments.
    """

    # Input validation (replicating argparse choices)
    if response not in ["FPRT", "TFT", "FFD", "FRT"]:
        raise ValueError(f"Invalid response: {response}")
    if analysis not in ["baseline", "surprisal", "both"]:
        raise ValueError(f"Invalid analysis: {analysis}")

    print("=" * 70)
    print("Extended Bayesian Analysis for Eye-Tracking Reading Measures")
    print("=" * 70)
    print(f"\nResponse variable: {response}")
    print(f"Analysis type: {analysis}")
    print(f"Iterations: {iterations}")

    # Load data
    data = load_and_preprocess_data(data_path)

    # Optional subsampling
    if n_subjects is not None:
        subjects = data["subject_id"].unique()[:n_subjects]
        data = data[data["subject_id"].isin(subjects)].copy()
        print(f"\nSubsampled to {n_subjects} subjects: {len(data):,} observations")

    # Run analyses
    if analysis in ["baseline", "both"]:
        print("\n" + "=" * 70)
        print("PART 1: Baseline Analysis with Millisecond Effect Sizes")
        print("=" * 70)

        model, results, median_rt, mean_rt = run_baseline_model(
            response, data, iterations
        )

        # Compute ms effects
        effects = compute_ms_effects(results, median_rt, data)

        # Print effects
        print(f"\nEffect sizes in milliseconds (median RT = {median_rt:.0f} ms):")
        print("-" * 60)
        for pred, vals in effects.items():
            print(
                f"{pred:15s}: {vals['mean_ms']:+7.1f} ms  "
                f"[{vals['hdi_2.5%_ms']:+7.1f}, {vals['hdi_97.5%_ms']:+7.1f}]"
            )

        # Create plot
        plot_baseline_ms_effects(response, effects, plot_dir)

        # Save results
        save_results(
            model,
            results,
            "baseline_ms",
            response,
            output_dir,
            effects=effects,
        )

    if analysis in ["surprisal", "both"]:
        print("\n" + "=" * 70)
        print("PART 2: Multi-LLM Surprisal Comparison")
        print("=" * 70)

        model, results, data_subset, median_rt, available_surp = run_surprisal_model(
            response, data, iterations
        )

        # Compute ms effects for each LLM
        surp_effects = compute_surprisal_ms_effects(results, median_rt, available_surp)

        # Print effects
        print(f"\nSurprisal effects in milliseconds (median RT = {median_rt:.0f} ms):")
        print("-" * 70)
        for surp_col, vals in surp_effects.items():
            llm_label = SURPRISAL_LABELS.get(surp_col, surp_col)
            print(
                f"{llm_label:15s}: {vals['mean_ms']:+7.1f} ms  "
                f"[{vals['hdi_2.5%_ms']:+7.1f}, {vals['hdi_97.5%_ms']:+7.1f}]"
            )

        # Generate predictions for each LLM
        predictions = generate_surprisal_predictions(
            results, data_subset, available_surp
        )

        # Create plots
        plot_surprisal_comparison(response, surp_effects, plot_dir)
        plot_surprisal_effect(response, predictions, data_subset, plot_dir)
        plot_surprisal_overlay(response, predictions, plot_dir)

        # Save results
        save_results(
            model,
            results,
            "surprisal_multi_llm",
            response,
            output_dir,
            effects=surp_effects,
        )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
