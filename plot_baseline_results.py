#!/usr/bin/env python
"""
Plot Bayesian Regression Coefficients from Baseline Analyses
=============================================================

This script creates visualizations of the coefficient estimates from the
baseline psycholinguistic analyses, replicating the plots from the EMTeC paper.

Available plot types:
- coefficients: Point estimates with credible intervals (default)
- posterior: Full posterior distributions for each coefficient
- trace: MCMC trace plots for convergence diagnostics
- forest: Compare effects across multiple models
- random_effects: Subject-level variability
- summary: Combined dashboard with multiple plots

Usage:
    python plot_baseline_results.py -r FPRT
    python plot_baseline_results.py -r FPRT --plot-type posterior
    python plot_baseline_results.py -r FPRT --plot-type trace
    python plot_baseline_results.py -r FPRT --plot-type random_effects
    python plot_baseline_results.py --all --plot-type forest
    python plot_baseline_results.py -r FPRT --plot-type summary
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Optional: arviz for loading NetCDF files with full posteriors
try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def load_summary(response_var: str, model_dir: str = "model_fits") -> pd.DataFrame:
    """
    Load the summary CSV for a given response variable.

    Parameters
    ----------
    response_var : str
        The response variable name (e.g., 'FPRT')
    model_dir : str
        Directory containing model results

    Returns
    -------
    pd.DataFrame
        Summary dataframe with coefficient estimates
    """
    summary_path = Path(model_dir) / f"baseline_{response_var}_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Make sure you've run the analysis for {response_var} first."
        )

    df = pd.read_csv(summary_path, index_col=0)
    return df


def extract_fixed_effects(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract fixed effects (excluding random effects and sigma).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Full summary dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with only fixed effects
    """
    # Fixed effects of interest
    fixed_effects = [
        "Intercept",
        "word_length",
        "log_lex_freq",
        "surprisal",
        "last_in_line",
    ]

    # Filter to only fixed effects that exist in the summary
    available_effects = [fe for fe in fixed_effects if fe in summary_df.index]

    return summary_df.loc[available_effects]


def plot_coefficients(
    response_var: str,
    model_dir: str = "model_fits",
    output_dir: str = "plots",
    show_intercept: bool = False,
    show_last_in_line: bool = False,
    figsize: tuple = (8, 6),
):
    """
    Create a coefficient plot with credible intervals.

    Parameters
    ----------
    response_var : str
        Response variable name
    model_dir : str
        Directory containing model results
    output_dir : str
        Directory to save plots
    show_intercept : bool
        Whether to include intercept in plot
    show_last_in_line : bool
        Whether to include last_in_line in plot
    figsize : tuple
        Figure size (width, height)
    """
    # Load summary
    summary_df = load_summary(response_var, model_dir)
    fixed_effects = extract_fixed_effects(summary_df)

    # Filter effects to plot
    effects_to_plot = []
    labels = []

    if show_intercept and "Intercept" in fixed_effects.index:
        effects_to_plot.append("Intercept")
        labels.append("Intercept")

    if "word_length" in fixed_effects.index:
        effects_to_plot.append("word_length")
        labels.append("Word length")

    if "log_lex_freq" in fixed_effects.index:
        effects_to_plot.append("log_lex_freq")
        labels.append("Lexical frequency")

    if "surprisal" in fixed_effects.index:
        effects_to_plot.append("surprisal")
        labels.append("Surprisal")

    if show_last_in_line and "last_in_line" in fixed_effects.index:
        effects_to_plot.append("last_in_line")
        labels.append("Last in line")

    # Extract estimates and credible intervals
    estimates = fixed_effects.loc[effects_to_plot, "mean"].values
    lower = fixed_effects.loc[effects_to_plot, "hdi_2.5%"].values
    upper = fixed_effects.loc[effects_to_plot, "hdi_97.5%"].values

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points and error bars
    x_pos = np.arange(len(labels))

    # Color points based on whether CI includes zero
    colors = []
    for i in range(len(estimates)):
        if lower[i] > 0 or upper[i] < 0:
            colors.append("#1f77b4")  # Significant - blue
        else:
            colors.append("#d62728")  # Not significant - red

    ax.errorbar(
        x_pos,
        estimates,
        yerr=[estimates - lower, upper - estimates],
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
        elinewidth=2,
        color=colors[0] if len(set(colors)) == 1 else None,
        ecolor=colors[0] if len(set(colors)) == 1 else None,
    )

    # Color individual points if they differ
    if len(set(colors)) > 1:
        for i, (x, y, c) in enumerate(zip(x_pos, estimates, colors)):
            ax.plot(x, y, "o", markersize=8, color=c)

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Coefficient estimate", fontsize=12)
    ax.set_title(f"Fixed effects for {response_var}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add text box with model info
    n_obs = f"Model: {response_var}"
    ax.text(
        0.02,
        0.98,
        n_obs,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"baseline_{response_var}_coefficients.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # Also save as PDF
    pdf_path = output_path / f"baseline_{response_var}_coefficients.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plot saved to {pdf_path}")

    plt.show()

    return fig, ax


def plot_multiple_models(
    response_vars: List[str],
    model_dir: str = "model_fits",
    output_dir: str = "plots",
    ncols: int = 2,
    figsize: tuple = (12, 10),
):
    """
    Create a grid of coefficient plots for multiple models.

    Parameters
    ----------
    response_vars : list
        List of response variable names
    model_dir : str
        Directory containing model results
    output_dir : str
        Directory to save plots
    ncols : int
        Number of columns in the grid
    figsize : tuple
        Figure size (width, height)
    """
    n_models = len(response_vars)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, response_var in enumerate(response_vars):
        ax = axes[idx]

        try:
            # Load summary
            summary_df = load_summary(response_var, model_dir)
            fixed_effects = extract_fixed_effects(summary_df)

            # Define effects to plot
            effects_to_plot = ["word_length", "log_lex_freq", "surprisal"]
            labels = ["Word length", "Lexical frequency", "Surprisal"]

            # Filter to available effects
            available = [e for e in effects_to_plot if e in fixed_effects.index]
            labels = [
                labels[i] for i, e in enumerate(effects_to_plot) if e in available
            ]

            # Extract estimates and credible intervals
            estimates = fixed_effects.loc[available, "mean"].values
            lower = fixed_effects.loc[available, "hdi_2.5%"].values
            upper = fixed_effects.loc[available, "hdi_97.5%"].values

            # Plot
            x_pos = np.arange(len(labels))

            # Color based on significance
            colors = [
                "#1f77b4" if (l > 0 or u < 0) else "#d62728"
                for l, u in zip(lower, upper)
            ]

            ax.errorbar(
                x_pos,
                estimates,
                yerr=[estimates - lower, upper - estimates],
                fmt="o",
                markersize=6,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
            )

            # Color points individually
            for i, (x, y, c) in enumerate(zip(x_pos, estimates, colors)):
                ax.plot(x, y, "o", markersize=6, color=c)

            ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
            ax.set_title(response_var, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        except FileNotFoundError:
            ax.text(
                0.5,
                0.5,
                f"{response_var}\nNo results found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Add common y-label
    fig.text(
        0.04, 0.5, "Coefficient estimate", va="center", rotation="vertical", fontsize=12
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / "baseline_all_coefficients.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to {plot_path}")

    pdf_path = output_path / "baseline_all_coefficients.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Combined plot saved to {pdf_path}")

    plt.show()

    return fig, axes


def create_summary_table(
    response_vars: List[str], model_dir: str = "model_fits", output_dir: str = "plots"
):
    """
    Create a summary table of all model coefficients.

    Parameters
    ----------
    response_vars : list
        List of response variable names
    model_dir : str
        Directory containing model results
    output_dir : str
        Directory to save table
    """
    results = []

    for response_var in response_vars:
        try:
            summary_df = load_summary(response_var, model_dir)
            fixed_effects = extract_fixed_effects(summary_df)

            for effect in ["word_length", "log_lex_freq", "surprisal", "last_in_line"]:
                if effect in fixed_effects.index:
                    row = fixed_effects.loc[effect]
                    results.append(
                        {
                            "Model": response_var,
                            "Predictor": effect,
                            "Estimate": row["mean"],
                            "SD": row["sd"],
                            "CI_Lower": row["hdi_2.5%"],
                            "CI_Upper": row["hdi_97.5%"],
                            "Significant": (
                                "Yes"
                                if (row["hdi_2.5%"] > 0 or row["hdi_97.5%"] < 0)
                                else "No"
                            ),
                        }
                    )
        except FileNotFoundError:
            continue

    table_df = pd.DataFrame(results)

    # Save table
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    table_path = output_path / "baseline_coefficients_table.csv"
    table_df.to_csv(table_path, index=False)
    print(f"\nSummary table saved to {table_path}")

    # Print formatted table
    print("\nCoefficient Summary Table:")
    print("=" * 90)
    print(table_df.to_string(index=False))

    return table_df


# =============================================================================
# Additional Plot Types
# =============================================================================


def load_posterior(response_var: str, model_dir: str = "model_fits"):
    """
    Load the full posterior samples from NetCDF file.

    Requires arviz to be installed.
    """
    if not HAS_ARVIZ:
        raise ImportError(
            "arviz is required to load posterior samples. Install with: pip install arviz"
        )

    nc_path = Path(model_dir) / f"baseline_{response_var}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    return az.from_netcdf(str(nc_path))


def plot_posterior_distributions(
    response_var: str,
    model_dir: str = "model_fits",
    output_dir: str = "plots",
    figsize: tuple = (12, 8),
):
    """
    Plot the full posterior distributions for each coefficient.

    Shows more information than just point estimates - you can see
    the full uncertainty and shape of the posterior.
    """
    print(f"\nCreating posterior distribution plot for {response_var}...")

    idata = load_posterior(response_var, model_dir)

    # Variables to plot
    var_names = ["word_length", "log_lex_freq", "surprisal", "last_in_line"]
    labels = {
        "word_length": "Word Length",
        "log_lex_freq": "Lexical Frequency",
        "surprisal": "Surprisal (GPT-2)",
        "last_in_line": "Last in Line",
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for idx, var in enumerate(var_names):
        ax = axes[idx]

        # Get posterior samples
        samples = idata.posterior[var].values.flatten()

        # Plot density
        sns.kdeplot(samples, ax=ax, fill=True, alpha=0.5, color=colors[idx])

        # Add vertical line at zero
        ax.axvline(
            x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Zero"
        )

        # Add mean and HDI
        mean_val = np.mean(samples)
        hdi = az.hdi(samples, hdi_prob=0.95)

        ax.axvline(
            x=mean_val,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {mean_val:.4f}",
        )
        ax.axvspan(
            hdi[0],
            hdi[1],
            alpha=0.2,
            color="gray",
            label=f"95% HDI: [{hdi[0]:.4f}, {hdi[1]:.4f}]",
        )

        ax.set_xlabel("Coefficient Value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(labels[var], fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Posterior Distributions - {response_var}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"baseline_{response_var}_posterior.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    plt.show()
    return fig


def plot_trace(
    response_var: str,
    model_dir: str = "model_fits",
    output_dir: str = "plots",
    figsize: tuple = (14, 10),
):
    """
    Plot MCMC trace plots for convergence diagnostics.

    Good traces should look like "fuzzy caterpillars" - well-mixed
    chains that explore the same region of parameter space.
    """
    print(f"\nCreating trace plot for {response_var}...")

    idata = load_posterior(response_var, model_dir)

    var_names = [
        "Intercept",
        "word_length",
        "log_lex_freq",
        "surprisal",
        "last_in_line",
        "sigma",
    ]

    # Use arviz plot_trace
    axes = az.plot_trace(idata, var_names=var_names, figsize=figsize, compact=True)

    plt.suptitle(
        f"Trace Plots - {response_var}", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"baseline_{response_var}_trace.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    plt.show()
    return axes


def plot_random_effects(
    response_var: str,
    model_dir: str = "model_fits",
    output_dir: str = "plots",
    figsize: tuple = (10, 8),
):
    """
    Plot the random effects (subject-level deviations from population mean).

    This shows individual differences in reading behavior across participants.
    """
    print(f"\nCreating random effects plot for {response_var}...")

    # Load summary to get random effects
    summary_df = load_summary(response_var, model_dir)

    # Filter to subject random effects
    re_rows = summary_df.index[summary_df.index.str.startswith("1|subject_id[")]

    if len(re_rows) == 0:
        print("No random effects found in the model summary.")
        return None

    # Extract subject IDs and values
    subjects = [idx.replace("1|subject_id[", "").replace("]", "") for idx in re_rows]
    means = summary_df.loc[re_rows, "mean"].values
    lower = summary_df.loc[re_rows, "hdi_2.5%"].values
    upper = summary_df.loc[re_rows, "hdi_97.5%"].values

    # Sort by mean
    sort_idx = np.argsort(means)
    subjects = [subjects[i] for i in sort_idx]
    means = means[sort_idx]
    lower = lower[sort_idx]
    upper = upper[sort_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(subjects))

    # Color by whether CI includes zero
    colors = [
        "#1f77b4" if (l > 0 or u < 0) else "#7f7f7f" for l, u in zip(lower, upper)
    ]

    ax.errorbar(
        means,
        y_pos,
        xerr=[means - lower, upper - means],
        fmt="o",
        markersize=6,
        capsize=3,
        capthick=1.5,
        elinewidth=1.5,
        ecolor="gray",
    )

    # Color points
    for i, (x, y, c) in enumerate(zip(means, y_pos, colors)):
        ax.plot(x, y, "o", markersize=6, color=c)

    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subjects, fontsize=9)
    ax.set_xlabel("Random Effect (deviation from population mean)", fontsize=11)
    ax.set_ylabel("Subject", fontsize=11)
    ax.set_title(
        f"Subject Random Effects - {response_var}", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add sigma info
    if "1|subject_id_sigma" in summary_df.index:
        sigma = summary_df.loc["1|subject_id_sigma", "mean"]
        ax.text(
            0.98,
            0.02,
            f"σ_subject = {sigma:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"baseline_{response_var}_random_effects.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    plt.show()
    return fig


def plot_probability_of_direction(
    response_var: str,
    model_dir: str = "model_fits",
    output_dir: str = "plots",
    figsize: tuple = (8, 6),
):
    """
    Plot the probability of direction (pd) for each coefficient.

    Shows the probability that each effect is positive or negative.
    A pd of 0.95 means 95% probability the effect is in that direction.
    """
    print(f"\nCreating probability of direction plot for {response_var}...")

    idata = load_posterior(response_var, model_dir)

    var_names = ["word_length", "log_lex_freq", "surprisal", "last_in_line"]
    labels = ["Word Length", "Lexical Frequency", "Surprisal", "Last in Line"]

    # Calculate probability of direction
    pd_values = []
    directions = []

    for var in var_names:
        samples = idata.posterior[var].values.flatten()
        prop_positive = np.mean(samples > 0)
        prop_negative = np.mean(samples < 0)

        if prop_positive > prop_negative:
            pd_values.append(prop_positive)
            directions.append("positive")
        else:
            pd_values.append(prop_negative)
            directions.append("negative")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(labels))
    colors = ["#2ca02c" if d == "positive" else "#d62728" for d in directions]

    bars = ax.bar(x_pos, pd_values, color=colors, alpha=0.7, edgecolor="black")

    # Add reference lines
    ax.axhline(y=0.95, color="gray", linestyle="--", linewidth=1, label="95% threshold")
    ax.axhline(y=0.99, color="gray", linestyle=":", linewidth=1, label="99% threshold")

    # Add value labels
    for i, (bar, pd, d) in enumerate(zip(bars, pd_values, directions)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{pd:.1%}\n({d[0]})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Probability of Direction", fontsize=12)
    ax.set_ylim(0.5, 1.05)
    ax.set_title(
        f"Probability of Direction - {response_var}", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"baseline_{response_var}_pd.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    plt.show()
    return fig


def plot_summary_dashboard(
    response_var: str,
    model_dir: str = "model_fits",
    output_dir: str = "plots",
    figsize: tuple = (16, 12),
):
    """
    Create a comprehensive dashboard with multiple plot types.
    """
    print(f"\nCreating summary dashboard for {response_var}...")

    fig = plt.figure(figsize=figsize)

    # Load data
    summary_df = load_summary(response_var, model_dir)
    fixed_effects = extract_fixed_effects(summary_df)

    has_posterior = False
    if HAS_ARVIZ:
        try:
            idata = load_posterior(response_var, model_dir)
            has_posterior = True
        except FileNotFoundError:
            pass

    # Layout: 2x3 grid
    # Top row: coefficients, random effects, probability of direction
    # Bottom row: posterior distributions (if available)

    # 1. Coefficient plot (top left)
    ax1 = fig.add_subplot(2, 3, 1)

    effects = ["word_length", "log_lex_freq", "surprisal"]
    labels = ["Word\nlength", "Lexical\nfrequency", "Surprisal"]
    available = [e for e in effects if e in fixed_effects.index]
    labels = [labels[i] for i, e in enumerate(effects) if e in available]

    estimates = fixed_effects.loc[available, "mean"].values
    lower = fixed_effects.loc[available, "hdi_2.5%"].values
    upper = fixed_effects.loc[available, "hdi_97.5%"].values

    x_pos = np.arange(len(labels))
    colors = [
        "#1f77b4" if (l > 0 or u < 0) else "#d62728" for l, u in zip(lower, upper)
    ]

    ax1.errorbar(
        x_pos,
        estimates,
        yerr=[estimates - lower, upper - estimates],
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
        elinewidth=2,
    )
    for i, (x, y, c) in enumerate(zip(x_pos, estimates, colors)):
        ax1.plot(x, y, "o", markersize=8, color=c)

    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Coefficient", fontsize=11)
    ax1.set_title("Fixed Effects", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 2. Random effects (top middle)
    ax2 = fig.add_subplot(2, 3, 2)

    re_rows = summary_df.index[summary_df.index.str.startswith("1|subject_id[")]
    if len(re_rows) > 0:
        subjects = [idx.split("[")[1].replace("]", "") for idx in re_rows]
        re_means = summary_df.loc[re_rows, "mean"].values
        sort_idx = np.argsort(re_means)

        y_pos = np.arange(len(subjects))
        ax2.barh(y_pos, re_means[sort_idx], color="steelblue", alpha=0.7)
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=1)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([subjects[i] for i in sort_idx], fontsize=8)
        ax2.set_xlabel("Random Effect", fontsize=11)
        ax2.set_title("Subject Variability", fontsize=12, fontweight="bold")

    # 3. Model info (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis("off")

    info_text = f"Model: {response_var}\n\n"
    info_text += "Fixed Effects:\n"
    for var in ["word_length", "log_lex_freq", "surprisal", "last_in_line"]:
        if var in fixed_effects.index:
            row = fixed_effects.loc[var]
            sig = "✓" if (row["hdi_2.5%"] > 0 or row["hdi_97.5%"] < 0) else "✗"
            info_text += f"  {var}: {row['mean']:.4f} {sig}\n"

    if "1|subject_id_sigma" in summary_df.index:
        sigma = summary_df.loc["1|subject_id_sigma", "mean"]
        info_text += f"\nSubject σ: {sigma:.4f}"

    if "sigma" in summary_df.index:
        sigma = summary_df.loc["sigma", "mean"]
        info_text += f"\nResidual σ: {sigma:.4f}"

    ax3.text(
        0.1,
        0.9,
        info_text,
        transform=ax3.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax3.set_title("Model Summary", fontsize=12, fontweight="bold")

    # Bottom row: posteriors (if available)
    if has_posterior:
        var_names = ["word_length", "log_lex_freq", "surprisal"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for idx, (var, color) in enumerate(zip(var_names, colors)):
            ax = fig.add_subplot(2, 3, 4 + idx)
            samples = idata.posterior[var].values.flatten()

            sns.kdeplot(samples, ax=ax, fill=True, alpha=0.5, color=color)
            ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.axvline(x=np.mean(samples), color="black", linestyle="-", linewidth=2)

            ax.set_xlabel("Coefficient", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(var.replace("_", " ").title(), fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Baseline Analysis Summary - {response_var}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_path = output_path / f"baseline_{response_var}_dashboard.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard saved to {plot_path}")

    plt.show()
    return fig


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Plot coefficient estimates from baseline analyses."
    )
    parser.add_argument(
        "-r",
        "--response",
        type=str,
        default="FPRT",
        help="Response variable to plot (e.g., FPRT, TFT, Fix, FPReg)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model_fits",
        help="Directory containing model results",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots", help="Directory to save plots"
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="coefficients",
        choices=[
            "coefficients",
            "posterior",
            "trace",
            "random_effects",
            "pd",
            "summary",
            "forest",
        ],
        help="Type of plot to create",
    )
    parser.add_argument(
        "--all", action="store_true", help="Plot all available models (for forest plot)"
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Create a summary table of all coefficients",
    )
    parser.add_argument(
        "--show-intercept", action="store_true", help="Include intercept in the plot"
    )
    parser.add_argument(
        "--show-last-in-line",
        action="store_true",
        help="Include last_in_line effect in the plot",
    )

    args = parser.parse_args()

    # Check what model files are available
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"Error: Model directory '{args.model_dir}' not found.")
        return

    available_models = []
    for summary_file in model_path.glob("baseline_*_summary.csv"):
        response_var = summary_file.stem.replace("baseline_", "").replace(
            "_summary", ""
        )
        available_models.append(response_var)

    print(f"Available models: {', '.join(available_models)}")

    # Handle different plot types
    if args.plot_type == "forest" or (args.all and available_models):
        print("\nCreating forest plot for all models...")
        plot_multiple_models(
            available_models, model_dir=args.model_dir, output_dir=args.output_dir
        )
    elif args.plot_type == "coefficients":
        print(f"\nCreating coefficient plot for {args.response}...")
        plot_coefficients(
            args.response,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            show_intercept=args.show_intercept,
            show_last_in_line=args.show_last_in_line,
        )
    elif args.plot_type == "posterior":
        plot_posterior_distributions(
            args.response, model_dir=args.model_dir, output_dir=args.output_dir
        )
    elif args.plot_type == "trace":
        plot_trace(args.response, model_dir=args.model_dir, output_dir=args.output_dir)
    elif args.plot_type == "random_effects":
        plot_random_effects(
            args.response, model_dir=args.model_dir, output_dir=args.output_dir
        )
    elif args.plot_type == "pd":
        plot_probability_of_direction(
            args.response, model_dir=args.model_dir, output_dir=args.output_dir
        )
    elif args.plot_type == "summary":
        plot_summary_dashboard(
            args.response, model_dir=args.model_dir, output_dir=args.output_dir
        )

    if args.table and available_models:
        create_summary_table(
            available_models, model_dir=args.model_dir, output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
