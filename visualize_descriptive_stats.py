"""
Visualize descriptive statistics from the EMTeC paper.
Creates plots for Tables 9, 10, 11 to illustrate key observations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Create output directory for plots
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Load the data
print("Loading reading measures data...")
df = pd.read_csv("data/reading_measures_corrected.csv", sep="\t")
print(f"Loaded {len(df)} rows")

# Reading measures
continuous_measures = ["FFD", "FPRT", "TFT", "RRT"]
binary_measures = ["Fix", "RR"]

measure_labels = {
    "FFD": "First-Fixation Duration",
    "FPRT": "First-Pass Reading Time",
    "TFT": "Total Fixation Time",
    "RRT": "Re-Reading Time",
    "Fix": "Fixation Proportion",
    "RR": "Re-Reading Proportion",
}


def compute_stats(data, measure):
    """Compute mean and std for measures, excluding zeros for continuous."""
    values = data[measure]
    if measure in continuous_measures:
        non_zero = values[values > 0]
        return non_zero.mean(), non_zero.std(), non_zero.sem()
    else:
        return values.mean(), values.std(), values.sem()


# ============================================================================
# Prepare data for plotting
# ============================================================================

# Model statistics
model_stats = []
for model in ["phi2", "mistral", "wizardlm"]:
    model_data = df[df["model"] == model]
    for measure in continuous_measures + binary_measures:
        mean, std, sem = compute_stats(model_data, measure)
        model_stats.append(
            {
                "Model": model.capitalize() if model != "wizardlm" else "WizardLM",
                "Measure": measure,
                "Mean": mean,
                "Std": std,
                "SEM": sem,
            }
        )
model_df = pd.DataFrame(model_stats)

# Decoding strategy statistics
strategy_map = {
    "greedy_search": "Greedy",
    "beam_search": "Beam",
    "sampling": "Sampling",
    "topk": "Top-k",
    "topp": "Top-p",
}
strategy_stats = []
for strategy, label in strategy_map.items():
    if strategy in df["decoding_strategy"].values:
        strat_data = df[df["decoding_strategy"] == strategy]
        for measure in continuous_measures + binary_measures:
            mean, std, sem = compute_stats(strat_data, measure)
            strategy_stats.append(
                {
                    "Strategy": label,
                    "Measure": measure,
                    "Mean": mean,
                    "Std": std,
                    "SEM": sem,
                }
            )
strategy_df = pd.DataFrame(strategy_stats)

# Text type statistics
task_map = {
    "non-fiction": "Non-fiction",
    "fiction": "Fiction",
    "poetry": "Poetry",
    "summarization": "Summarization",
    "article_synopsis": "Synopsis",
    "words_given": "Key words",
}
task_stats = []
for task, label in task_map.items():
    if task in df["task"].values:
        task_data = df[df["task"] == task]
        task_type = (
            "Unconstrained"
            if task in ["non-fiction", "fiction", "poetry"]
            else "Constrained"
        )
        for measure in continuous_measures + binary_measures:
            mean, std, sem = compute_stats(task_data, measure)
            task_stats.append(
                {
                    "Task": label,
                    "Type": task_type,
                    "Measure": measure,
                    "Mean": mean,
                    "Std": std,
                    "SEM": sem,
                }
            )
task_df = pd.DataFrame(task_stats)

# ============================================================================
# FIGURE 1: Table 9 - Reading measures by Model
# ============================================================================
print("\nCreating Figure 1: Reading measures by Model...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle(
    "Table 9: Reading Measures by Model\n(Minimal variation between models)",
    fontsize=14,
    fontweight="bold",
)

colors = ["#4C72B0", "#55A868", "#C44E52"]
model_order = ["Phi2", "Mistral", "WizardLM"]

# Continuous measures (top row + first of bottom)
for idx, measure in enumerate(continuous_measures):
    ax = axes[idx // 3, idx % 3]
    data = model_df[model_df["Measure"] == measure]
    data = data.set_index("Model").loc[model_order].reset_index()

    bars = ax.bar(
        data["Model"],
        data["Mean"],
        yerr=data["Std"],
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        error_kw={"linewidth": 1.5},
    )

    ax.set_ylabel("Duration (ms)", fontsize=11)
    ax.set_title(measure_labels[measure], fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)

    # Add value labels on bars
    for bar, val in zip(bars, data["Mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + data["Std"].max() * 0.05,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# Binary measures (bottom row)
for idx, measure in enumerate(binary_measures):
    ax = axes[1, idx + 1]
    data = model_df[model_df["Measure"] == measure]
    data = data.set_index("Model").loc[model_order].reset_index()

    bars = ax.bar(
        data["Model"], data["Mean"], color=colors, edgecolor="black", linewidth=1.2
    )

    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title(measure_labels[measure], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)

    for bar, val in zip(bars, data["Mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# Remove empty subplot
axes[1, 0].axis("off")

plt.tight_layout()
plt.savefig(output_dir / "fig1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "fig1_model_comparison.pdf", bbox_inches="tight")
print(f"  Saved to {output_dir}/fig1_model_comparison.png")

# ============================================================================
# FIGURE 2: Table 10 - Reading measures by Decoding Strategy
# ============================================================================
print("\nCreating Figure 2: Reading measures by Decoding Strategy...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    "Table 10: Reading Measures by Decoding Strategy\n(Beam search → highest TFT/RRT; Greedy → lowest)",
    fontsize=14,
    fontweight="bold",
)

strategy_order = ["Greedy", "Beam", "Sampling", "Top-k", "Top-p"]
colors_strat = sns.color_palette("Set2", 5)

for idx, measure in enumerate(continuous_measures):
    ax = axes[idx // 3, idx % 3]
    data = strategy_df[strategy_df["Measure"] == measure]
    data = data.set_index("Strategy").loc[strategy_order].reset_index()

    bars = ax.bar(
        data["Strategy"],
        data["Mean"],
        yerr=data["Std"],
        capsize=4,
        color=colors_strat,
        edgecolor="black",
        linewidth=1.2,
        error_kw={"linewidth": 1.5},
    )

    ax.set_ylabel("Duration (ms)", fontsize=11)
    ax.set_title(measure_labels[measure], fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    # Highlight min/max for TFT and RRT
    if measure in ["TFT", "RRT"]:
        min_idx = data["Mean"].idxmin()
        max_idx = data["Mean"].idxmax()
        bars[min_idx].set_edgecolor("green")
        bars[min_idx].set_linewidth(3)
        bars[max_idx].set_edgecolor("red")
        bars[max_idx].set_linewidth(3)

    for bar, val in zip(bars, data["Mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + data["Std"].max() * 0.05,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

for idx, measure in enumerate(binary_measures):
    ax = axes[1, idx + 1]
    data = strategy_df[strategy_df["Measure"] == measure]
    data = data.set_index("Strategy").loc[strategy_order].reset_index()

    bars = ax.bar(
        data["Strategy"],
        data["Mean"],
        color=colors_strat,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title(measure_labels[measure], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)

    for bar, val in zip(bars, data["Mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

axes[1, 0].axis("off")

plt.tight_layout()
plt.savefig(
    output_dir / "fig2_decoding_strategy_comparison.png", dpi=150, bbox_inches="tight"
)
plt.savefig(output_dir / "fig2_decoding_strategy_comparison.pdf", bbox_inches="tight")
print(f"  Saved to {output_dir}/fig2_decoding_strategy_comparison.png")

# ============================================================================
# FIGURE 3: Table 11 - Reading measures by Text Type
# ============================================================================
print("\nCreating Figure 3: Reading measures by Text Type...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Table 11: Reading Measures by Text Type\n(Poetry → longest; Fiction → shortest reading times)",
    fontsize=14,
    fontweight="bold",
)

task_order = [
    "Non-fiction",
    "Fiction",
    "Poetry",
    "Summarization",
    "Synopsis",
    "Key words",
]
colors_uncon = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
]  # Blue, orange, green for unconstrained
colors_con = ["#d62728", "#9467bd", "#8c564b"]  # Red, purple, brown for constrained
colors_task = colors_uncon + colors_con

for idx, measure in enumerate(continuous_measures):
    ax = axes[idx // 3, idx % 3]
    data = task_df[task_df["Measure"] == measure]
    data = data.set_index("Task").loc[task_order].reset_index()

    bars = ax.bar(
        data["Task"],
        data["Mean"],
        yerr=data["Std"],
        capsize=4,
        color=colors_task,
        edgecolor="black",
        linewidth=1.2,
        error_kw={"linewidth": 1.5},
    )

    ax.set_ylabel("Duration (ms)", fontsize=11)
    ax.set_title(measure_labels[measure], fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    # Add vertical line to separate unconstrained/constrained
    ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    for bar, val in zip(bars, data["Mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + data["Std"].max() * 0.03,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

for idx, measure in enumerate(binary_measures):
    ax = axes[1, idx + 1]
    data = task_df[task_df["Measure"] == measure]
    data = data.set_index("Task").loc[task_order].reset_index()

    bars = ax.bar(
        data["Task"], data["Mean"], color=colors_task, edgecolor="black", linewidth=1.2
    )

    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title(measure_labels[measure], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    for bar, val in zip(bars, data["Mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

axes[1, 0].axis("off")

plt.tight_layout()
plt.savefig(output_dir / "fig3_text_type_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "fig3_text_type_comparison.pdf", bbox_inches="tight")
print(f"  Saved to {output_dir}/fig3_text_type_comparison.png")

# ============================================================================
# FIGURE 4: Combined heatmap showing all conditions
# ============================================================================
print("\nCreating Figure 4: Heatmap summary...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Summary Heatmaps: Normalized Reading Measures Across Conditions",
    fontsize=14,
    fontweight="bold",
)

# Model heatmap
ax = axes[0]
model_pivot = model_df[model_df["Measure"].isin(continuous_measures)].pivot(
    index="Measure", columns="Model", values="Mean"
)
model_pivot = model_pivot[model_order]
model_pivot = model_pivot.loc[continuous_measures]

# Normalize by row for better visualization
model_norm = model_pivot.div(model_pivot.mean(axis=1), axis=0)
sns.heatmap(
    model_norm,
    annot=model_pivot.round(1),
    fmt="",
    cmap="RdYlGn_r",
    center=1,
    ax=ax,
    cbar_kws={"label": "Relative to mean"},
)
ax.set_title("By Model", fontsize=12, fontweight="bold")
ax.set_ylabel("")

# Strategy heatmap
ax = axes[1]
strat_pivot = strategy_df[strategy_df["Measure"].isin(continuous_measures)].pivot(
    index="Measure", columns="Strategy", values="Mean"
)
strat_pivot = strat_pivot[strategy_order]
strat_pivot = strat_pivot.loc[continuous_measures]

strat_norm = strat_pivot.div(strat_pivot.mean(axis=1), axis=0)
sns.heatmap(
    strat_norm,
    annot=strat_pivot.round(1),
    fmt="",
    cmap="RdYlGn_r",
    center=1,
    ax=ax,
    cbar_kws={"label": "Relative to mean"},
)
ax.set_title("By Decoding Strategy", fontsize=12, fontweight="bold")
ax.set_ylabel("")

# Task heatmap
ax = axes[2]
task_pivot = task_df[task_df["Measure"].isin(continuous_measures)].pivot(
    index="Measure", columns="Task", values="Mean"
)
task_pivot = task_pivot[task_order]
task_pivot = task_pivot.loc[continuous_measures]

task_norm = task_pivot.div(task_pivot.mean(axis=1), axis=0)
sns.heatmap(
    task_norm,
    annot=task_pivot.round(0),
    fmt="",
    cmap="RdYlGn_r",
    center=1,
    ax=ax,
    cbar_kws={"label": "Relative to mean"},
)
ax.set_title("By Text Type", fontsize=12, fontweight="bold")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "fig4_heatmap_summary.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "fig4_heatmap_summary.pdf", bbox_inches="tight")
print(f"  Saved to {output_dir}/fig4_heatmap_summary.png")

# ============================================================================
# FIGURE 5: Key observation - Poetry vs Fiction comparison
# ============================================================================
print("\nCreating Figure 5: Poetry vs Fiction highlight...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Key Finding: Poetry vs Fiction - Distinct Reading Patterns",
    fontsize=14,
    fontweight="bold",
)

# Select poetry and fiction data
highlight_tasks = ["Fiction", "Poetry"]
highlight_data = task_df[task_df["Task"].isin(highlight_tasks)]

# Continuous measures
ax = axes[0]
cont_data = highlight_data[highlight_data["Measure"].isin(continuous_measures)]
x = np.arange(len(continuous_measures))
width = 0.35

fiction_vals = cont_data[cont_data["Task"] == "Fiction"]["Mean"].values
poetry_vals = cont_data[cont_data["Task"] == "Poetry"]["Mean"].values
fiction_std = cont_data[cont_data["Task"] == "Fiction"]["Std"].values
poetry_std = cont_data[cont_data["Task"] == "Poetry"]["Std"].values

bars1 = ax.bar(
    x - width / 2,
    fiction_vals,
    width,
    yerr=fiction_std,
    label="Fiction",
    color="#ff7f0e",
    capsize=5,
    edgecolor="black",
)
bars2 = ax.bar(
    x + width / 2,
    poetry_vals,
    width,
    yerr=poetry_std,
    label="Poetry",
    color="#2ca02c",
    capsize=5,
    edgecolor="black",
)

ax.set_ylabel("Duration (ms)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(
    [measure_labels[m] for m in continuous_measures], rotation=30, ha="right"
)
ax.legend()
ax.set_title("Continuous Measures", fontsize=12)

# Add percentage difference annotations
for i, (f, p) in enumerate(zip(fiction_vals, poetry_vals)):
    diff = ((p - f) / f) * 100
    ax.annotate(
        f"+{diff:.0f}%",
        xy=(i + width / 2, p + poetry_std[i] + 10),
        ha="center",
        fontsize=9,
        color="darkgreen",
        fontweight="bold",
    )

# Binary measures
ax = axes[1]
bin_data = highlight_data[highlight_data["Measure"].isin(binary_measures)]
x = np.arange(len(binary_measures))

fiction_vals = bin_data[bin_data["Task"] == "Fiction"]["Mean"].values
poetry_vals = bin_data[bin_data["Task"] == "Poetry"]["Mean"].values

bars1 = ax.bar(
    x - width / 2,
    fiction_vals,
    width,
    label="Fiction",
    color="#ff7f0e",
    edgecolor="black",
)
bars2 = ax.bar(
    x + width / 2,
    poetry_vals,
    width,
    label="Poetry",
    color="#2ca02c",
    edgecolor="black",
)

ax.set_ylabel("Proportion", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([measure_labels[m] for m in binary_measures])
ax.legend()
ax.set_ylim(0, 1)
ax.set_title("Binary Measures", fontsize=12)

# Add percentage difference annotations
for i, (f, p) in enumerate(zip(fiction_vals, poetry_vals)):
    diff = ((p - f) / f) * 100
    ax.annotate(
        f"+{diff:.0f}%",
        xy=(i + width / 2, p + 0.03),
        ha="center",
        fontsize=9,
        color="darkgreen",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(output_dir / "fig5_poetry_vs_fiction.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "fig5_poetry_vs_fiction.pdf", bbox_inches="tight")
print(f"  Saved to {output_dir}/fig5_poetry_vs_fiction.png")

# ============================================================================
# FIGURE 6: Coefficient of Variation comparison
# ============================================================================
print("\nCreating Figure 6: Coefficient of Variation analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Coefficient of Variation (CV = Std/Mean): Variability Analysis",
    fontsize=14,
    fontweight="bold",
)


# Calculate CV for each grouping
def add_cv(df):
    df = df.copy()
    df["CV"] = df["Std"] / df["Mean"]
    return df


model_df_cv = add_cv(model_df)
strategy_df_cv = add_cv(strategy_df)
task_df_cv = add_cv(task_df)

# Model CV
ax = axes[0]
cv_data = model_df_cv[model_df_cv["Measure"].isin(continuous_measures)]
cv_pivot = cv_data.pivot(index="Measure", columns="Model", values="CV")
cv_pivot = cv_pivot[model_order].loc[continuous_measures]
cv_pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=1)
ax.set_title("By Model", fontsize=12, fontweight="bold")
ax.set_ylabel("Coefficient of Variation")
ax.set_xticklabels([m for m in continuous_measures], rotation=45, ha="right")
ax.legend(title="Model")

# Strategy CV
ax = axes[1]
cv_data = strategy_df_cv[strategy_df_cv["Measure"].isin(continuous_measures)]
cv_pivot = cv_data.pivot(index="Measure", columns="Strategy", values="CV")
cv_pivot = cv_pivot[strategy_order].loc[continuous_measures]
cv_pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=1)
ax.set_title("By Decoding Strategy", fontsize=12, fontweight="bold")
ax.set_ylabel("Coefficient of Variation")
ax.set_xticklabels([m for m in continuous_measures], rotation=45, ha="right")
ax.legend(title="Strategy")

# Task CV
ax = axes[2]
cv_data = task_df_cv[task_df_cv["Measure"].isin(continuous_measures)]
cv_pivot = cv_data.pivot(index="Measure", columns="Task", values="CV")
cv_pivot = cv_pivot[task_order].loc[continuous_measures]
cv_pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=1)
ax.set_title("By Text Type", fontsize=12, fontweight="bold")
ax.set_ylabel("Coefficient of Variation")
ax.set_xticklabels([m for m in continuous_measures], rotation=45, ha="right")
ax.legend(title="Task", fontsize=8)

plt.tight_layout()
plt.savefig(
    output_dir / "fig6_coefficient_of_variation.png", dpi=150, bbox_inches="tight"
)
plt.savefig(output_dir / "fig6_coefficient_of_variation.pdf", bbox_inches="tight")
print(f"  Saved to {output_dir}/fig6_coefficient_of_variation.png")

plt.close("all")

print("\n" + "=" * 80)
print("All figures saved to the 'plots/' directory!")
print("=" * 80)
print("\nFigures created:")
print("  1. fig1_model_comparison.png - Table 9 visualization")
print("  2. fig2_decoding_strategy_comparison.png - Table 10 visualization")
print("  3. fig3_text_type_comparison.png - Table 11 visualization")
print("  4. fig4_heatmap_summary.png - Combined heatmap overview")
print("  5. fig5_poetry_vs_fiction.png - Key finding highlight")
print("  6. fig6_coefficient_of_variation.png - Variability analysis")
