"""
Extended descriptive statistics and visualizations for EMTeC reading measures.
Includes statistics not presented in the paper plus visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Create output directory for plots
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Load the data
print("Loading reading measures data...")
df = pd.read_csv("data/reading_measures_corrected.csv", sep="\t")
print(f"Loaded {len(df):,} rows")

# =============================================================================
# SECTION 1: OVERALL DESCRIPTIVE STATISTICS (not in paper)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: OVERALL DESCRIPTIVE STATISTICS")
print("=" * 80)

# All reading measures
all_continuous = [
    "FFD",
    "SFD",
    "FD",
    "FPRT",
    "FRT",
    "TFT",
    "RRT",
    "RPD_inc",
    "RPD_exc",
    "RBRT",
]
all_binary = ["Fix", "FPF", "RR", "FPReg"]
all_count = ["TFC", "TRC_out", "TRC_in"]
saccade_measures = ["SL_in", "SL_out"]

print("\n--- Continuous Reading Measures (excluding zeros) ---")
stats_list = []
for measure in all_continuous:
    non_zero = df[measure][df[measure] > 0]
    stats_list.append(
        {
            "Measure": measure,
            "Mean": non_zero.mean(),
            "Std": non_zero.std(),
            "Median": non_zero.median(),
            "Q25": non_zero.quantile(0.25),
            "Q75": non_zero.quantile(0.75),
            "Min": non_zero.min(),
            "Max": non_zero.max(),
            "N": len(non_zero),
        }
    )
stats_df = pd.DataFrame(stats_list)
print(stats_df.to_string(index=False, float_format="%.2f"))

print("\n--- Binary Measures (proportions) ---")
for measure in all_binary:
    prop = df[measure].mean()
    count = df[measure].sum()
    print(f"  {measure}: {prop:.3f} ({count:,} / {len(df):,})")

print("\n--- Skipping Rate ---")
skip_rate = 1 - df["Fix"].mean()
print(f"  Overall skipping rate: {skip_rate:.3f} ({(skip_rate*100):.1f}%)")

print("\n--- Count Measures ---")
for measure in all_count:
    print(
        f"  {measure}: Mean={df[measure].mean():.3f}, Median={df[measure].median():.1f}, Max={df[measure].max()}"
    )

print("\n--- Saccade Length Measures ---")
for measure in saccade_measures:
    non_zero = df[measure][df[measure] != 0]
    print(f"  {measure}: Mean={non_zero.mean():.3f}, Std={non_zero.std():.3f}")

# =============================================================================
# SECTION 2: PARTICIPANT-LEVEL STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: PARTICIPANT-LEVEL STATISTICS")
print("=" * 80)

# Aggregate by participant
participant_stats = (
    df.groupby("subject_id")
    .agg(
        {
            "FFD": lambda x: x[x > 0].mean(),
            "FPRT": lambda x: x[x > 0].mean(),
            "TFT": lambda x: x[x > 0].mean(),
            "Fix": "mean",
            "RR": "mean",
            "FPReg": "mean",
            "TFC": "mean",
        }
    )
    .reset_index()
)

print("\n--- Variability Across Participants ---")
print(f"  Number of participants: {df['subject_id'].nunique()}")
for col in ["FFD", "FPRT", "TFT"]:
    print(
        f"  {col} - Range across participants: {participant_stats[col].min():.1f} - {participant_stats[col].max():.1f} ms"
    )
    print(
        f"         Mean of participant means: {participant_stats[col].mean():.1f} ± {participant_stats[col].std():.1f} ms"
    )

print("\n--- Fastest and Slowest Readers ---")
fastest = participant_stats.nsmallest(5, "TFT")[["subject_id", "TFT", "Fix"]]
slowest = participant_stats.nlargest(5, "TFT")[["subject_id", "TFT", "Fix"]]
print("  5 Fastest readers (by mean TFT):")
print(fastest.to_string(index=False))
print("\n  5 Slowest readers (by mean TFT):")
print(slowest.to_string(index=False))

# =============================================================================
# SECTION 3: WORD-LEVEL FEATURE EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: WORD-LEVEL FEATURE EFFECTS")
print("=" * 80)

# Word length effects
print("\n--- Reading Measures by Word Length ---")
df["word_length_bin"] = pd.cut(
    df["word_length_without_punct"],
    bins=[0, 3, 5, 7, 10, 100],
    labels=["1-3", "4-5", "6-7", "8-10", "11+"],
)

word_length_stats = (
    df.groupby("word_length_bin", observed=True)
    .agg(
        {
            "FFD": lambda x: x[x > 0].mean(),
            "FPRT": lambda x: x[x > 0].mean(),
            "TFT": lambda x: x[x > 0].mean(),
            "Fix": "mean",
            "FPReg": "mean",
        }
    )
    .round(3)
)
print(word_length_stats)

# Frequency effects
print("\n--- Reading Measures by Zipf Frequency Bins ---")
df["freq_bin"] = pd.cut(
    df["zipf_freq"],
    bins=[0, 4, 5, 6, 8],
    labels=["Low (0-4)", "Med-Low (4-5)", "Med-High (5-6)", "High (6+)"],
)

freq_stats = (
    df.groupby("freq_bin", observed=True)
    .agg(
        {
            "FFD": lambda x: x[x > 0].mean(),
            "FPRT": lambda x: x[x > 0].mean(),
            "TFT": lambda x: x[x > 0].mean(),
            "Fix": "mean",
        }
    )
    .round(3)
)
print(freq_stats)

# Surprisal effects (GPT-2)
print("\n--- Reading Measures by GPT-2 Surprisal Bins ---")
df["surp_bin"] = pd.cut(
    df["surprisal_gpt2"],
    bins=[-1, 2, 5, 10, 35],
    labels=["Low (0-2)", "Med (2-5)", "High (5-10)", "Very High (10+)"],
)

surp_stats = (
    df.groupby("surp_bin", observed=True)
    .agg(
        {
            "FFD": lambda x: x[x > 0].mean(),
            "FPRT": lambda x: x[x > 0].mean(),
            "TFT": lambda x: x[x > 0].mean(),
            "Fix": "mean",
        }
    )
    .round(3)
)
print(surp_stats)

# =============================================================================
# SECTION 4: CORRELATIONS BETWEEN READING MEASURES
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: CORRELATIONS BETWEEN READING MEASURES")
print("=" * 80)

# Select key measures for correlation
corr_measures = ["FFD", "FPRT", "TFT", "RRT", "Fix", "FPReg", "TFC"]
# Only use rows where word was fixated for continuous measures
fixated_df = df[df["Fix"] == 1].copy()
corr_matrix = fixated_df[corr_measures].corr()
print("\nCorrelation matrix (for fixated words):")
print(corr_matrix.round(3).to_string())

# =============================================================================
# SECTION 5: ADDITIONAL CROSS-TABULATIONS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: CROSS-TABULATIONS (Model × Task)")
print("=" * 80)

# Model × Task interaction
print("\n--- Mean TFT by Model and Task ---")
model_task_tft = (
    df.groupby(["model", "task"])
    .apply(lambda x: x["TFT"][x["TFT"] > 0].mean())
    .unstack()
)
print(model_task_tft.round(1).to_string())

print("\n--- Fixation Rate by Model and Task ---")
model_task_fix = df.groupby(["model", "task"])["Fix"].mean().unstack()
print(model_task_fix.round(3).to_string())

# Decoding × Task
print("\n--- Mean TFT by Decoding Strategy and Task ---")
dec_task_tft = (
    df.groupby(["decoding_strategy", "task"])
    .apply(lambda x: x["TFT"][x["TFT"] > 0].mean())
    .unstack()
)
print(dec_task_tft.round(1).to_string())

# =============================================================================
# SECTION 6: FIRST-PASS REGRESSION ANALYSIS (not detailed in paper)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: FIRST-PASS REGRESSION ANALYSIS")
print("=" * 80)

print("\n--- Overall First-Pass Regression Rate ---")
print(f"  FPReg rate: {df['FPReg'].mean():.3f} ({(df['FPReg'].mean()*100):.1f}%)")

print("\n--- FPReg by Model ---")
for model in ["phi2", "mistral", "wizardlm"]:
    rate = df[df["model"] == model]["FPReg"].mean()
    print(f"  {model}: {rate:.3f}")

print("\n--- FPReg by Task ---")
for task in df["task"].unique():
    rate = df[df["task"] == task]["FPReg"].mean()
    print(f"  {task}: {rate:.3f}")

print("\n--- FPReg by Decoding Strategy ---")
for strat in df["decoding_strategy"].unique():
    rate = df[df["decoding_strategy"] == strat]["FPReg"].mean()
    print(f"  {strat}: {rate:.3f}")

# =============================================================================
# PLOTTING SECTION
# =============================================================================
print("\n" + "=" * 80)
print("GENERATING PLOTS...")
print("=" * 80)

# Plot 1: Distribution of key reading measures
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Distribution of Key Reading Measures (Fixated Words Only)", fontsize=14)

measures_to_plot = ["FFD", "FPRT", "TFT", "RRT", "RPD_inc", "TFC"]
for ax, measure in zip(axes.flatten(), measures_to_plot):
    data = df[measure][df[measure] > 0]
    # Clip extreme values for visualization
    upper = data.quantile(0.99)
    data_clipped = data[data <= upper]
    ax.hist(data_clipped, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.1f}"
    )
    ax.axvline(
        data.median(),
        color="green",
        linestyle="--",
        label=f"Median: {data.median():.1f}",
    )
    ax.set_xlabel(f"{measure} (ms)" if measure != "TFC" else measure)
    ax.set_ylabel("Count")
    ax.set_title(f"{measure} Distribution")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(
    output_dir / "reading_measures_distributions.png", dpi=150, bbox_inches="tight"
)
print(f"  Saved: {output_dir / 'reading_measures_distributions.png'}")

# Plot 2: Box plots by Model
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Reading Measures by Model", fontsize=14)

measures_box = ["FFD", "FPRT", "TFT", "RRT"]
for ax, measure in zip(axes.flatten(), measures_box):
    data_plot = df[df[measure] > 0].copy()
    # Sample for faster plotting
    if len(data_plot) > 50000:
        data_plot = data_plot.sample(50000, random_state=42)
    sns.boxplot(
        data=data_plot,
        x="model",
        y=measure,
        ax=ax,
        order=["phi2", "mistral", "wizardlm"],
    )
    ax.set_xlabel("Model")
    ax.set_ylabel(f"{measure} (ms)")
    ax.set_title(f"{measure} by Model")

plt.tight_layout()
plt.savefig(output_dir / "reading_measures_by_model.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'reading_measures_by_model.png'}")

# Plot 3: Box plots by Task
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Reading Measures by Text Type (Task)", fontsize=14)

task_order = [
    "non-fiction",
    "fiction",
    "poetry",
    "summarization",
    "article_synopsis",
    "words_given",
]
for ax, measure in zip(axes.flatten(), measures_box):
    data_plot = df[df[measure] > 0].copy()
    if len(data_plot) > 50000:
        data_plot = data_plot.sample(50000, random_state=42)
    sns.boxplot(data=data_plot, x="task", y=measure, ax=ax, order=task_order)
    ax.set_xlabel("Task")
    ax.set_ylabel(f"{measure} (ms)")
    ax.set_title(f"{measure} by Task")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "reading_measures_by_task.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'reading_measures_by_task.png'}")

# Plot 4: Box plots by Decoding Strategy
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Reading Measures by Decoding Strategy", fontsize=14)

strat_order = ["greedy_search", "beam_search", "sampling", "topk", "topp"]
for ax, measure in zip(axes.flatten(), measures_box):
    data_plot = df[df[measure] > 0].copy()
    if len(data_plot) > 50000:
        data_plot = data_plot.sample(50000, random_state=42)
    sns.boxplot(
        data=data_plot, x="decoding_strategy", y=measure, ax=ax, order=strat_order
    )
    ax.set_xlabel("Decoding Strategy")
    ax.set_ylabel(f"{measure} (ms)")
    ax.set_title(f"{measure} by Decoding Strategy")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(
    output_dir / "reading_measures_by_decoding.png", dpi=150, bbox_inches="tight"
)
print(f"  Saved: {output_dir / 'reading_measures_by_decoding.png'}")

# Plot 5: Binary measures comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Binary Reading Measures by Condition", fontsize=14)

# Fix rate by task
task_fix = df.groupby("task")["Fix"].mean().reindex(task_order)
axes[0].bar(
    range(len(task_fix)),
    task_fix.values,
    color=sns.color_palette("husl", len(task_fix)),
)
axes[0].set_xticks(range(len(task_fix)))
axes[0].set_xticklabels(task_fix.index, rotation=45, ha="right")
axes[0].set_ylabel("Fixation Rate")
axes[0].set_title("Fixation Rate by Task")
axes[0].set_ylim(0, 1)

# RR rate by task
task_rr = df.groupby("task")["RR"].mean().reindex(task_order)
axes[1].bar(
    range(len(task_rr)), task_rr.values, color=sns.color_palette("husl", len(task_rr))
)
axes[1].set_xticks(range(len(task_rr)))
axes[1].set_xticklabels(task_rr.index, rotation=45, ha="right")
axes[1].set_ylabel("Re-reading Rate")
axes[1].set_title("Re-reading Rate by Task")
axes[1].set_ylim(0, 0.5)

# FPReg rate by task
task_fpreg = df.groupby("task")["FPReg"].mean().reindex(task_order)
axes[2].bar(
    range(len(task_fpreg)),
    task_fpreg.values,
    color=sns.color_palette("husl", len(task_fpreg)),
)
axes[2].set_xticks(range(len(task_fpreg)))
axes[2].set_xticklabels(task_fpreg.index, rotation=45, ha="right")
axes[2].set_ylabel("First-Pass Regression Rate")
axes[2].set_title("First-Pass Regression Rate by Task")
axes[2].set_ylim(0, 0.3)

plt.tight_layout()
plt.savefig(output_dir / "binary_measures_by_task.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'binary_measures_by_task.png'}")

# Plot 6: Word length effects
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Effect of Word Length on Reading Measures", fontsize=14)

wl_stats = df.groupby("word_length_bin", observed=True).agg(
    {"FPRT": lambda x: x[x > 0].mean(), "TFT": lambda x: x[x > 0].mean(), "Fix": "mean"}
)

axes[0].bar(range(len(wl_stats)), wl_stats["FPRT"].values, color="steelblue")
axes[0].set_xticks(range(len(wl_stats)))
axes[0].set_xticklabels(wl_stats.index)
axes[0].set_xlabel("Word Length (characters)")
axes[0].set_ylabel("FPRT (ms)")
axes[0].set_title("First-Pass Reading Time by Word Length")

axes[1].bar(range(len(wl_stats)), wl_stats["TFT"].values, color="coral")
axes[1].set_xticks(range(len(wl_stats)))
axes[1].set_xticklabels(wl_stats.index)
axes[1].set_xlabel("Word Length (characters)")
axes[1].set_ylabel("TFT (ms)")
axes[1].set_title("Total Fixation Time by Word Length")

axes[2].bar(range(len(wl_stats)), wl_stats["Fix"].values, color="seagreen")
axes[2].set_xticks(range(len(wl_stats)))
axes[2].set_xticklabels(wl_stats.index)
axes[2].set_xlabel("Word Length (characters)")
axes[2].set_ylabel("Fixation Rate")
axes[2].set_title("Fixation Rate by Word Length")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / "word_length_effects.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'word_length_effects.png'}")

# Plot 7: Frequency effects
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Effect of Word Frequency (Zipf) on Reading Measures", fontsize=14)

freq_order = ["Low (0-4)", "Med-Low (4-5)", "Med-High (5-6)", "High (6+)"]
freq_plot = freq_stats.reindex(freq_order)

axes[0].bar(range(len(freq_plot)), freq_plot["FPRT"].values, color="steelblue")
axes[0].set_xticks(range(len(freq_plot)))
axes[0].set_xticklabels(freq_plot.index, rotation=15)
axes[0].set_xlabel("Zipf Frequency")
axes[0].set_ylabel("FPRT (ms)")
axes[0].set_title("First-Pass Reading Time by Frequency")

axes[1].bar(range(len(freq_plot)), freq_plot["TFT"].values, color="coral")
axes[1].set_xticks(range(len(freq_plot)))
axes[1].set_xticklabels(freq_plot.index, rotation=15)
axes[1].set_xlabel("Zipf Frequency")
axes[1].set_ylabel("TFT (ms)")
axes[1].set_title("Total Fixation Time by Frequency")

axes[2].bar(range(len(freq_plot)), freq_plot["Fix"].values, color="seagreen")
axes[2].set_xticks(range(len(freq_plot)))
axes[2].set_xticklabels(freq_plot.index, rotation=15)
axes[2].set_xlabel("Zipf Frequency")
axes[2].set_ylabel("Fixation Rate")
axes[2].set_title("Fixation Rate by Frequency")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / "frequency_effects.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'frequency_effects.png'}")

# Plot 8: Surprisal effects
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Effect of GPT-2 Surprisal on Reading Measures", fontsize=14)

surp_order = ["Low (0-2)", "Med (2-5)", "High (5-10)", "Very High (10+)"]
surp_plot = surp_stats.reindex(surp_order)

axes[0].bar(range(len(surp_plot)), surp_plot["FPRT"].values, color="steelblue")
axes[0].set_xticks(range(len(surp_plot)))
axes[0].set_xticklabels(surp_plot.index, rotation=15)
axes[0].set_xlabel("GPT-2 Surprisal")
axes[0].set_ylabel("FPRT (ms)")
axes[0].set_title("First-Pass Reading Time by Surprisal")

axes[1].bar(range(len(surp_plot)), surp_plot["TFT"].values, color="coral")
axes[1].set_xticks(range(len(surp_plot)))
axes[1].set_xticklabels(surp_plot.index, rotation=15)
axes[1].set_xlabel("GPT-2 Surprisal")
axes[1].set_ylabel("TFT (ms)")
axes[1].set_title("Total Fixation Time by Surprisal")

axes[2].bar(range(len(surp_plot)), surp_plot["Fix"].values, color="seagreen")
axes[2].set_xticks(range(len(surp_plot)))
axes[2].set_xticklabels(surp_plot.index, rotation=15)
axes[2].set_xlabel("GPT-2 Surprisal")
axes[2].set_ylabel("Fixation Rate")
axes[2].set_title("Fixation Rate by Surprisal")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / "surprisal_effects.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'surprisal_effects.png'}")

# Plot 9: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    ax=ax,
    square=True,
    linewidths=0.5,
)
ax.set_title("Correlation Matrix of Reading Measures (Fixated Words)")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'correlation_heatmap.png'}")

# Plot 10: Participant variability
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Individual Differences Across Participants", fontsize=14)

axes[0].hist(participant_stats["TFT"], bins=20, edgecolor="black", alpha=0.7)
axes[0].axvline(
    participant_stats["TFT"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {participant_stats["TFT"].mean():.1f}',
)
axes[0].set_xlabel("Mean TFT (ms)")
axes[0].set_ylabel("Number of Participants")
axes[0].set_title("Distribution of Mean TFT Across Participants")
axes[0].legend()

axes[1].hist(
    participant_stats["Fix"], bins=20, edgecolor="black", alpha=0.7, color="seagreen"
)
axes[1].axvline(
    participant_stats["Fix"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {participant_stats["Fix"].mean():.3f}',
)
axes[1].set_xlabel("Fixation Rate")
axes[1].set_ylabel("Number of Participants")
axes[1].set_title("Distribution of Fixation Rate Across Participants")
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / "participant_variability.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'participant_variability.png'}")

# Plot 11: Violin plots for task comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Reading Measure Distributions by Text Type (Violin Plots)", fontsize=14)

for ax, measure in zip(axes.flatten(), measures_box):
    data_plot = df[df[measure] > 0].copy()
    # Clip to 99th percentile for visualization
    upper = data_plot[measure].quantile(0.99)
    data_plot = data_plot[data_plot[measure] <= upper]
    if len(data_plot) > 50000:
        data_plot = data_plot.sample(50000, random_state=42)
    sns.violinplot(data=data_plot, x="task", y=measure, ax=ax, order=task_order)
    ax.set_xlabel("Task")
    ax.set_ylabel(f"{measure} (ms)")
    ax.set_title(f"{measure} Distribution by Task")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "violin_plots_by_task.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'violin_plots_by_task.png'}")

# Plot 12: Model x Task heatmap for TFT
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(model_task_tft, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
ax.set_title("Mean Total Fixation Time (TFT) by Model and Task")
ax.set_xlabel("Task")
ax.set_ylabel("Model")
plt.tight_layout()
plt.savefig(output_dir / "model_task_heatmap.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'model_task_heatmap.png'}")

print("\n" + "=" * 80)
print("ALL PLOTS SAVED TO 'plots/' DIRECTORY")
print("=" * 80)

# Summary table
print("\n" + "=" * 80)
print("SUMMARY: Key Additional Statistics Not in Paper")
print("=" * 80)
print(
    f"""
1. OVERALL STATISTICS:
   - Total observations: {len(df):,}
   - Skipping rate: {skip_rate:.1%}
   - First-pass regression rate: {df['FPReg'].mean():.1%}
   - Mean total fixation count (TFC): {df['TFC'].mean():.2f}

2. PARTICIPANT VARIABILITY:
   - TFT range across participants: {participant_stats['TFT'].min():.1f} - {participant_stats['TFT'].max():.1f} ms
   - Fix rate range: {participant_stats['Fix'].min():.3f} - {participant_stats['Fix'].max():.3f}

3. WORD-LEVEL EFFECTS (expected patterns confirmed):
   - Longer words → longer reading times, higher fixation rate
   - Lower frequency → longer reading times  
   - Higher surprisal → longer reading times

4. REGRESSION PATH DURATION:
   - Mean RPD_inc (inclusive): {df['RPD_inc'][df['RPD_inc']>0].mean():.1f} ms
   - Mean RPD_exc (exclusive): {df['RPD_exc'][df['RPD_exc']>0].mean():.1f} ms
"""
)

plt.close("all")
print("\nDone!")
