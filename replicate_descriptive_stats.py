"""
Replicate descriptive statistics from the EMTeC paper.
Tables 9, 10, 11: Reading measures by model, decoding strategy, and text type.
"""

import pandas as pd
import numpy as np

# Load the data
print("Loading reading measures data...")
df = pd.read_csv("data/reading_measures_corrected.csv", sep="\t")
print(f"Loaded {len(df)} rows")

# Display column names to verify
print("\nColumns available:")
print(df.columns.tolist())

# Reading measures to analyze (continuous)
continuous_measures = ["FFD", "FPRT", "TFT", "RRT"]
# Binary measures
binary_measures = ["Fix", "RR"]


def compute_stats_continuous(data, measure):
    """Compute mean and std for continuous measures, excluding zeros."""
    values = data[measure]
    non_zero = values[values > 0]
    return non_zero.mean(), non_zero.std()


def compute_stats_binary(data, measure):
    """Compute mean proportion for binary measures."""
    return data[measure].mean()


def print_table(title, groups, group_col, df):
    """Print a formatted table of descriptive statistics."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    for group in groups:
        group_data = df[df[group_col] == group]
        print(f"\n{group}")
        print("-" * 60)

        # Continuous measures
        print("Mean and standard deviation of continuous measures in ms:")
        for measure in continuous_measures:
            mean, std = compute_stats_continuous(group_data, measure)
            print(f"  {measure}: {mean:.3f} ± {std:.3f}")

        # Binary measures
        print("\nMean proportions of binary measures:")
        for measure in binary_measures:
            prop = compute_stats_binary(group_data, measure)
            print(f"  {measure}: {prop:.3f}")


# ============================================================================
# TABLE 9: Descriptive statistics for each model
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 9: Descriptive statistics of reading measures for each MODEL")
print("=" * 80)

models = df["model"].unique()
print(f"\nModels found in data: {models}")

for model in ["phi2", "mistral", "wizardlm"]:
    if model in df["model"].values:
        model_data = df[df["model"] == model]
        print(f"\n{model.capitalize()}")
        print("-" * 60)

        print("Mean and standard deviation of continuous measures in ms:")
        for measure in continuous_measures:
            mean, std = compute_stats_continuous(model_data, measure)
            print(f"  {measure}: {mean:.3f} ± {std:.3f}")

        print("\nMean proportions of binary measures:")
        for measure in binary_measures:
            prop = compute_stats_binary(model_data, measure)
            print(f"  {measure}: {prop:.3f}")

# ============================================================================
# TABLE 10: Descriptive statistics for each decoding strategy
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 10: Descriptive statistics of reading measures for each DECODING STRATEGY")
print("=" * 80)

strategies = df["decoding_strategy"].unique()
print(f"\nDecoding strategies found in data: {strategies}")

strategy_names = {
    "greedy": "Greedy search",
    "greedy_search": "Greedy search",
    "beam": "Beam search",
    "beam_search": "Beam search",
    "sampling": "Sampling",
    "topk": "Top-k",
    "topp": "Top-p",
}

for strategy in [
    "greedy",
    "greedy_search",
    "beam",
    "beam_search",
    "sampling",
    "topk",
    "topp",
]:
    if strategy in df["decoding_strategy"].values:
        strategy_data = df[df["decoding_strategy"] == strategy]
        print(f"\n{strategy_names.get(strategy, strategy)}")
        print("-" * 60)

        print("Mean and standard deviation of continuous measures in ms:")
        for measure in continuous_measures:
            mean, std = compute_stats_continuous(strategy_data, measure)
            print(f"  {measure}: {mean:.3f} ± {std:.3f}")

        print("\nMean proportions of binary measures:")
        for measure in binary_measures:
            prop = compute_stats_binary(strategy_data, measure)
            print(f"  {measure}: {prop:.3f}")

# ============================================================================
# TABLE 11: Descriptive statistics for each text type (task)
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 11: Descriptive statistics of reading measures for each TEXT TYPE")
print("=" * 80)

# Check available task types
tasks = df["task"].unique()
print(f"\nTasks found in data: {tasks}")

# Also check 'type' column if it exists
if "type" in df.columns:
    types = df["type"].unique()
    print(f"Types found in data: {types}")

# Map task names to paper names
task_names = {
    "non-fiction": "Non-fiction",
    "nonfiction": "Non-fiction",
    "fiction": "Fiction",
    "poetry": "Poetry",
    "summarization": "Summarization",
    "synopsis": "Synopsis",
    "article_synopsis": "Synopsis",
    "keywords": "Key words",
    "key_words": "Key words",
    "key words": "Key words",
    "words_given": "Key words",
}

# Group by unconstrained vs constrained
print("\nUnconstrained text types:")
for task in ["non-fiction", "nonfiction", "fiction", "poetry"]:
    if task in df["task"].values:
        task_data = df[df["task"] == task]
        print(f"\n{task_names.get(task, task)}")
        print("-" * 60)

        print("Mean and standard deviation of continuous measures in ms:")
        for measure in continuous_measures:
            mean, std = compute_stats_continuous(task_data, measure)
            print(f"  {measure}: {mean:.3f} ± {std:.3f}")

        print("\nMean proportions of binary measures:")
        for measure in binary_measures:
            prop = compute_stats_binary(task_data, measure)
            print(f"  {measure}: {prop:.3f}")

print("\nConstrained text types:")
for task in [
    "summarization",
    "synopsis",
    "article_synopsis",
    "keywords",
    "key_words",
    "key words",
    "words_given",
]:
    if task in df["task"].values:
        task_data = df[df["task"] == task]
        print(f"\n{task_names.get(task, task)}")
        print("-" * 60)

        print("Mean and standard deviation of continuous measures in ms:")
        for measure in continuous_measures:
            mean, std = compute_stats_continuous(task_data, measure)
            print(f"  {measure}: {mean:.3f} ± {std:.3f}")

        print("\nMean proportions of binary measures:")
        for measure in binary_measures:
            prop = compute_stats_binary(task_data, measure)
            print(f"  {measure}: {prop:.3f}")

# ============================================================================
# Summary comparison with paper values
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON WITH PAPER VALUES")
print("=" * 80)

print("\nPaper Table 9 (Model statistics) - Expected values:")
print(
    """
Phi-2:     FFD: 223.165±101.362, FPRT: 268.674±154.69, TFT: 326.477±226.091, RRT: 295.616±213.118
           Fix: 0.694, RR: 0.221
Mistral:   FFD: 222.166±103.895, FPRT: 265.261±151.312, TFT: 325.399±230.474, RRT: 300.596±224.297
           Fix: 0.694, RR: 0.220
WizardLM:  FFD: 222.414±100.985, FPRT: 266.831±151.84, TFT: 327.748±231.686, RRT: 304.003±229.13
           Fix: 0.695, RR: 0.221
"""
)

print("\nData overview:")
print(f"Total observations: {len(df)}")
print(f"Unique subjects: {df['subject_id'].nunique()}")
print(f"Unique items: {df['item_id'].nunique()}")
