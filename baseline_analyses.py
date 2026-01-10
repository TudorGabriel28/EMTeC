#!/usr/bin/env python
"""
Bayesian Regression Analyses for Eye-Tracking Reading Measures
==============================================================

This script replicates the psycholinguistic analyses from the EMTeC paper.
It fits Bayesian mixed-effects regression models to predict reading measures
from lexical predictors (word length, lexical frequency, surprisal).

The analyses follow the methodology in baseline_analyses.R, using:
- Lognormal models for continuous reading time measures
- Bernoulli (logistic) models for binary fixation measures
- Poisson models for count measures

Usage:
    python baseline_analyses.py --response FPRT --iterations 6000
    python baseline_analyses.py -r TFT -i 6000
"""

import argparse
import warnings
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Define variable types for model selection
LOG_LINEAR_VARIABLES = ["FFD", "SFD", "FD", "FPRT", "FRT", "TFT", "RRT"]
BINARY_VARIABLES = ["Fix", "FPF", "RR", "FPReg"]
COUNT_VARIABLES = ["TFC"]

# Model parameters
WARMUP_ITERATIONS = 2000
DEFAULT_ITERATIONS = 6000
N_CHAINS = 4
RANDOM_SEED = 42


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the reading measures data.

    Preprocessing steps:
    - Convert subject_id to categorical
    - Scale predictors (z-score standardization)
    - Create log lexical frequency from Zipf frequency

    Parameters
    ----------
    filepath : str
        Path to the reading measures CSV file

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe ready for modeling
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep="\t")

    print(f"  Loaded {len(df):,} observations")
    print(f"  Number of subjects: {df['subject_id'].nunique()}")
    print(f"  Number of items: {df['item_id'].nunique()}")

    # Convert subject_id to categorical
    df["subject_id"] = df["subject_id"].astype("category")

    # Scale predictors (z-score standardization)
    # This matches the R code: scale(df$zipf_freq), scale(df$word_length_with_punct), etc.
    df["log_lex_freq"] = stats.zscore(df["zipf_freq"])
    df["word_length"] = stats.zscore(df["word_length_with_punct"])
    df["surprisal"] = stats.zscore(df["surprisal_gpt2"])

    # Ensure last_in_line is numeric (0/1)
    df["last_in_line"] = df["last_in_line"].astype(int)

    print("  Preprocessing complete.")
    return df


def get_model_formula(response_var: str) -> str:
    """
    Construct the model formula.

    The formula includes:
    - Random intercept for subject_id
    - Fixed effects: word_length, log_lex_freq, surprisal, last_in_line

    Parameters
    ----------
    response_var : str
        The response variable name

    Returns
    -------
    str
        Bambi formula string
    """
    return f"{response_var} ~ (1 | subject_id) + word_length + log_lex_freq + surprisal + last_in_line"


def run_lognormal_model(
    formula: str, data: pd.DataFrame, iterations: int = DEFAULT_ITERATIONS
) -> bmb.Model:
    """
    Fit a Gaussian regression model on log-transformed reading time measures.

    This is appropriate for positively-skewed reaction time data.
    We log-transform the response and fit a Gaussian model: log(y) ~ Normal(μ, σ)
    This is equivalent to a lognormal model.

    Priors (matching R code):
    - Intercept: Normal(6, 1.5)
    - Fixed effects: Normal(0, 1)
    - Sigma: HalfNormal(1)

    Parameters
    ----------
    formula : str
        Model formula
    data : pd.DataFrame
        Data for fitting
    iterations : int
        Number of MCMC iterations

    Returns
    -------
    bmb.Model
        Fitted Bambi model
    """
    print("\nFitting lognormal model (via log-transformed Gaussian)...")

    # Extract response variable name from formula
    response_var = formula.split("~")[0].strip()

    # Log-transform the response variable
    data = data.copy()
    data[response_var] = np.log(data[response_var])

    # Define priors matching the R code
    priors = {
        "Intercept": bmb.Prior("Normal", mu=6, sigma=1.5),
        "word_length": bmb.Prior("Normal", mu=0, sigma=1),
        "log_lex_freq": bmb.Prior("Normal", mu=0, sigma=1),
        "surprisal": bmb.Prior("Normal", mu=0, sigma=1),
        "last_in_line": bmb.Prior("Normal", mu=0, sigma=1),
        "sigma": bmb.Prior("HalfNormal", sigma=1),
    }

    # Build and fit model using Gaussian family on log-transformed data
    model = bmb.Model(formula, data, family="gaussian", priors=priors)

    results = model.fit(
        draws=max(iterations - WARMUP_ITERATIONS, 1000),
        tune=min(WARMUP_ITERATIONS, iterations // 2),
        chains=N_CHAINS,
        random_seed=RANDOM_SEED,
        progressbar=True,
    )

    return model, results


def run_bernoulli_model(
    formula: str, data: pd.DataFrame, iterations: int = DEFAULT_ITERATIONS
) -> bmb.Model:
    """
    Fit a Bernoulli (logistic) regression model for binary measures.

    This is appropriate for binary outcomes like fixation (yes/no).
    Uses logit link function: logit(p) = Xβ

    Priors (matching R code):
    - Intercept: Normal(0, 4)
    - Fixed effects: Normal(0, 1)

    Parameters
    ----------
    formula : str
        Model formula
    data : pd.DataFrame
        Data for fitting
    iterations : int
        Number of MCMC iterations

    Returns
    -------
    bmb.Model
        Fitted Bambi model
    """
    print("\nFitting Bernoulli (logistic) model...")

    # Define priors matching the R code
    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=4),
        "word_length": bmb.Prior("Normal", mu=0, sigma=1),
        "log_lex_freq": bmb.Prior("Normal", mu=0, sigma=1),
        "surprisal": bmb.Prior("Normal", mu=0, sigma=1),
        "last_in_line": bmb.Prior("Normal", mu=0, sigma=1),
    }

    model = bmb.Model(formula, data, family="bernoulli", priors=priors)

    results = model.fit(
        draws=max(iterations - WARMUP_ITERATIONS, 1000),
        tune=min(WARMUP_ITERATIONS, iterations // 2),
        chains=N_CHAINS,
        random_seed=RANDOM_SEED,
        progressbar=True,
    )

    return model, results


def run_poisson_model(
    formula: str, data: pd.DataFrame, iterations: int = DEFAULT_ITERATIONS
) -> bmb.Model:
    """
    Fit a Poisson regression model for count measures.

    This is appropriate for count data like total fixation count (TFC).
    Uses log link function: log(λ) = Xβ

    Parameters
    ----------
    formula : str
        Model formula
    data : pd.DataFrame
        Data for fitting
    iterations : int
        Number of MCMC iterations

    Returns
    -------
    bmb.Model
        Fitted Bambi model
    """
    print("\nFitting Poisson model...")

    model = bmb.Model(formula, data, family="poisson")

    results = model.fit(
        draws=max(iterations - WARMUP_ITERATIONS, 1000),
        tune=min(WARMUP_ITERATIONS, iterations // 2),
        chains=N_CHAINS,
        random_seed=RANDOM_SEED,
        progressbar=True,
    )

    return model, results


def print_model_summary(results, response_var: str):
    """
    Print a summary of the fitted model results.

    Parameters
    ----------
    results : InferenceData
        ArviZ InferenceData object with posterior samples
    response_var : str
        Name of the response variable
    """
    print(f"\n{'='*60}")
    print(f"Model Summary for {response_var}")
    print("=" * 60)

    # Print ArviZ summary
    summary = az.summary(
        results,
        var_names=[
            "Intercept",
            "word_length",
            "log_lex_freq",
            "surprisal",
            "last_in_line",
        ],
        hdi_prob=0.95,
    )
    print(summary)

    # Print convergence diagnostics
    print(f"\n{'='*60}")
    print("Convergence Diagnostics")
    print("=" * 60)

    # R-hat values
    rhat = az.rhat(results)
    print("\nR-hat values (should be close to 1.0):")
    for var in [
        "Intercept",
        "word_length",
        "log_lex_freq",
        "surprisal",
        "last_in_line",
    ]:
        if var in rhat:
            print(f"  {var}: {float(rhat[var].values):.4f}")

    # Effective sample size
    ess = az.ess(results)
    print("\nEffective Sample Size:")
    for var in [
        "Intercept",
        "word_length",
        "log_lex_freq",
        "surprisal",
        "last_in_line",
    ]:
        if var in ess:
            print(f"  {var}: {float(ess[var].values):.0f}")


def save_results(model, results, response_var: str, output_dir: str = "model_fits"):
    """
    Save model results to disk.

    Parameters
    ----------
    model : bmb.Model
        The fitted Bambi model
    results : InferenceData
        ArviZ InferenceData object
    response_var : str
        Name of the response variable
    output_dir : str
        Directory for saving results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save the InferenceData (posterior samples)
    netcdf_path = output_path / f"baseline_{response_var}.nc"
    results.to_netcdf(str(netcdf_path))
    print(f"\nResults saved to {netcdf_path}")

    # Save summary as CSV
    summary = az.summary(results, hdi_prob=0.95)
    csv_path = output_path / f"baseline_{response_var}_summary.csv"
    summary.to_csv(csv_path)
    print(f"Summary saved to {csv_path}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Fit Bayesian regression models for reading measures."
    )
    parser.add_argument(
        "-r",
        "--response",
        type=str,
        default="FPRT",
        help="Response variable to model (e.g., FPRT, TFT, Fix, FPReg, TFC)",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of MCMC iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/reading_measures_corrected.csv",
        help="Path to the reading measures data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_fits",
        help="Directory for saving model results",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to use (0.0-1.0). Use smaller values for faster testing."
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Number of subjects to include. Use smaller values for faster testing."
    )
    args = parser.parse_args()
    response_var = args.response
    iterations = args.iterations

    print("=" * 60)
    print("EMTeC Baseline Psycholinguistic Analysis")
    print("=" * 60)
    print(f"\nResponse variable: {response_var}")
    print(f"Iterations: {iterations}")
    print(f"Warmup: {WARMUP_ITERATIONS}")
    print(f"Chains: {N_CHAINS}")

    # Validate response variable
    all_valid = LOG_LINEAR_VARIABLES + BINARY_VARIABLES + COUNT_VARIABLES
    if response_var not in all_valid:
        raise ValueError(
            f"Invalid response variable: {response_var}. " f"Valid options: {all_valid}"
        )

    # Load and preprocess data
    data = load_and_preprocess_data(args.data_path)
    
    # Optionally subsample data for faster testing
    if args.n_subjects is not None:
        subjects = data["subject_id"].unique()[:args.n_subjects]
        data = data[data["subject_id"].isin(subjects)].copy()
        print(f"\nSubsampled to {args.n_subjects} subjects: {len(data):,} observations")
    
    if args.sample_frac < 1.0:
        data = data.sample(frac=args.sample_frac, random_state=RANDOM_SEED).copy()
        print(f"\nSubsampled to {args.sample_frac*100:.1f}% of data: {len(data):,} observations")

    # Get the model formula
    formula = get_model_formula(response_var)
    print(f"\nModel formula: {formula}")

    # Select appropriate model type and fit
    if response_var in LOG_LINEAR_VARIABLES:
        # For lognormal models, remove zero values
        data_subset = data[data[response_var] > 0].copy()
        print(f"\nFiltered to {len(data_subset):,} non-zero observations")
        model, results = run_lognormal_model(formula, data_subset, iterations)

    elif response_var in BINARY_VARIABLES:
        model, results = run_bernoulli_model(formula, data, iterations)

    elif response_var in COUNT_VARIABLES:
        model, results = run_poisson_model(formula, data, iterations)

    # Print and save results
    print_model_summary(results, response_var)
    save_results(model, results, response_var, args.output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return model, results


if __name__ == "__main__":
    main()
