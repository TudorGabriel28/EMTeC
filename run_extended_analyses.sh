#!/bin/bash
# Run all 4 extended analyses on Kaggle
# =====================================
#
# Iteration 1: Baseline FPRT with ms effect sizes
# Iteration 2: Baseline TFT with ms effect sizes  
# Iteration 3: Surprisal effect on FPRT
# Iteration 4: Surprisal effect on TFT

echo "=========================================="
echo "Running Extended Baseline Analyses"
echo "=========================================="

# Create output directories
mkdir -p model_fits plots

# Iteration 1: Baseline FPRT
echo ""
echo ">>> Iteration 1: Baseline FPRT with millisecond effects"
python extended_baseline_analyses.py --analysis baseline -r FPRT -i 6000

# Iteration 2: Baseline TFT
echo ""
echo ">>> Iteration 2: Baseline TFT with millisecond effects"
python extended_baseline_analyses.py --analysis baseline -r TFT -i 6000

# Iteration 3: Surprisal effect on FPRT
echo ""
echo ">>> Iteration 3: Surprisal effect on FPRT"
python extended_baseline_analyses.py --analysis surprisal -r FPRT -i 6000

# Iteration 4: Surprisal effect on TFT
echo ""
echo ">>> Iteration 4: Surprisal effect on TFT"
python extended_baseline_analyses.py --analysis surprisal -r TFT -i 6000

echo ""
echo "=========================================="
echo "All analyses complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  model_fits/baseline_ms_FPRT_summary.csv"
echo "  model_fits/baseline_ms_TFT_summary.csv"
echo "  model_fits/surprisal_FPRT_predictions.csv"
echo "  model_fits/surprisal_TFT_predictions.csv"
echo ""
echo "Plots:"
echo "  plots/baseline_FPRT_ms_effects.png"
echo "  plots/baseline_TFT_ms_effects.png"
echo "  plots/surprisal_effect_FPRT.png"
echo "  plots/surprisal_effect_FPRT_clean.png"
echo "  plots/surprisal_effect_TFT.png"
echo "  plots/surprisal_effect_TFT_clean.png"
