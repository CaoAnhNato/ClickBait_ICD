#!/bin/bash

# Script chạy Full Ablation cho ICDv6
# Các cấu hình: C0 (Simple), C0 (ESIM), C1 (Reweight), C2 (Residual)

HW_PROFILE="ada5000" # Mặc định cho máy ảo mạnh, có thể đổi thành rtx3050
EPOCHS=15

echo "================================================================"
echo "STARTING FULL ABLATION STUDY - ICDv6"
echo "================================================================"

# 1. C0: Simple Baseline
echo -e "\n>>> Running Experiment 1: C0 Simple Baseline"
python training/ICD/train_ICD_v6.py --hw_profile $HW_PROFILE --variant simple --epochs $EPOCHS --run_name "ICDv6_C0_Simple"

# 2. C0: ESIM Variant
echo -e "\n>>> Running Experiment 2: C0 ESIM Interaction"
python training/ICD/train_ICD_v6.py --hw_profile $HW_PROFILE --variant esim --epochs $EPOCHS --run_name "ICDv6_C0_ESIM"

# 3. C1: Loss Reweighting
echo -e "\n>>> Running Experiment 3: C1 Loss Reweighting"
python training/ICD/train_ICD_v6.py --hw_profile $HW_PROFILE --variant simple --use_reweighting --epochs $EPOCHS --run_name "ICDv6_C1_Reweight"

# 4. C2: Residual Head (Full Model)
echo -e "\n>>> Running Experiment 4: C2 Pattern Residual (Full)"
python training/ICD/train_ICD_v6.py --hw_profile $HW_PROFILE --variant simple --use_reweighting --use_residual --epochs $EPOCHS --run_name "ICDv6_C2_Full"

echo -e "\n================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "Check src/experience/icdv6/ for logs and classification reports."
echo "================================================================"
