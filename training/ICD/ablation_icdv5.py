import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

def run_experiment(name: str, args: list):
    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU EXPERIMENT: {name}")
    print(f"{'='*50}")
    
    cmd = ["conda", "run", "-n", "MLE", "python", str(BASE_DIR / "training" / "ICD" / "train_ICD_v5.py")] + args
    
    # In case conda is not accessible or we are already in conda env, we can just run python
    # But sticking to conda run to be safe per project specs.
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[✓] Hoàn thành {name}!")
    except subprocess.CalledProcessError as e:
        print(f"\n[✗] Lỗi khi chạy {name}: {e}")
        sys.exit(1)

def main():
    # Cấu hình chung cho tất cả ablation
    common_args = [
        "--hw_profile", "rtx3050",
        "--phase1_epochs", "5",
        "--phase2_epochs", "10",
        "--rdrop_alpha", "1.0",
        "--lambda_kl", "0.5",
        "--lambda_expert", "1.0",
        "--patience", "3"
    ]
    
    # 1. ICDv5_Full
    run_experiment(
        name="ICDv5_Full",
        args=common_args + [
            "--experiment_name", "ICDv5_Full",
            "--run_name", "full_run"
        ]
    )
    
    # 2. ICDv5_NoRouterSup
    run_experiment(
        name="ICDv5_NoRouterSup",
        args=common_args + [
            "--experiment_name", "ICDv5_NoRouterSup",
            "--run_name", "no_router_sup_run",
            "--no_router_sup"
        ]
    )
    
    # 3. ICDv5_NoRouter
    run_experiment(
        name="ICDv5_NoRouter",
        args=common_args + [
            "--experiment_name", "ICDv5_NoRouter",
            "--run_name", "no_router_run",
            "--no_router"
        ]
    )
    
    print("\n" + "="*50)
    print("HOÀN THÀNH TOÀN BỘ QUY TRÌNH ABLATION STUDY CHO ICDV5")
    print("Vui lòng check MLflow UI hoặc file test_metrics để xem kết quả so sánh.")

if __name__ == "__main__":
    main()
