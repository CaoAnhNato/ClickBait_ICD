import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run all training scripts for clickbait detection.")
    parser.add_argument("-e",  "--epochs",       type=int,   default=10,  help="Number of training epochs.")
    parser.add_argument("-b",  "--batch-size",   type=int,   default=8,   help="Batch size per device.")
    parser.add_argument("-ga", "--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--models", nargs="+", default=["mBERT", "XLMR", "videberta", "viBERT", "vELECTRA", "phobert_base_v2", "phobert_large"], 
                        help="List of models to run. Default is all models.")
    args = parser.parse_args()

    models = args.models
    scripts = ["Tune_LoRA.py", "Tune_LLRD.py"]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for model in models:
        for script in scripts:
            script_path = os.path.join(base_dir, model, script)
            if not os.path.exists(script_path):
                print(f"Skipping missing script: {script_path}")
                continue

            print(f"\\n{'='*60}")
            print(f"Running {model} with {script}")
            print(f"{'='*60}\\n")
            
            cmd = [
                "python", script_path,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--gradient-accumulation", str(args.gradient_accumulation)
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\\nError occurred while running {model}/{script}: {e}")
                print("Continuing to next model...")

    print("\\nAll training scripts completed!")

if __name__ == "__main__":
    main()
