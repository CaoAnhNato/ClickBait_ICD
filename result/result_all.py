import os
import json
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = ["mBERT", "XLMR", "videberta", "viBERT", "vELECTRA"]
    scenarios = ["lora_base", "llrd_base"]

    results = []

    for model in models:
        for scenario in scenarios:
            config_path = os.path.join(base_dir, f"results_{model}", scenario, "config.json")
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    
                    test_metrics = config.get("test_metrics", {})
                    
                    results.append({
                        "Model": model,
                        "Scenario": "LoRA" if scenario == "lora_base" else "LLRD",
                        "Accuracy": test_metrics.get("accuracy", 0.0),
                        "Precision": test_metrics.get("precision", 0.0),
                        "Recall": test_metrics.get("recall", 0.0),
                        "F1-Macro": test_metrics.get("f1-score", 0.0),
                        "F1-Clickbait": test_metrics.get("f1_clickbait", 0.0)
                    })
                except Exception as e:
                    print(f"Error reading {config_path}: {e}")
            else:
                # Add placeholder if not exists yet
                results.append({
                    "Model": model,
                    "Scenario": "LoRA" if scenario == "lora_base" else "LLRD",
                    "Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                    "F1-Macro": None,
                    "F1-Clickbait": None
                })

    if results:
        df = pd.DataFrame(results)
        out_path = os.path.join(base_dir, "result_all.csv")
        df.to_csv(out_path, index=False)
        print(f"Results successfully synthesized into: {out_path}")
        print(df)
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
