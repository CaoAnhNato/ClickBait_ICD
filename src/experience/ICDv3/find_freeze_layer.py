import os
import subprocess
import re
import sys

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_script = os.path.join(base_dir, 'training', 'ICD', 'train_ICD_v3.py')
    output_dir = os.path.join(base_dir, 'src', 'experience', 'ICDv3')
    
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'find_freeze_layer_log.txt')
    
    freeze_layers = [4, 6, 10]
    
    results = {}
    best_overall_f1 = 0.0
    best_layer = None
    
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write("=== FIND FREEZE LAYER LOG ===\n")
        f.write("Parameters: Threshold=0.45, Label Smoothing=0.05, R-Drop Alpha=1.0\n\n")
        
    for layer in freeze_layers:
        print(f"\n[*] Training with freeze_layer={layer}...")
        
        cmd = [
            "conda", "run", "-n", "MLE", 
            "python", train_script,
            "--threshold", "0.45",
            "--label_smoothing", "0.05",
            "--rdrop_alpha", "1.0",
            "--freeze_layers", str(layer),
            "--epochs", "20",  # Limit epochs if needed, assuming default or keeping it
            "--hw_profile", "rtx3050"  # Assuming RTX 3050 from earlier conversations
        ]
        
        # Run process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        best_f1 = 0.0
        output_lines = []
        
        # Capture and parse output in real-time
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
            
            # Match best F1
            match = re.search(r'New best F1: (\d+\.\d+)', line)
            if match:
                f1_val = float(match.group(1))
                if f1_val > best_f1:
                    best_f1 = f1_val
                    
        process.wait()
        
        results[layer] = best_f1
        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_layer = layer
            
        # Write to log incrementally
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"Freeze Layer: {layer} - Best Validation F1: {best_f1:.4f}\n")
            
    print(f"\n[*] Grid search completed. Best freeze_layer: {best_layer} (F1: {best_overall_f1:.4f})")
    
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n[*] Grid search completed. Best freeze_layer: {best_layer} (F1: {best_overall_f1:.4f})\n")

if __name__ == "__main__":
    main()
