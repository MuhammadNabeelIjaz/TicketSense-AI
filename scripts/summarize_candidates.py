import json
from pathlib import Path

path = Path("reports/ADVANCED_PRIORITY_EXPERIMENT_RESULTS.json")
if path.exists():
    with open(path) as f:
        data = json.load(f)
    print(f"{'Candidate Name':38} | {'Val Acc':7} | {'Macro F1':8} | {'High Prec':9} | {'High Rec':8} | {'High FPs':8}")
    print("-" * 90)
    for k, v in data.items():
        acc = v.get("val_accuracy", 0)
        f1 = v.get("val_macro_f1", 0)
        hp = v.get("high_precision", 0)
        hr = v.get("high_recall", 0)
        fp = v.get("high_false_positives", 0)
        print(f"{k:38} | {acc:.4f}  | {f1:.4f}   | {hp:.4f}    | {hr:.4f}   | {fp}")
