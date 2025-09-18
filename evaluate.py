from jiwer import cer
import glob
import os
import json

all_gt = []
all_pred = []
verbose = True

if __name__ == "__main__":
    try:
        gt_files = glob.glob("sampleCaptchas/output/*.txt")
        for gt_file in gt_files:
            with open(gt_file) as f:
                gt = [item.strip() for item in f.readlines()]
            with open(
                os.path.join(
                    "results", os.path.basename(gt_file).replace(".txt", ".json")
                )
            ) as json_file:
                pred = json.load(json_file)

            assert len(pred["rec_texts"]) == len(
                gt
            ), f"Length mismatch {len(pred['rec_texts'])} vs {len(gt)} for {gt_file}"
            if verbose and pred["rec_texts"] != gt:
                print(
                    f"GT: {gt}, Pred: {pred['rec_texts']} for {os.path.basename(gt_file)}"
                )
            all_pred.extend(pred["rec_texts"])
            all_gt.extend(gt)

        print(f"Character Error Rate (CER): {cer(all_gt, all_pred)}")
        overall_acc = sum([1 if p == g else 0 for p, g in zip(all_pred, all_gt)]) / len(
            all_gt
        )
        print(f"Accuracy: {overall_acc*100:.0f}%")
    except FileNotFoundError as e:
        print("Make sure you have run main.py to generate the results first!")
        raise e
