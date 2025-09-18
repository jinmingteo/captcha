import glob
import os
import json
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split


def get_most_frequent_bbox():
    pred_files = glob.glob("results/*.json")
    all_x, all_y, all_w, all_h = [], [], [], []
    for pred_file in pred_files:
        with open(pred_file) as f:
            res = json.load(f)
            x, y, w, h = res["rec_boxes"][0]
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_h.append(h)

    all_x = np.array(all_x)
    common_x = np.argmax(np.bincount(all_x))

    all_y = np.array(all_y)
    common_y = np.argmax(np.bincount(all_y))

    all_w = np.array(all_w)
    common_w = np.argmax(np.bincount(all_w))

    all_h = np.array(all_h)
    common_h = np.argmax(np.bincount(all_h))

    print("x, y, w, h : ", common_x, common_y, common_w, common_h)
    return common_x, common_y, common_w, common_h


def generate_character_library():
    img_paths, img_gt = [], []

    x, y, w, h = get_most_frequent_bbox()
    gt_files = glob.glob("sampleCaptchas/output/*.txt")
    for gt_file in gt_files:
        with open(gt_file) as f:
            gt = [item.strip() for item in f.readlines()]
            characters = list(gt[0])
        num_slices = len(characters)
        slice_width = (w - x) // num_slices

        # Loop through and save each slice
        slices = []
        for i in range(num_slices):
            start_x = i * slice_width
            end_x = w if i == num_slices - 1 else (i + 1) * slice_width
            img = cv2.imread(gt_file.replace("output", "input").replace(".txt", ".jpg"))
            cropped_img = img[y:h, x:w]
            # manual adjustment for better cropping
            slice_img = cropped_img[:, start_x + 2 : end_x + 1]
            os.makedirs(f"characters_library/", exist_ok=True)
            out_file = os.path.join(
                "characters_library",
                f"{os.path.basename(gt_file).replace(".txt", "")}_{characters[i]}.jpg",
            )
            cv2.imwrite(out_file, slice_img)
            img_paths.append(out_file)
            img_gt.append(characters[i])

    with open("characters_library/gt.txt", "w") as f:
        for p, g in zip(img_paths, img_gt):
            f.write(f"{p}\t{g}\n")

    df = pd.DataFrame({"img_path": img_paths, "gt": img_gt})
    df.to_csv("characters_library/gt.csv", index=False)
    print("Generated characters_library")
    return df

def generate_synthetic_captchas(df):
    generate_num = 350
    negative_mining_chars = ["0", "O", "Q", "7", "2", "Z"]
    syn_img_paths, syn_gt = [], []
    for i in range(generate_num):
        hard_mining = np.random.choice([0, 1], p=[0.7, 0.3])
        if hard_mining:
            hard_characters = df[df["gt"].isin(negative_mining_chars)]
            hard_sampled = hard_characters.sample(3, replace=True)
            normal_sampled = df.sample(2, replace=True)
            sampled = pd.concat([hard_sampled, normal_sampled], ignore_index=True)
        else:
            sampled = df.sample(5, replace=True)

        annotation = "".join(sampled["gt"].tolist())
        out_file = f"synthetic/synth_{annotation}_{i}.jpg"
        os.makedirs("synthetic", exist_ok=True)
        images = [cv2.imread(p) for p in sampled["img_path"]]
        stitched = np.hstack(images)
        cv2.imwrite(out_file, stitched)
        syn_img_paths.append(out_file)
        syn_gt.append(annotation)
    syn_df = pd.DataFrame({"img_path": syn_img_paths, "gt": syn_gt})
    syn_df.to_csv("synthetic/gt.csv", index=False)
    print(f"Generated {generate_num} synthetic images in synthetic/ folder")

    train_df, test_df = train_test_split(syn_df, test_size=0.2, random_state=42)
    train_df.to_csv("synthetic/gt_train.txt", index=False, header=False, sep="\t")
    test_df.to_csv("synthetic/gt_test.txt", index=False, header=False, sep="\t")
    print(len(train_df), len(test_df), "train and test split done")
    return syn_df

if __name__ == "__main__":
    if not os.path.exists("characters_library/gt.csv"):
        df = generate_character_library()
    else:
        print ("characters_library already exists, skipping generation")
        df = pd.read_csv("characters_library/gt.csv")

    if not os.path.exists("synthetic/gt.csv"):
        syn_df = generate_synthetic_captchas(df)
    else:
        print ("synthetic captchas already exists, skipping generation")
        syn_df = pd.read_csv("synthetic/gt.csv")

    print (syn_df.head())
    
    


