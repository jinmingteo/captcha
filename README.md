# CAPTCHA Recognition with PaddleOCR

## Model Selection
We chose **PaddleOCR** because of prior experience with the library and its active maintenance.
Initial testing showed that the text detector performs well, successfully cropping regions of interest.

However, for text recognition, the following issues surfaced:

- **'O'** often predicted as **'0'**
- **'Q'** often predicted as **'0'**
- **'7'** often predicted as **'Z'**
- **'Z'** often predicted as **'2'**

### default v5 config 
```
GT: ['O1R7Q'], Pred: ['01R7Q'] for output05.txt
GT: ['6O5W1'], Pred: ['605W1'] for output02.txt
GT: ['WGST7'], Pred: ['WGSTZ'] for output14.txt
GT: ['OAH0V'], Pred: ['OAHOV'] for output18.txt
GT: ['Z97ME'], Pred: ['297ME'] for output20.txt
Character Error Rate (CER): 4.17%
Accuracy: 80% 
```

### text_recognition_model_name: "en_PP-OCRv5_mobile_rec"
```
GT: ['O1R7Q'], Pred: ['01R7Q'] for output05.txt
GT: ['6O5W1'], Pred: ['605W1'] for output02.txt
GT: ['WGST7'], Pred: ['WGSTZ'] for output14.txt
GT: ['OAH0V'], Pred: ['OAHOV'] for output18.txt
GT: ['N9DQS'], Pred: ['N9D0S'] for output08.txt
Character Error Rate (CER): 4.17%
Accuracy : 80% 
```

### text_recognition_model_name: "en_PP-OCRv4_mobile_rec"
```
GT: ['O1R7Q'], Pred: ['01R7Q'] for output05.txt
GT: ['OYTAD'], Pred: ['QYTAD'] for output06.txt
GT: ['6O5W1'], Pred: ['605W1'] for output02.txt
GT: ['WGST7'], Pred: ['WGSTZ'] for output14.txt
GT: ['5I8VE'], Pred: ['518VE'] for output19.txt
GT: ['OAH0V'], Pred: ['OAHOV'] for output18.txt
Character Error Rate (CER): 5%
Accuracy: 76%
```

---

### Baseline Summary

- **CER:** 4.17%
- **Accuracy:** 80%

Both default and mobile recognition models show similar performance.  
We proceed with fine-tuning `en_PP-OCRv5_mobile_rec` due to computational constraints.

---

## Synthetic Captcha Generation
**Observations:**

The captcha generator places bounding boxes in nearly fixed positions.

**Detected bounding boxes (x, y, w, h) across 26 cases:**

```
2 7 51 24
3 8 50 23
3 8 50 23
2 7 51 25
3 8 49 23
3 8 50 23
2 8 51 24
2 8 51 24
4 7 51 25
3 8 49 23
2 7 50 24
3 8 49 24
2 8 50 24
3 8 48 24
2 7 51 25
2 7 51 24
3 8 50 23
3 8 50 23
3 8 50 24
2 8 49 23
4 8 50 24
2 7 50 24
2 8 51 23
4 8 51 23
3 8 50 23
3 8 50 23
```

By taking the most frequent bounding box and applying minor manual adjustments, the 5-letter captcha can be split into individual character images.
### Hard Negative Mining

Since the current model confuses 0 vs O vs Q and 7 vs 2 vs Z, we generate additional synthetic captchas containing these characters to strengthen the recognizer.

- **Total images:** 350 (280 train / 70 validation)
- **30%** of images include at least 3 “hard” characters.


## Model Fine-tuning
We fine-tune the `en_PP-OCRv5_mobile_rec` model for 5 epochs using the default optimal configuration provided by PaddleOCR.

**Final Model Performance**
```
Character Error Rate (CER): 0%
Accuracy: 100%
```

## Training & Export Commands

```bash
python3 tools/train.py -c train/en_PP-OCRv5_mobile_rec.yaml
python3 tools/export_model.py \
  -c train/en_PP-OCRv5_mobile_rec.yaml \
  -o Global.pretrained_model=./output/en_rec_ppocr_v5/best_accuracy.pdparams \
     Global.save_inference_dir=./output/best_accuracy
```
