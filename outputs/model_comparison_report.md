# Iris Vision - 3-Way Model Comparison Report

> Each model is evaluated on the **test split of the dataset it was trained on**. The two *New Dataset* models share the exact same test set, so their comparison is a fair architecture comparison. The *Old Dataset* row reflects a different (anterior-segment cataract) test set and is a dataset-level reference, not a like-for-like benchmark.

## Overall Metrics

| Metric | MobileNetV2 (Old Dataset) | MobileNetV2 (New Dataset) | ResNet-50 (New Dataset) |
| :--- | :---: | :---: | :---: |
| Test dataset | dataset2_balanced | Retinal Fundus Images | Retinal Fundus Images |
| Test samples | 416 | 963 | 963 |
| Backbone | `mobilenet_v2` | `mobilenet_v2` | `resnet50` |
| Best epoch | 20 | 12 | 12 |
| **Test Accuracy** | **93.03%** | **97.09%** | **97.20%** |
| Macro Precision | 93.22% | 95.75% | 95.41% |
| Macro Recall | 93.65% | 94.95% | 95.04% |
| **Macro F1** | **93.32%** | **95.20%** | **95.16%** |
| Weighted F1 | 93.05% | 97.07% | 97.17% |
| F1 `cataract` | 1.0000 | 0.9321 | 0.9148 |
| F1 `diabetic_retinopathy` | 0.9832 | 1.0000 | 1.0000 |
| F1 `glaucoma` | 0.8793 | 0.9288 | 0.9267 |
| F1 `normal` | 0.8705 | 0.9471 | 0.9650 |
| Parameters (M) | 2.56 | 2.56 | 24.04 |
| Checkpoint size (MB) | 10.0 | 10.0 | 92.0 |
| GPU latency (ms/img, batch=1) | 5.39 | 5.18 | 5.55 |

## Paired Significance Test (Same Test Set)

Exact McNemar test between **MobileNetV2 (New Dataset)** and **ResNet-50 (New Dataset)** on 963 identical test images:

- Only MobileNetV2 (New Dataset) correct: **6** images
- Only ResNet-50 (New Dataset) correct: **7** images
- p-value: **1.0000** -> difference is **not** statistically significant (p >= 0.05).
