# Iris Vision - 5-Class Model Comparison Report (+ Layer 1 Gatekeeper)

> Both models are trained and evaluated on the **exact same 5-class unseen test split** of `dataset2_balanced(Updated)` (100% fundus imagery for all 4 ocular disease classes + `non_fundus` OOD class), paired with Layer 1 Binary Gatekeeper (`mobilenet_v2`, size `10.0 MB`, latency `7.71 ms`).

## Overall Metrics

| Metric | MobileNetV2 (5-Class) | ResNet-50 (5-Class) |
| :--- | :---: | :---: |
| Test dataset | dataset2_balanced(Updated) | dataset2_balanced(Updated) |
| Test samples (5-Class) | 898 | 898 |
| Backbone | `mobilenet_v2` | `resnet50` |
| Best epoch | 13 | 14 |
| **5-Class Test Accuracy** | **93.65%** | **94.43%** |
| **4-Class Ocular Accuracy (Renorm)** | **91.83%** | **92.84%** |
| Macro Precision (5-Class) | 94.04% | 94.70% |
| Macro Recall (5-Class) | 94.17% | 94.90% |
| **Macro F1 (5-Class)** | **94.01%** | **94.76%** |
| Weighted F1 (5-Class) | 93.73% | 94.49% |
| F1 `cataract` | 1.0000 | 1.0000 |
| F1 `diabetic_retinopathy` | 0.9572 | 0.9612 |
| F1 `glaucoma` | 0.8690 | 0.8896 |
| F1 `non_fundus` | 1.0000 | 1.0000 |
| F1 `normal` | 0.8744 | 0.8872 |
| Layer-2 OOD Rejection Rate (%) | 100.00% | 100.00% |
| Layer-2 OOD FAR (%) | 0.00% | 0.00% |
| **2-Layer Combined OOD Rejection (%)** | **100.00%** | **100.00%** |
| **2-Layer Combined OOD FAR (%)** | **0.00%** | **0.00%** |
| 2-Layer Valid Fundus Pass Rate (%) | 100.00% | 100.00% |
| Parameters (M) | 2.56 | 24.04 |
| Checkpoint size (MB) | 10.0 | 92.0 |
| Model latency (ms/img, batch=1) | 6.11 | 6.58 |
| 2-Layer Pipeline latency (ms/img) | 13.82 | 14.30 |

## Paired Significance Tests (Identical Test Set)

### 1. Full 5-Class Test Set (n=898)
Exact McNemar test between **MobileNetV2 (5-Class)** and **ResNet-50 (5-Class)**:

- Only MobileNetV2 (5-Class) correct: **18** images
- Only ResNet-50 (5-Class) correct: **25** images
- p-value: **0.3604** -> difference is **not** statistically significant (p >= 0.05).

### 2. Ocular 4-Class Fundus Subset (n=698)
Exact McNemar test (renormalized 4-class predictions) between **MobileNetV2 (5-Class)** and **ResNet-50 (5-Class)**:

- Only MobileNetV2 (5-Class) correct: **18** images
- Only ResNet-50 (5-Class) correct: **25** images
- p-value: **0.3604** -> difference is **not** statistically significant (p >= 0.05).
