# Iris Vision — AI Sentinel (Ocular Disease Classification)

Sistem klasifikasi penyakit mata berbasis PyTorch dan FastAPI untuk 4 kelas citra retina fundus (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`), dilengkapi dengan pra-pemrosesan ROI fundus, penajaman kontras LAB CLAHE, deteksi blur (Laplacian variance), *confidence thresholding* 80%, dan visualisasi interpretabilitas **Grad-CAM**.

---

## Struktur Proyek

```text
Project IrisVision/
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI REST API (/health, /api/v1/predict, Web UI /)
├── src/
│   ├── __init__.py
│   ├── config.py               # Path, kelas, parameter CLAHE/ROI, hyperparameter training
│   ├── preprocess.py           # safe_crop_fundus_roi(), CLAHE ruang warna LAB, resize 224x224
│   ├── dataset.py              # Pembersihan duplikat/kebocoran split, balancing, DataLoader RAM
│   ├── audit_dataset.py        # Audit otomatis integritas, MD5, blur, dan EDA visual
│   ├── model.py                # EyeDiseaseClassifier (MobileNetV2, EfficientNet-B0, ResNet-50)
│   ├── train.py                # Pelatihan 2-fase GPU (AMP FP16, weighted label-smoothed loss)
│   ├── evaluate.py             # Evaluasi test set, confusion matrix, kurva, Grad-CAM
│   └── compare_models.py       # Komparasi 3-model & uji signifikansi McNemar
├── tests/
│   └── test_api.py             # Pengujian endpoint FastAPI pada 4 kelas & validasi error
├── models/
│   └── model_config.json       # Metadata model aktif (file bobot .pth di-gitignore)
├── outputs/
│   ├── model_comparison_report.md
│   └── training_history.json
├── .gitignore
└── requirements.txt
```

---

## Setup Lingkungan (Windows 11 Native + CUDA 12.4)

Gunakan Python 3.12 agar kompatibel dengan PyTorch CUDA `cu124`:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

---

## Perintah Utama

```powershell
# 1. Audit Dataset
.\.venv\Scripts\python.exe src/audit_dataset.py

# 2. Training 2-Fase di GPU (mobilenet_v2 | efficientnet_b0 | resnet50)
.\.venv\Scripts\python.exe src/train.py --backbone resnet50

# 3. Evaluasi Test Set & Visualisasi Grad-CAM
.\.venv\Scripts\python.exe src/evaluate.py

# 4. Komparasi Model & Uji McNemar
.\.venv\Scripts\python.exe src/compare_models.py

# 5. Pengujian Otomatis API FastAPI
.\.venv\Scripts\python.exe tests/test_api.py

# 6. Menjalankan Server FastAPI + Dashboard Web
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000
```

---

## Ringkasan Performa Model (`Retinal Fundus Images`, `n=963`)

| Model | Test Accuracy | Macro F1 | Ukuran Model | Latensi GPU (`batch=1`) |
| :--- | :---: | :---: | :---: | :---: |
| **MobileNetV2** | `97.09%` | **`95.20%`** | **`10.0 MB`** | **`5.18 ms`** |
| **ResNet-50** | **`97.20%`** | `95.16%` | `92.0 MB` | `5.55 ms` |
