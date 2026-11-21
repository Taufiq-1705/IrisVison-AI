# Panduan & Aturan Jangka Panjang Git Workflow — IrisVision AI

Dokumen ini menetapkan standar jangka panjang untuk manajemen versi (`git commit`, `git push`, branching, dan hygiene repositori) pada proyek **IrisVision AI (AI Sentinel)**.

---

## 1. Format Pesan Commit (Conventional Commits)

Setiap commit **wajib** mengikuti pola **Conventional Commits**:

```text
<type>(<scope>): <ringkasan singkat maks 72 karakter, imperative mood>

- <detail perubahan 1>
- <detail perubahan 2>

Verified: <ringkasan hasil test / evaluasi>
```

### Daftar `<type>` yang Digunakan
| Type | Kegunaan | Contoh |
| :--- | :--- | :--- |
| `feat` | Penambahan fitur baru, arsitektur model, atau endpoint API | `feat(ood): implement 2-layer defense-in-depth gatekeeper and 5-class classifier` |
| `fix` | Perbaikan bug pada kode, pipeline data, atau API | `fix(preprocess): ensure uniform ROI crop for cataract fundus images` |
| `test` | Penambahan atau pembaruan unit/integration/robustness test | `test(qa): add OOD rejection and white-box test suites` |
| `perf` | Optimasi kecepatan inferensi, VRAM, atau dataloader | `perf(dataset): enable multithreaded RAM preloading` |
| `refactor` | Perapihan struktur kode tanpa mengubah perilaku fungsional | `refactor(model): unify backbone target layer for Grad-CAM` |
| `docs` | Pembaruan dokumentasi (`README.md`, `CONTRIBUTING.md`, laporan) | `docs(git): add long-term git commit and push workflow rules` |
| `chore` | Konfigurasi proyek, `.gitignore`, dependensi (`requirements.txt`) | `chore(git): ignore 5-class training output archives` |

### Daftar `<scope>` Proyek
- `ood` (Gatekeeper, `non_fundus`, Defense-in-Depth)
- `model` (`src/model.py`, `src/train.py`, `src/evaluate.py`, `models/*.json`)
- `data` (`src/dataset.py`, `src/preprocess.py`, `src/audit_dataset.py`, `src/prepare_ood_data.py`)
- `api` (`app/main.py`, `src/image_validator.py`)
- `qa` (`src/robustness_test.py`, `tests/*`, `outputs/robustness/*.csv`)
- `git` / `deps` (`.gitignore`, `CONTRIBUTING.md`, `requirements.txt`)

---

## 2. Aturan File yang Dilacak vs Diabaikan (`.gitignore` Policy)

### Wajib Dilacak Git (Tracked)
- Seluruh source code (`src/`, `app/`, `tests/`).
- Konfigurasi & metadata model ringan (`models/model_config.json`, `models/gatekeeper_config.json`, `models/*/model_config.json`).
- Laporan terstruktur & hasil tabel QA CSV (`outputs/model_comparison_report.md`, `outputs/training_history.json`, `outputs/robustness/*.csv`).
- Dokumentasi & aturan proyek (`README.md`, `CONTRIBUTING.md`, `requirements.txt`, `.gitignore`, `.agents/rules/`).

### Wajib Diabaikan (Ignored — Dilarang Masuk Git)
- **Dataset lokal**: `DATASETS/`, folder citra mentah apa pun.
- **Bobot biner model**: `*.pth`, `*.pt`, `*.onnx`, `*.safetensors`, `*.h5` (terlalu besar untuk Git; gunakan GitHub Releases / Google Drive / Hugging Face bila ingin distribusi bobot).
- **Virtual environment & cache**: `.venv/`, `__pycache__/`, `.pytest_cache/`.
- **Output gambar besar & arsip eksperimen**: `outputs/*.png`, `outputs/*.csv` (kecuali `outputs/robustness/*.csv`), `outputs/*_dataset/`, `outputs/*_5class/`.
- **Rahasia & kredensial**: `.env`, `.env.*`, API keys, token.

---

## 3. Checklist Wajib Sebelum `git commit` & `git push`

1. **Verifikasi Kebersihan File**:
   - Jalankan `git status` dan pastikan tidak ada file biner (`.pth`), dataset (`DATASETS/`), atau `.env` yang ikut ter-stage.
   - Pastikan ukuran file yang akan di-commit tidak ada yang mendekati batas GitHub (`< 10 MB` per file).
2. **Verifikasi Otomatis (Quality Gate)**:
   - Jalankan pengujian otomatis sebelum commit kode/API/model:
     ```powershell
     .\.venv\Scripts\pytest.exe tests/ -v
     ```
   - Pastikan seluruh test **LULUS (`0 failed`)**.
3. **Sinkronisasi Remote yang Aman**:
   - Jalankan `git fetch origin` dan pastikan branch lokal sejajar dengan `origin/main` (gunakan `git pull --rebase origin main` bila ada commit baru di remote).
   - **Dilarang** melakukan `git push --force` ke branch `main`.
4. **Commit Atomik & Terstruktur**:
   - Kelompokkan perubahan yang saling berkaitan dalam satu commit yang jelas (atau pisahkan commit per tema bila mengerjakan beberapa fitur berbeda).

---

## 4. Strategi Branching (Opsional untuk Pengembangan Fitur Besar)

- `main`: Branch stabil yang selalu lulus seluruh test (`pytest tests/ -v`).
- `feat/<nama-fitur>`: Untuk pengembangan eksperimen/fitur besar (misal `feat/ensemble-model`, `feat/docker-deploy`), lalu digabung ke `main` setelah lolos pengujian.
- `fix/<nama-bug>`: Untuk perbaikan bug terisolasi.
