---
description: Long-term Git commit, push, and repository hygiene rules for IrisVision AI.
---

# IrisVision AI — Git Commit & Push Rules

Always follow these rules whenever staging, committing, or pushing changes in this repository:

1. **Conventional Commits Required**:
   - Format: `<type>(<scope>): <concise imperative summary <= 72 chars>` with a bulleted body detailing key changes and verification results.
   - Allowed types: `feat`, `fix`, `test`, `perf`, `refactor`, `docs`, `chore`.
   - Allowed scopes: `ood`, `model`, `data`, `api`, `qa`, `git`, `deps`.

2. **Strict Repository Hygiene (`.gitignore` Enforcement)**:
   - NEVER commit `DATASETS/`, `.venv/`, `.env`, `*.pth`, `*.pt`, `*.onnx`, `*.png` output plots, or `outputs/*_5class/` / `outputs/*_dataset/` archive folders.
   - ALWAYS keep `models/model_config.json`, `models/gatekeeper_config.json`, `outputs/robustness/*.csv`, `outputs/model_comparison_report.md`, and source/test files tracked.
   - Before `git add`, check `git status` and update `.gitignore` if any new large generated artifact folder appears.

3. **Pre-Commit Verification Gate**:
   - Run `.\.venv\Scripts\pytest.exe tests/ -v` before committing code or API changes and confirm all tests pass.

4. **Safe Push Policy**:
   - Never force-push (`git push --force`) to `main`.
   - If `origin/main` is ahead, run `git pull --rebase origin main` before pushing.
   - Always report the exact commit hash, branch, summary of pushed files, and full commit message to the user after pushing.
