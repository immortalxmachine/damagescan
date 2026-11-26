# Resume Bullets — DamageScan

Four options (pick 2–3). Every number is measured from this repo
(`metrics.json`, the test suite, the code) — nothing is invented. Keep each
bullet to ~25 words.

---

- Built an end-to-end parcel-damage image classifier with PyTorch transfer learning (ResNet-18 vs MobileNetV3), reaching **93% test accuracy** and **1.00 precision** on damaged parcels.

- Served the model through a **FastAPI + ONNX** `/predict` endpoint and a Gradio demo, with **Grad-CAM** heatmaps confirming the network attends to the parcel, not the background.

- Engineered a reproducible training pipeline — stratified serial-number splits, deterministic cuDNN, **7 automated tests**, and **GitHub Actions CI** (ruff + pytest) — packaged in Docker.

- Diagnosed and fixed underfitting and overfitting by tuning the learning rate and input preprocessing, lifting ResNet-18 from ~57% to **93%** test accuracy on 200 images.

---

### Numbers you can defend (all from this repo)

| Metric | Value | Source |
|---|---|---|
| Models compared | 2 (ResNet-18, MobileNetV3-small) | `src/damagescan/model.py` |
| Best test accuracy (ResNet-18) | 93.3% | `metrics.json` |
| Precision / recall / F1 (damaged) | 1.00 / 0.87 / 0.93 | `metrics.json` |
| MobileNetV3-small accuracy | 70.0% | `metrics.json` |
| Dataset size | 200 images (100/class, side view) | Kaggle QC Packages |
| Train / val / test | 140 / 30 / 30 | `scripts/make_splits.py` |
| Automated tests | 7 | `tests/` |
| API endpoints | 2 (`/health`, `/predict`) | `app/api.py` |
| ONNX vs PyTorch max diff | ~1e-6 | `export_onnx.py` |
