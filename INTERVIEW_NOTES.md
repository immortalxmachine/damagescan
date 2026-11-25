# DamageScan — Interview Notes

Twenty questions an interviewer is likely to ask about this project, answered
against the actual code. File references point to where each concept lives.
Concrete metrics (accuracy, precision, recall, the model comparison) are in
[`metrics.json`](metrics.json) and the README — always quote those, not memory.

---

### 1. What problem does DamageScan solve, end to end?
It classifies a parcel photo as **intact** or **damaged**. A photo is uploaded
to a FastAPI endpoint (`app/api.py`), preprocessed exactly as during training
(`src/damagescan/data.py:preprocess_image`), run through a CNN exported to ONNX,
and returned as a label with a confidence score. The training side is a
standard supervised pipeline: download → split → train → evaluate → explain →
export → serve.

### 2. Why transfer learning instead of training a CNN from scratch?
A ResNet-18 pretrained on ImageNet has already learned general visual features
(edges, textures, corners) from 1.2M images. Parcel damage is mostly about
those same low-level cues (torn edges, dark dents), so we reuse the backbone and
only train a new 2-unit classification head (`src/damagescan/model.py:build_model`).
This needs far less data and far less compute, and generalises better on a small
dataset. From-scratch training would overfit badly on a few thousand images.

### 3. Walk me through the training loop.
`src/damagescan/train.py:run_epoch` is written out explicitly: for each batch we
do a forward pass, compute cross-entropy loss, then (in training) `zero_grad()`,
`loss.backward()`, `optimizer.step()`. `train()` runs this per epoch for train
and validation, steps a `ReduceLROnPlateau` scheduler on val loss, checkpoints
the best model by validation macro-F1, and applies early stopping.

### 4. What is early stopping and why use it?
We stop training once validation F1 hasn't improved for `patience` epochs
(`train.py`, `epochs_without_gain`). It prevents overfitting (the model starts
memorising training data once val performance plateaus) and saves time. We keep
the *best* checkpoint, not the last one.

### 5. What is a learning-rate scheduler doing here?
`ReduceLROnPlateau` halves the learning rate when validation loss stops falling
(`train.py`). Early epochs make big updates to fit quickly; later, smaller steps
fine-tune without overshooting the minimum.

### 6. What data augmentation did you use, and why?
In `data.py:build_transforms` (train branch): `RandomResizedCrop`, horizontal
flip, ±15° rotation, and colour jitter. Augmentation shows the model varied
views of each image so it learns damage cues rather than memorising exact
pixels — a cheap, effective regulariser. **Evaluation uses no augmentation**
(deterministic resize + centre-crop) so results are reproducible.

### 7. How do you prevent train/serve skew?
There is one transform pipeline. `preprocess_image` (used by the API and Gradio)
calls the same `build_transforms(train=False)` used in evaluation, with the same
ImageNet normalisation constants from `config.py`. A model trained with one
normalisation but served with another silently degrades; sharing the code
removes that risk.

### 8. How is the dataset split, and what does "stratified" mean?
`scripts/make_splits.py` shuffles each class separately and splits 70/15/15, so
every split keeps the same intact/damaged proportion as the whole dataset.
Without stratification a random split could, by chance, under-represent damaged
parcels in the test set and give a misleading score. The split is seeded, so it
is reproducible.

### 9. What is overfitting and how did you check for it?
Overfitting is when the model fits training noise and fails to generalise. We
watch the **gap between train and validation metrics** across epochs (logged in
`models/<arch>_history.json`): a rising train F1 with falling val F1 is the
signal. Early stopping, augmentation and weight decay (`Adam(weight_decay=...)`)
are our countermeasures.

### 10. Accuracy vs precision vs recall vs F1 — define them for this task.
With "damaged" as the positive class (`config.POSITIVE_CLASS`):
- **Accuracy** = fraction of all parcels classified correctly.
- **Precision (damaged)** = of parcels we flagged damaged, how many really were.
- **Recall (damaged)** = of truly damaged parcels, how many we caught.
- **F1** = harmonic mean of precision and recall.
Computed in `evaluate.py` via scikit-learn with `pos_label` set to the damaged
index.

### 11. Precision or recall — which matters more for damage detection?
Usually **recall on "damaged"**: missing a damaged parcel (false negative) means
a broken item ships to a customer, which is costly. We'd rather over-flag
(lower precision, a human re-checks) than miss damage. In a real deployment
you'd tune the decision threshold to hit a target recall. (See the actual
numbers in `metrics.json`.)

### 12. Why accuracy can be misleading here?
If 90% of parcels are intact, a model that always predicts "intact" scores 90%
accuracy while catching zero damage. That's why we report per-class precision
and recall and inspect the **confusion matrix** (`reports/confusion_matrix_*.png`),
not accuracy alone.

### 13. What does the confusion matrix tell you?
It's a 2×2 table of true vs predicted labels (`evaluate.py`). The off-diagonal
cells are the errors: false negatives (damaged predicted intact — the dangerous
ones) and false positives. It shows *where* the model fails, not just how often.

### 14. How does Grad-CAM work? Explain it mechanically.
`src/damagescan/gradcam.py`. We hook the last conv block to capture its feature
maps **A** (forward hook) and the gradients of the class score w.r.t. **A**
(backward hook). The importance of feature map *k* is the global average of its
gradients (α_k). We take `ReLU(Σ_k α_k · A_k)`, upsample to the image size and
overlay it. Bright regions are the pixels that most increased the score for the
predicted class — so we can check the model looks at the actual damage, not the
background.

### 15. Why hook the *last* convolutional layer for Grad-CAM?
It's the last place spatial structure survives (e.g. 7×7×512 in ResNet-18)
before global pooling collapses it to a vector. Deeper = more semantic; still
spatial = we can map importance back onto the image. `model.py:get_gradcam_target_layer`
returns `layer4[-1]` for ResNet and `features[-1]` for MobileNet.

### 16. Why export to ONNX? What does it buy you?
ONNX is a framework-neutral model format (`export_onnx.py`). Serving with
`onnxruntime` means the API container doesn't need PyTorch at inference time
(smaller image, fewer deps), and the model becomes portable to other runtimes
and hardware. We verify the ONNX output matches PyTorch within 1e-3 before
trusting it.

### 17. Why two architectures (ResNet-18 vs MobileNetV3)?
To show a size/accuracy trade-off. ResNet-18 is a strong, simple baseline;
MobileNetV3-Small is much lighter and faster, suited to edge devices like a
camera at a sorting belt. Both are built from the same `build_model` function
and compared in `metrics.json`.

### 18. How is the serving layer structured and tested?
`app/api.py` exposes `GET /health` and `POST /predict`. The model is resolved
through a FastAPI dependency (`get_predictor`) so tests inject a tiny ONNX model
(`tests/conftest.py:TinyNet`) instead of a trained one. `tests/test_api.py`
checks the health endpoint, the prediction schema, and that a non-image upload
returns HTTP 400. Inference runs in a threadpool so it doesn't block the event
loop.

### 19. How would this run in a real FedEx sorting facility?
A fixed camera photographs each parcel on the belt; frames hit the `/predict`
endpoint (or an edge-deployed ONNX/MobileNet model for low latency). Damaged
predictions above a tuned threshold divert the parcel to a manual inspection
lane and log an image for audit. You'd monitor recall over time, retrain on
newly labelled data, and watch for data drift (new packaging, lighting).

### 20. How would you improve the model and the system?
More and more varied real data; address class imbalance (class weights / focal
loss / oversampling); threshold tuning for a recall target; test-time
augmentation; a larger backbone if latency allows; calibration of confidence
scores; and a feedback loop that captures human corrections from the inspection
lane to retrain. On the engineering side: a model registry, versioned datasets,
and monitoring for drift.

---

**Where things live (quick map)**
| Concept | File |
|---|---|
| Config, class order, normalisation | `src/damagescan/config.py` |
| Transforms, split-time dataloaders, serve preprocessing | `src/damagescan/data.py` |
| Model building, Grad-CAM target layer, checkpoint I/O | `src/damagescan/model.py` |
| Training loop, early stopping, scheduler | `src/damagescan/train.py` |
| Metrics, confusion matrix | `src/damagescan/evaluate.py` |
| Grad-CAM | `src/damagescan/gradcam.py` |
| ONNX export + verification | `src/damagescan/export_onnx.py` |
| Shared inference | `src/damagescan/inference.py` |
| FastAPI service | `app/api.py` |
| Gradio demo | `app/gradio_app.py` |
| Data download / split | `scripts/download_data.py`, `scripts/make_splits.py` |
