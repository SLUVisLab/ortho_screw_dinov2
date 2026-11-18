# Orthopedic screw classification with dual-view DINOv2

This repository hands over a trained **dual-view DINOv2 manufacturer classifier** and code to train, evaluate, and use it. It takes an AP screw crop and a lateral screw crop from the same patient. Austin manages the separate data handoff; no images, masks, metadata rows, patient identifiers, or retrieval index are included here.

The supplied checkpoint achieved **59.58% crop-pair accuracy (143/240)** in a fresh evaluation on November 18, 2026. This is a diagnostic on the reconstructed evaluation split, not an independent final test. See [the evaluation report](reports/checkpoint_evaluation.json).

## Get the code and weights

```bash
gh repo clone SLUVisLab/ortho_screw_dinov2
cd ortho_screw_dinov2

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

gh release download handoff-v1 --repo SLUVisLab/ortho_screw_dinov2 \
  --pattern '*.pth' --dir weights
python verify_weights.py
```

Access to this private repository is required. Alternatively, download the two weight files from the [release page](https://github.com/SLUVisLab/ortho_screw_dinov2/releases/tag/handoff-v1) and place them in `weights/`.

The release contains:

| File | Purpose |
| --- | --- |
| `dinov2_dual_view.pth` | The trained four-class dual-view model |
| `dinov2_vits14_pretrain.pth` | Pretrained DINOv2 backbone initialization for a new training run |

Checksums and file sizes are in [weights_manifest.json](weights_manifest.json). The architecture source is vendored, so loading these files does not fetch changing code or weights from Torch Hub.

Verification used Python 3.10.17, PyTorch 2.9.1, torchvision 0.24.1, and Apple MPS. For a CUDA server, install the corresponding CUDA-enabled PyTorch build. `--device auto` selects CUDA, then MPS, then CPU. Missing optional xFormers warnings are harmless for this workflow.

## Data expected from the handoff

Pass the directory containing `metadata_summary.csv` as `--data-dir`. The CSV must contain:

| Column | Meaning |
| --- | --- |
| `manufacturer` | Manufacturer label using the class names below |
| `patient_number` | Patient grouping identifier; all rows for one patient must share a label |
| `view_position` | `AP` or `LATERAL` |
| `relative_file_path` | Image path relative to the data directory |

Images are arranged under `manufacturer/patient/images/`. Their masks are under the matching `manufacturer/patient/masks/` directory. A mask has the same filename with `_mask` inserted before the extension. Preserve the CSV row order when reproducing the historical split.

The loader samples AP and lateral images from the same patient, finds screw mask contours larger than 100 pixels in area, and independently selects one screw in each view. It extracts a context crop twice the largest bounding-box dimension, with a small center jitter and replicated border padding. It does not establish anatomical correspondence between individual screws across views. Patients without a valid pair are skipped and counted in evaluation.

## Evaluate the supplied model

```bash
python evaluate.py --data-dir /absolute/path/to/transferred_data \
  --checkpoint weights/dinov2_dual_view.pth \
  --device auto --seed 42 --workers 4 --batch-size 32 \
  --output runs/checkpoint_evaluation.json
```

This reproduces the reported protocol: a patient-disjoint 80/20 split with `random_state=42`, 10 sampled pairs per evaluation patient, and the supplied checkpoint. The data used for verification had 104 training patients and 26 evaluation patients. Of 260 nominal evaluation pairs, 240 were valid and 20 were skipped.

The JSON includes overall accuracy, balanced accuracy, per-class recall and support, the confusion matrix, skipped counts, sampling settings, and SHA256 hashes of the model and metadata. Crop sampling is stochastic; keep the seed, worker count, software versions, metadata ordering, and files fixed for repeatability. Hardware differences can also change numerical results.

For a newly trained checkpoint, use its saved patient split:

```bash
python evaluate.py --data-dir /absolute/path/to/transferred_data \
  --checkpoint runs/new_training/best_dual_view.pth \
  --split-config runs/new_training/run_config.json \
  --output runs/new_training/reevaluation.json
```

The evaluator verifies that the split configuration belongs to the checkpoint. This is another evaluation of the selection set, not a new final test.

## Train a new model

```bash
python train.py --data-dir /absolute/path/to/transferred_data \
  --output-dir runs/new_training \
  --epochs 50 --batch-size 16 --workers 2 --device auto
```

The output directory must not exist. Training starts a new classifier head over the frozen pretrained backbone; it does not resume the supplied classifier. The recipe recovered from the original local source is:

| Setting | Value |
| --- | --- |
| Backbone | Shared pretrained DINOv2 ViT-S/14, frozen |
| View features | 384 dimensions each, concatenated to 768 |
| Classifier | Linear 768→512, ReLU, dropout 0.5, Linear 512→4 |
| Loss | Cross entropy |
| Optimizer | Adam on the classifier head, learning rate `1e-4`, weight decay `1e-4` |
| Training | 50 epochs; batch size 16; inverse-class-frequency patient sampling with replacement |
| Training transform | Resize to 256×256; random 224×224 crop; horizontal and vertical flips |
| Evaluation transform | Resize to 224×224 |
| Both transforms | Three-channel grayscale; ImageNet normalization |

The model's class indices are fixed:

| Index | Manufacturer |
| --- | --- |
| 0 | Depuy expedium (Synthes) |
| 1 | medtronic solera |
| 2 | nuvasive reline |
| 3 | seaspine mariner |

Filtering excludes missing manufacturer labels, `Styker everest (K2M)` and `orthofix firebird`, then drops manufacturers with fewer than 10 patients. The transferred metadata used for verification retained the four classes above.

Training writes `best_dual_view.pth`, `last_dual_view.pth`, `metrics.json`, and `run_config.json`. The best checkpoint is selected by evaluation accuracy. Checkpoints include the full model state, class order, epoch, metrics, and configuration. The configuration includes patient split IDs and should remain with the data, outside Git; `runs/` is ignored.

### Reproduction limits

The supplied checkpoint was recovered from the local `dinov2_dual_view.pth`, with file timestamps of January 26, 2026. The corresponding source saves the final epoch, and the saved backbone exactly matches the pretrained backbone included in the release. A complete original training manifest and random-number state were not retained. The recipe is supported by preserved source, and the released weights were verified directly; an identical retrained checkpoint or score is not guaranteed.

The original split code passes sorted patient labels against patients in first-seen order. The default `--split-mode historical` retains this behavior for comparison. `--split-mode aligned` corrects label alignment and creates a different split. For new generalization claims, establish separate patient-disjoint training, validation, and final test sets rather than treating the checkpoint-selection set as a final test. This small dataset's evaluation is also class imbalanced; use the per-class and balanced metrics alongside overall accuracy.

## Predict from prepared crops

```bash
python predict.py --ap /path/to/ap_context_crop.png \
  --lateral /path/to/lateral_context_crop.png --device cpu
```

Inputs should be screw-centered context crops, approximately twice the screw's largest bounding-box dimension. The script returns four softmax scores. It uses the released checkpoint by default; `--checkpoint` can select another compatible checkpoint. Both views are required, and loading fails explicitly for missing or incompatible weights.

## Verify the implementation

```bash
python -m unittest discover -s tests -v
```

The handoff checks cover strict checkpoint loading, real-pair inference, head optimization with a frozen backbone, save/load equivalence, invalid-weight rejection, empty batches, and data split validation. Aggregate verification results are in `reports/`; no sample images or patient-level records are committed.

Core files: `model.py` defines the model; `dataset.py` makes paired crops; `train.py`, `evaluate.py`, and `predict.py` are the executable entry points. [THIRD_PARTY.md](THIRD_PARTY.md) documents the vendored DINOv2 source.
