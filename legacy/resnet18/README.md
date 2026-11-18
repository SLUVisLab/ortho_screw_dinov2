# Legacy: dual-view ResNet18

This folder archives the **dual-view ResNet18** classifier that the OrthoScrew ID Hugging Face Space served before it switched to the DINOv2 model in November 2026. It is kept for comparison and reproducibility; the handed-over model is the DINOv2 model at the repository root.

The original training code is in [SLUVisLab/ortho_screw_classification](https://github.com/SLUVisLab/ortho_screw_classification) (`original_classifier/`). The scripts here re-create it on top of this repository's data split and crop sampler.

## Model

| | |
| --- | --- |
| Backbones | Two ImageNet-pretrained ResNet18s, one per view, fully fine-tuned |
| Features | 512 per view, concatenated to 1024 (also used as the similarity-search embedding) |
| Head | Linear 1024→512, ReLU, dropout 0.5, Linear 512→4 |
| Training | Adam (lr 3.87e-4, weight decay 6.7e-5), StepLR (8 epochs, ×0.1), label smoothing 0.042, Mixup α 0.239, inverse-frequency patient sampling, 100 epochs, batch 128 |
| Training transform | Resize 400, random rotation ±15°, center crop 224, flips, brightness/contrast jitter |
| Evaluation transform | Resize 400, center crop 224 |
| Space inference transform | Resize 224 (no center crop), which did not match training |
| Checkpoint | Best epoch by a random class-balanced subset of the evaluation split |

The class order is the same as the DINOv2 model's (`model.CLASSES` at the repository root).

## Weights

The checkpoint and its similarity index are not in Git. They stay downloadable from the Hugging Face model repository's history at the revision that served them:

```bash
hf download austin-carnahan/orthopedic-screws-model \
  original_classifier_best.pth embeddings.pkl \
  --revision f439fc0fb997b9f2792deb0f7e2c029b7f09171a \
  --local-dir legacy/resnet18/weights
```

Checksums and sizes are in [`weights_manifest.json`](weights_manifest.json). The same checkpoint is also committed in the original repository as `original_classifier/original_classifier_best.pth`. `embeddings.pkl` holds 1024-dim embeddings of uncropped AP/lateral image pairs, which does not match the cropped inputs the model was trained on.

## Results on the same crops as the DINOv2 model

`evaluate_resnet.py` reuses the root `evaluate.py` protocol (historical split, seed 42, 10 sampled pairs per patient, 4 workers, batch 32), so it scores **exactly the same 240 crop pairs** as the DINOv2 report.

| Model | Accuracy | Balanced accuracy | Macro-F1 | DePuy | Medtronic | NuVasive | SeaSpine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 (handed over) | **59.6%** | 46.9% | 46.3% | 45.0% | **89.1%** | **36.7%** | 16.7% |
| ResNet18, training preprocessing | 49.6% | **48.4%** | **49.4%** | 20.0% | 73.6% | 0.0% | **100.0%** |
| ResNet18, Space preprocessing | 51.7% | 48.1% | 47.8% | 15.0% | 80.9% | 0.0% | 96.7% |

Per-manufacturer columns are per-class accuracy (recall). Full numbers are in [`reports/resnet18_evaluation.json`](reports/resnet18_evaluation.json).

DINOv2 has higher overall accuracy, mostly from Medtronic, the largest class. On the class-balanced metrics the two models are within a few points of each other, with ResNet18 slightly ahead. They fail differently: ResNet18 never predicts NuVasive and labels about half of all pairs SeaSpine (which is why its SeaSpine recall is near 100%), while DINOv2 predicts every class and confuses DePuy, NuVasive, and SeaSpine with each other. With 26 evaluation patients, none of these differences is statistically solid.

## Usage

From the repository root, after downloading the data (see the main README) and the weights above:

```bash
# Score the archived checkpoint with both preprocessing variants
python legacy/resnet18/evaluate_resnet.py --data-dir data/hf/data \
  --output runs/resnet18_evaluation.json

# Retrain (defaults reproduce the original recipe; lower batch size / epochs for smaller GPUs)
python legacy/resnet18/train_resnet.py --data-dir data/hf/data \
  --output-dir runs/resnet18 --batch-size 64 --epochs 100
```

Unlike the DINOv2 model, ResNet18 trains both backbones end to end, so a full run is GPU work: at the default batch size of 128, two ResNet18s may not fit on an 11 GB GPU; 64 is a safer starting point. Retraining will not reproduce the archived checkpoint exactly (different random state, and the archived checkpoint was most likely trained before the Optuna-tuned settings were adopted).
