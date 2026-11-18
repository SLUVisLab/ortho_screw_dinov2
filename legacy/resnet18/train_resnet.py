"""Train the legacy dual-view ResNet18 (archived for comparison; not the handed-over model).

Recipe from the original ortho_screw_classification/original_classifier/train.py:
two ImageNet ResNet18 backbones trained end to end, Optuna-tuned Adam + StepLR,
label smoothing, Mixup, inverse-frequency patient sampling, and a class-balanced random
evaluation subset each epoch. Data loading and the split come from this repository.

    python legacy/resnet18/train_resnet.py --data-dir data/hf/data --output-dir runs/resnet18
    python legacy/resnet18/train_resnet.py --data-dir data/hf/data --output-dir runs/resnet18 \
        --batch-size 64 --epochs 50     # e.g. an 11 GB GPU / shorter run
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from evaluate_resnet import PREPROCESSING
from resnet_model import DualViewClassifier
from train import collate_valid, prepare_data, seed_worker
from torchvision import transforms

# Best hyperparameters from the original Optuna search
LR = 0.000387403
WEIGHT_DECAY = 6.7452e-05
STEP_SIZE = 8
GAMMA = 0.1
LABEL_SMOOTHING = 0.042239
MIXUP_ALPHA = 0.239336

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((400, 400)), transforms.RandomRotation(15), transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Grayscale(3), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        smooth = (-log_preds.sum(dim=-1)).mean() * self.eps / output.size(-1)
        return smooth + (1 - self.eps) * torch.nn.functional.nll_loss(log_preds, target)


def evaluate_balanced(model, dataset, device, criterion):
    """Accuracy on a random class-balanced subset of the evaluation pairs (original protocol)."""
    model.eval()
    by_class = {}
    for idx in range(len(dataset)):
        pid = dataset.patient_ids[idx // dataset.samples_per_patient]
        label = dataset.class_to_idx[dataset.patient_groups.get_group(pid).manufacturer.iloc[0]]
        by_class.setdefault(label, []).append(idx)
    per_class = min(len(v) for v in by_class.values())
    chosen = [i for v in by_class.values() for i in np.random.choice(v, per_class, replace=False)]
    loader = DataLoader(Subset(dataset, chosen), batch_size=16, shuffle=False, collate_fn=collate_valid)
    correct = total = 0
    loss_sum = batches = 0.0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            ap, lat, labels = (x.to(device) for x in batch)
            outputs = model(ap, lat)
            loss_sum += criterion(outputs, labels).item()
            batches += 1
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.numel()
    return (100 * correct / total if total else 0.0), (loss_sum / batches if batches else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-samples-per-patient", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["historical", "aligned"], default="historical")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new run directory")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    train, evaluation, train_ids, evaluation_ids = prepare_data(args.data_dir, args.split_mode)
    train.transform = TRAIN_TRANSFORM
    evaluation.transform = PREPROCESSING["training"]
    evaluation.samples_per_patient = args.eval_samples_per_patient

    labels = [train.class_to_idx[train.patient_groups.get_group(p).manufacturer.iloc[0]] for p in train.patient_ids]
    counts = Counter(labels)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler([1 / counts[y] for y in labels], len(labels), replacement=True, generator=generator)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              collate_fn=collate_valid, worker_init_fn=seed_worker, generator=generator)

    model = DualViewClassifier(num_classes=4).to(device)
    criterion = LabelSmoothingCrossEntropy(eps=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=device, train_patients=train_ids.tolist(), evaluation_patients=evaluation_ids.tolist())
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, batches, correct, total = 0.0, 0, 0, 0
        for batch in train_loader:
            if batch is None:
                continue
            ap, lat, y = (x.to(device) for x in batch)
            optimizer.zero_grad(set_to_none=True)
            # Mixup with one permutation + lambda shared by both views
            index = torch.randperm(ap.size(0), device=ap.device)
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            outputs = model(lam * ap + (1 - lam) * ap[index], lam * lat + (1 - lam) * lat[index])
            loss = lam * criterion(outputs, y) + (1 - lam) * criterion(outputs, y[index])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item(); batches += 1
            correct += (outputs.argmax(1) == y).sum().item(); total += y.numel()
        scheduler.step()
        eval_acc, eval_loss = evaluate_balanced(model, evaluation, device, criterion)
        record = dict(epoch=epoch, train_loss=loss_sum / max(batches, 1),
                      train_accuracy=correct / total if total else 0.0,
                      balanced_subset_accuracy=eval_acc / 100, eval_loss=eval_loss)
        history.append(record)
        print(json.dumps(record), flush=True)
        if eval_acc > best:
            best = eval_acc
            torch.save(model.state_dict(), args.output_dir / "best_resnet18.pth")
        torch.save(model.state_dict(), args.output_dir / "last_resnet18.pth")
        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
