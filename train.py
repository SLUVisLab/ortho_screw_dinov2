"""Reference reconstruction of the dual-view recipe, not an exact historical rerun.

Based on the local dinov2_dual_view.py and dataset.py. Preserves
the historical split by default; records its patient IDs for future auditing.
New behavior: explicit seed/device/output directory, empty-batch handling,
local backbone initialization, and checkpoint provenance. The included local
checkpoint is supplied separately in weights/; this script trains a new run.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import hashlib
import json
import random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from model import DualViewDINOv2, CLASSES, ROOT
from dataset import ScrewDataset


def collate_valid(batch):
    batch = [x for x in batch if x is not None]
    return torch.utils.data.default_collate(batch) if batch else None


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_transforms(training=False):
    steps = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] if training else [transforms.Resize((224, 224))]
    return transforms.Compose(steps + [transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def prepare_data(root, split_mode="historical", split_config=None):
    df = pd.read_csv(Path(root) / "metadata_summary.csv")
    required = {"manufacturer", "patient_number", "view_position", "relative_file_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(df.columns))}")
    df = df[~df.manufacturer.isin(["No Accession Number Match", "No manufacturer data", "Styker everest (K2M)", "orthofix firebird"])]
    counts = df.groupby("manufacturer").patient_number.nunique()
    df = df[df.manufacturer.isin(counts[counts >= 10].index)]
    if sorted(df.manufacturer.unique()) != CLASSES:
        raise ValueError("Dataset does not contain exactly the four documented classes after filtering.")
    if df.groupby("patient_number").manufacturer.nunique().max() != 1:
        raise ValueError("A patient has multiple manufacturer labels.")
    patients = df.patient_number.unique()
    labels = df.groupby("patient_number").manufacturer.first()
    # Historical source passes sorted group labels against first-seen patient order.
    # Keep that behavior only for comparison; aligned is a different split.
    if split_mode == "aligned":
        labels = labels.reindex(patients)
    if split_config is None:
        train_ids, evaluation_ids = train_test_split(patients, test_size=.2, random_state=42, stratify=labels)
    else:
        config = json.loads(Path(split_config).read_text())
        expected_hash = config.get("metadata_sha256")
        actual_hash = hashlib.sha256((Path(root) / "metadata_summary.csv").read_bytes()).hexdigest()
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError("Metadata differs from the saved split configuration.")
        train_ids = np.asarray(config["train_patients"], dtype=patients.dtype)
        evaluation_ids = np.asarray(config["evaluation_patients"], dtype=patients.dtype)
        if not len(train_ids) or not len(evaluation_ids):
            raise ValueError("Both patient splits must be nonempty.")
        if set(train_ids) & set(evaluation_ids):
            raise ValueError("Training and evaluation patients overlap.")
        if not (set(train_ids) | set(evaluation_ids)) <= set(patients):
            raise ValueError("Saved split contains patients absent from the filtered data.")
    train = ScrewDataset(df[df.patient_number.isin(train_ids)], root, transform=make_transforms(True))
    evaluation = ScrewDataset(df[df.patient_number.isin(evaluation_ids)], root, transform=make_transforms(False), samples_per_patient=10)
    if train.manufacturers != CLASSES or evaluation.manufacturers != CLASSES:
        raise ValueError("Every split must contain all four classes to preserve label indices.")
    return train, evaluation, train_ids, evaluation_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist; prevents overwriting a run.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["historical", "aligned"], default="historical")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new run directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train, evaluation, train_ids, evaluation_ids = prepare_data(args.data_dir, args.split_mode)
    labels = [train.class_to_idx[train.patient_groups.get_group(p).manufacturer.iloc[0]] for p in train.patient_ids]
    counts = Counter(labels)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler([1 / counts[y] for y in labels], len(labels), replacement=True, generator=generator)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_valid, worker_init_fn=seed_worker, generator=generator, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train, sampler=sampler, **loader_args)
    eval_loader = DataLoader(evaluation, shuffle=False, **loader_args)
    model = DualViewDINOv2().to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=device, classes=CLASSES, learning_rate=1e-4, weight_decay=1e-4, dropout=.5,
                  provenance="New training run from the preserved local dual-view recipe",
                  metadata_sha256=hashlib.sha256((args.data_dir / "metadata_summary.csv").read_bytes()).hexdigest(),
                  train_patients=train_ids.tolist(), evaluation_patients=evaluation_ids.tolist(),
                  torch_version=str(torch.__version__))
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum, batches = 0, 0, 0., 0
        for batch in train_loader:
            if batch is None:
                continue
            ap, lat, labels = (x.to(device) for x in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ap, lat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            loss_sum += loss.item()
            batches += 1
        if not total:
            raise RuntimeError("No valid training pairs; verify image and mask paths.")
        model.eval()
        eval_total, eval_correct = 0, 0
        with torch.inference_mode():
            for batch in eval_loader:
                if batch is None:
                    continue
                ap, lat, labels = (x.to(device) for x in batch)
                logits = model(ap, lat)
                eval_total += labels.numel()
                eval_correct += (logits.argmax(1) == labels).sum().item()
        if not eval_total:
            raise RuntimeError("No valid evaluation pairs; verify image and mask paths.")
        record = dict(epoch=epoch, train_loss=loss_sum / batches, train_accuracy=correct / total,
                      selection_accuracy=eval_correct / eval_total, evaluated_pairs=eval_total,
                      skipped_virtual_pairs=len(evaluation) - eval_total)
        history.append(record)
        print(json.dumps(record), flush=True)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint = dict(model_state_dict=state, epoch=epoch, metrics=record, config=config)
        if record["selection_accuracy"] > best:
            best = record["selection_accuracy"]
            torch.save(checkpoint, args.output_dir / "best_dual_view.pth")
        torch.save(checkpoint, args.output_dir / "last_dual_view.pth")
        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
