"""Evaluate a dual-view checkpoint using the transferred images and masks."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CLASSES, ROOT, load_checkpoint
from train import prepare_data, collate_valid, seed_worker


def summarize(confusion):
    support = confusion.sum(1)
    correct = confusion.diag()
    recalls = [int(c) / int(n) if n else None for c, n in zip(correct, support)]
    supported = [x for x in recalls if x is not None]
    return {
        "valid_pairs": int(support.sum()),
        "correct_pairs": int(correct.sum()),
        "accuracy_percent": 100 * int(correct.sum()) / int(support.sum()),
        "balanced_accuracy_percent": 100 * sum(supported) / len(supported),
        "class_order": CLASSES,
        "per_class": [{"manufacturer": name, "pairs": int(n), "correct": int(c),
                       "recall_percent": 100 * recall if recall is not None else None}
                      for name, n, c, recall in zip(CLASSES, support, correct, recalls)],
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights" / "dinov2_dual_view.pth")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--split-config", type=Path, help="Saved run_config.json for a newly trained checkpoint")
    parser.add_argument("--split-mode", choices=["historical", "aligned"], default="historical")
    parser.add_argument("--samples-per-patient", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if min(args.samples_per_patient, args.batch_size) < 1 or args.workers < 0:
        parser.error("sample and batch counts must be positive; workers must be nonnegative")
    if args.output.exists():
        parser.error("output already exists; choose a new output path")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)
    # A saved split is mandatory for our self-describing new checkpoints, so
    # evaluation cannot silently fall back to the old checkpoint's split.
    checkpoint_info = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "config" in checkpoint_info and args.split_config is None:
        parser.error("for a newly trained checkpoint, pass --split-config RUN/run_config.json")
    if "config" in checkpoint_info:
        split = json.loads(args.split_config.read_text())
        for key in ["train_patients", "evaluation_patients", "metadata_sha256"]:
            if split.get(key) != checkpoint_info["config"].get(key):
                parser.error("split-config does not match the checkpoint")
    del checkpoint_info
    _, dataset, _, test_ids = prepare_data(args.data_dir, args.split_mode, args.split_config)
    dataset.samples_per_patient = args.samples_per_patient
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_valid, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed), pin_memory=device == "cuda")
    confusion = torch.zeros(4, 4, dtype=torch.long)
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            ap, lat, labels = batch
            logits = model(ap.to(device), lat.to(device)).cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced non-finite outputs.")
            predictions = logits.argmax(1)
            confusion += torch.bincount(labels * 4 + predictions, minlength=16).reshape(4, 4)
    if not confusion.sum():
        raise RuntimeError("No valid evaluation pairs; check image and mask paths.")
    report = summarize(confusion)
    report.update(evaluation_patients=len(test_ids), nominal_pairs=len(dataset),
                  skipped_pairs=len(dataset) - report["valid_pairs"], seed=args.seed, workers=args.workers,
                  batch_size=args.batch_size, device=device, samples_per_patient=args.samples_per_patient,
                  split_mode="saved_config" if args.split_config else args.split_mode,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  metadata_sha256=hashlib.sha256((args.data_dir / "metadata_summary.csv").read_bytes()).hexdigest(),
                  torch_version=str(torch.__version__),
                  metric_description="Crop-pair accuracy on a reconstructed or saved selection split; not an independent final test.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
