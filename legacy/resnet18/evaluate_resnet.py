"""Evaluate the legacy dual-view ResNet18 with the same protocol as ../../evaluate.py.

Uses this repository's split, crop sampler, seeding, and metric code, so with the same
seed/workers/batch size the sampled crop pairs are identical to the DINOv2 evaluation;
only the model and its (deterministic) input transform differ.

    python legacy/resnet18/evaluate_resnet.py --data-dir data/hf/data \
        --output runs/resnet18_evaluation.json
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate import summarize
from train import collate_valid, prepare_data, seed_worker
from resnet_model import DualViewClassifier

NORMALIZE = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
PREPROCESSING = {
    # How original_classifier/train.py evaluated during training
    "training": transforms.Compose([transforms.Resize((400, 400)), transforms.CenterCrop((224, 224)),
                                    transforms.Grayscale(3), transforms.ToTensor(), NORMALIZE]),
    # What the HF Space's model_wrapper.py did at inference
    "space": transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3),
                                 transforms.ToTensor(), NORMALIZE]),
}


def evaluate(model, data_dir, transform, args, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    _, dataset, _, test_ids = prepare_data(data_dir, args.split_mode)
    dataset.transform = transform
    dataset.samples_per_patient = args.samples_per_patient
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_valid, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed))
    confusion = torch.zeros(4, 4, dtype=torch.long)
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            ap, lat, labels = batch
            predictions = model(ap.to(device), lat.to(device)).cpu().argmax(1)
            confusion += torch.bincount(labels * 4 + predictions, minlength=16).reshape(4, 4)
    report = summarize(confusion)
    report.update(evaluation_patients=len(test_ids), nominal_pairs=len(dataset),
                  skipped_pairs=len(dataset) - report["valid_pairs"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "weights" / "original_classifier_best.pth")
    parser.add_argument("--preprocessing", choices=["training", "space", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("runs/resnet18_evaluation.json"))
    parser.add_argument("--split-mode", choices=["historical", "aligned"], default="historical")
    parser.add_argument("--samples-per-patient", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new output path")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = DualViewClassifier(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()

    names = ["training", "space"] if args.preprocessing == "both" else [args.preprocessing]
    results = {name: evaluate(model, args.data_dir, PREPROCESSING[name], args, device) for name in names}
    output = {
        "model": "dual-view ResNet18 (legacy)",
        "results_by_preprocessing": results,
        "seed": args.seed, "workers": args.workers, "batch_size": args.batch_size, "device": device,
        "samples_per_patient": args.samples_per_patient, "split_mode": args.split_mode,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256((args.data_dir / "metadata_summary.csv").read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "metric_description": "Crop-pair metrics on the same reconstructed selection split and sampled crops as the DINOv2 report.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for name, r in results.items():
        print(f"{name}: {r['correct_pairs']}/{r['valid_pairs']} = {r['accuracy_percent']:.1f}% "
              f"(balanced {r['balanced_accuracy_percent']:.1f}%, macro-F1 {r['macro_f1_percent']:.1f}%)")


if __name__ == "__main__":
    main()
