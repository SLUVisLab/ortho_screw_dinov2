import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from evaluate import summarize
from model import CLASSES, load_checkpoint
from train import collate_valid, prepare_data


class Contracts(unittest.TestCase):
    def test_all_invalid_batch(self):
        self.assertIsNone(collate_valid([None, None]))

    def test_non_dual_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.pth'
            torch.save({'classifier.0.weight': torch.zeros(512, 384)}, path)
            with self.assertRaisesRegex(ValueError, 'dual-view'):
                load_checkpoint(path)

    def test_metrics_account_for_imbalance(self):
        matrix = torch.tensor([[18, 6, 14, 2], [10, 98, 2, 0], [15, 11, 22, 12], [15, 3, 7, 5]])
        metrics = summarize(matrix)
        self.assertEqual(metrics['valid_pairs'], 240)
        self.assertEqual(metrics['correct_pairs'], 143)
        self.assertAlmostEqual(metrics['accuracy_percent'], 59.58333333, places=5)
        self.assertAlmostEqual(metrics['balanced_accuracy_percent'], 46.85606061, places=5)

    def test_saved_split_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{'manufacturer': name, 'patient_number': c * 100 + i,
                     'view_position': 'AP', 'relative_file_path': 'placeholder/images/example.png'}
                    for c, name in enumerate(CLASSES) for i in range(10)]
            pd.DataFrame(rows).to_csv(root / 'metadata_summary.csv', index=False)
            config = {'train_patients': [0, 1], 'evaluation_patients': [1, 2]}
            (root / 'split.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                prepare_data(root, split_config=root / 'split.json')


if __name__ == '__main__':
    unittest.main()
