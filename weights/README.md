Weights are release assets, not Git objects. From the repository root:

```bash
gh release download handoff-v1 --repo SLUVisLab/ortho_screw_dinov2 --pattern '*.pth' --dir weights
python verify_weights.py
```

The private repository requires access through GitHub. You can also download
the two `.pth` assets from the release page and place them in this directory.

- `dinov2_dual_view.pth`: the trained four-class AP/lateral classifier.
- `dinov2_vits14_pretrain.pth`: the pretrained backbone used to initialize a new training run.
