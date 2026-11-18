# DINOv2 source and pretrained weights

`vendor/dinov2/` contains the architecture code needed to load the DINOv2 ViT-S/14 backbone. It was recovered from the project's Torch Hub cache on gpu3 on November 18, 2026. It retains Meta's copyright notices and [Apache 2.0 license](vendor/dinov2/LICENSE).

Upstream project: https://github.com/facebookresearch/dinov2

The original upstream commit ID of the cache was not recorded. The new repository commit pins the delivered source files. The pretrained weight file and trained classifier are pinned by SHA256 in `weights_manifest.json`.

The project-specific code was consolidated from the existing local dual-view training script and dataset loader. Explicit seeding, device selection, unique run directories, checkpoint metadata, evaluation reporting, and strict validation were added for the handoff. Crop sampling and model tensor layout preserve the original behavior.
