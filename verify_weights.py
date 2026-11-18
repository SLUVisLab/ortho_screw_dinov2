"""Verify release weights against the recorded SHA256 checksums."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "weights_manifest.json").read_text())
    for item in manifest["assets"]:
        path = ROOT / "weights" / item["filename"]
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise SystemExit(f"Checksum mismatch: {item['filename']}")
        print(f"Verified {item['filename']}")


if __name__ == "__main__":
    main()
