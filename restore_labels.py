"""Restore all original .txt labels from .txt.bak backup files."""
import shutil
from pathlib import Path

DATASET_DIR = r"C:\Users\thanh.nguyen2\Documents\project-python\cad\technical_drawings_v6"

dataset_path = Path(DATASET_DIR)
restored = 0
skipped = 0

for bak_file in dataset_path.rglob("*.txt.bak"):
    original = bak_file.with_suffix("").with_suffix(".txt")
    bak_size = bak_file.stat().st_size
    if bak_size == 0:
        skipped += 1
        continue
    shutil.copy2(bak_file, original)
    restored += 1

print(f"[DONE] Restored {restored} label files from .bak backups.")
print(f"       Skipped {skipped} empty .bak files.")
