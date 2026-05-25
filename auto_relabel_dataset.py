#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
          AUTOMATED DATASET RELABELING TOOL FOR 2D-TO-3D CAD PIPELINE
================================================================================
Converts a 3-class Roboflow YOLO dataset:
   ['figure'(0), 'note'(1), 'table'(2)]
Into a 6-class fine-grained CAD dataset:
   ['front_view'(0), 'side_view'(1), 'section_view'(2),
    'isometric_view'(3), 'note'(4), 'table'(5)]

Methods (no manual labeling required):
   1. HSV Color/Saturation filter  -> detects colored/shaded 3D isometric views
   2. Layout heuristics (area rank) -> separates front_view vs side_view
   3. Local OCR (easyocr optional)  -> detects SECTION / DETAIL keywords
"""

import os
import sys
import shutil
import cv2
import numpy as np
import yaml
from pathlib import Path

# ─── Class Mappings ───────────────────────────────────────────────────────────
NEW_CLASSES = {
    0: 'front_view',
    1: 'side_view',
    2: 'section_view',
    3: 'isometric_view',
    4: 'note',
    5: 'table',
}
# Old Roboflow classes: 0=figure, 1=note, 2=table
OLD_TO_NEW_SIMPLE = {0: 0, 1: 4, 2: 5}  # fallback when image unreadable

DATASET_DIR = r"C:\Users\thanh.nguyen2\Documents\project-python\cad\technical_drawings_v6"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def imread_safe(image_path: Path):
    """
    Read image safely for long/Unicode paths on Windows.
    cv2.imread() silently returns None for paths > ~260 chars or with Unicode.
    numpy.fromfile() + cv2.imdecode() is the standard workaround.
    """
    try:
        buf = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(str(image_path))


def backup_label(label_path: Path) -> Path:
    """Create a .txt.bak backup of the original label file (only once)."""
    bak = label_path.with_suffix('.txt.bak')
    if not bak.exists():
        shutil.copy2(label_path, bak)
    return bak


def restore_from_backup(label_path: Path) -> bool:
    """Restore label file from .txt.bak if it exists."""
    bak = label_path.with_suffix('.txt.bak')
    if bak.exists():
        shutil.copy2(bak, label_path)
        return True
    return False


# ─── Detection Helpers ────────────────────────────────────────────────────────

def is_isometric_view(crop) -> bool:
    """
    Detect 3D rendered isometric/shaded views using pixel distribution analysis.

    Key insight (confirmed by pixel stats on this dataset):
      - 2D line drawings: >74% pixels are white (plain paper), <20% mid-gray
      - 3D shaded renders: <65% white, >25% mid-gray (smooth surface shading)
      - Colored 3D renders: high color saturation (mean_s > 25)

    The old val_variance threshold (800) fired on ALL 2D drawings because
    hatching and dense linework also creates high brightness variance.
    """
    if crop is None or crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    mean_s = float(np.mean(s))

    # Test 1: Colored 3D render (CAD software colored shading)
    if mean_s > 25:
        return True

    # Test 2: Grayscale 3D shaded render
    # 3D renders have smooth gradient shading → many mid-gray pixels
    # 2D drawings are mostly white paper + thin black lines (bimodal)
    total_px = max(1, v.size)
    white_ratio    = float(np.sum(v > 240)) / total_px   # near-white background
    mid_gray_ratio = float(np.sum((v > 50) & (v < 210))) / total_px  # shaded surface

    # 3D rendered: white_ratio < 0.65 AND mid_gray > 0.25
    if white_ratio < 0.65 and mid_gray_ratio > 0.25:
        return True

    return False


# EasyOCR reader — initialized once and reused across all crops
_OCR_READER = None
_OCR_AVAILABLE = None  # None = not yet checked, True/False after first check

def detect_by_ocr(crop) -> str | None:
    """
    Use EasyOCR to scan the bottom 30% of a figure crop for view-type keywords.
    Returns 'section_view', 'isometric_view', or None.
    The EasyOCR reader is initialized ONCE (singleton) to avoid per-crop model loading.
    """
    global _OCR_READER, _OCR_AVAILABLE

    # First call: try to initialize the reader
    if _OCR_AVAILABLE is None:
        try:
            import easyocr
            print("  [OCR] Initializing EasyOCR reader (one-time)...")
            _OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
            _OCR_AVAILABLE = True
            print("  [OCR] EasyOCR ready.")
        except ImportError:
            _OCR_AVAILABLE = False
            print("  [OCR] easyocr not installed — skipping OCR detection.")
        except Exception as e:
            _OCR_AVAILABLE = False
            print(f"  [OCR] Failed to init EasyOCR: {e} — skipping.")

    if not _OCR_AVAILABLE or _OCR_READER is None:
        return None

    try:
        h = crop.shape[0]
        bottom = crop[int(h * 0.7):, :]
        results = _OCR_READER.readtext(bottom, detail=0)
        text = " ".join(results).upper()
        if "SECTION" in text or "SEC " in text:
            return "section_view"
        if "ISOMETRIC" in text or "ISO " in text:
            return "isometric_view"
        if "DETAIL" in text or "DET " in text:
            return "section_view"
    except Exception:
        pass
    return None


# ─── Core Processing ──────────────────────────────────────────────────────────

def classify_figure(crop, is_largest: bool) -> int:
    """
    Classify a single 'figure' bounding box crop into one of the 4 view classes.
    Priority: isometric_view > section_view (OCR) > front_view (largest) > side_view
    """
    # 1. Isometric detection (color/shading)
    if is_isometric_view(crop):
        return 3  # isometric_view

    # 2. OCR keyword detection
    ocr_result = detect_by_ocr(crop)
    if ocr_result == "section_view":
        return 2
    if ocr_result == "isometric_view":
        return 3

    # 3. Layout heuristic: largest non-isometric figure = front view
    if is_largest:
        return 0  # front_view
    return 1  # side_view


def process_single_drawing(image_path: Path, label_path: Path):
    """
    Re-label one drawing:
      - Reads original labels FIRST (before any writes).
      - Backs up the original .txt before overwriting.
      - Restores backup automatically if any error occurs.
    """
    if not label_path.exists():
        return

    # --- Read and validate original labels ---
    with open(label_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    valid_lines = [l for l in raw_lines if len(l.strip().split()) >= 5]
    if not valid_lines:
        return  # already empty or malformed — skip, do not touch

    # --- Backup original before any modification ---
    backup_label(label_path)

    try:
        # --- Separate figures from notes/tables ---
        img = imread_safe(image_path)
        img_h, img_w = (img.shape[:2] if img is not None else (640, 640))

        figures = []    # list of dicts: {yolo_coords, box_px, area}
        non_figures = []  # (new_class_id, x_c, y_c, w_c, h_c)

        for line in valid_lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            x_c, y_c, w_c, h_c = map(float, parts[1:5])

            if cls_id == 1:
                non_figures.append((4, x_c, y_c, w_c, h_c))  # note -> 4
            elif cls_id == 2:
                non_figures.append((5, x_c, y_c, w_c, h_c))  # table -> 5
            else:
                # cls_id == 0 (figure): compute pixel box
                px = max(0, int((x_c - w_c / 2) * img_w))
                py = max(0, int((y_c - h_c / 2) * img_h))
                pw = min(img_w - px, max(1, int(w_c * img_w)))
                ph = min(img_h - py, max(1, int(h_c * img_h)))
                figures.append({
                    'coords': (x_c, y_c, w_c, h_c),
                    'box':    (px, py, pw, ph),
                    'area':   pw * ph,
                })

        # --- Classify each figure ---
        new_labels = []

        if not figures:
            # No figures at all — only write note/table mappings
            new_labels = non_figures
        else:
            # Sort by area descending so index 0 = largest (likely front view)
            figures.sort(key=lambda d: d['area'], reverse=True)

            for idx, fig in enumerate(figures):
                x_c, y_c, w_c, h_c = fig['coords']
                px, py, pw, ph = fig['box']

                if img is not None and pw > 0 and ph > 0:
                    crop = img[py:py + ph, px:px + pw]
                    new_cls = classify_figure(crop, is_largest=(idx == 0))
                else:
                    # Cannot read image — map figure -> front_view as fallback
                    new_cls = 0 if idx == 0 else 1

                new_labels.append((new_cls, x_c, y_c, w_c, h_c))

            new_labels.extend(non_figures)

        # --- Write re-labeled file ---
        with open(label_path, 'w', encoding='utf-8') as f:
            for item in new_labels:
                f.write(f"{item[0]} {item[1]:.6f} {item[2]:.6f} {item[3]:.6f} {item[4]:.6f}\n")

    except Exception as exc:
        print(f"  [WARN] Error on {image_path.name}: {exc} — restoring original...")
        restore_from_backup(label_path)


# ─── Dataset Runner ───────────────────────────────────────────────────────────

def auto_relabel_dataset(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    print(f"\n[START] Dataset: {dataset_path.resolve()}")

    # Update data.yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        data['nc'] = 6
        data['names'] = [NEW_CLASSES[i] for i in range(6)]
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
        print("[OK] data.yaml updated to 6 classes.")

    # Process each split
    total = 0
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']:
            img_files.extend(img_dir.glob(ext))

        done = 0
        for img_file in img_files:
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                process_single_drawing(img_file, lbl_file)
                done += 1

        total += done
        print(f"  [{split}] processed {done}/{len(img_files)} files.")

    print(f"\n[DONE] Total processed: {total}")
    print("[INFO] Original labels backed up as .txt.bak — delete after verification.")


def restore_all_backups(dataset_dir: str):
    """Rollback all labels to their original .bak copies."""
    restored = 0
    for bak in Path(dataset_dir).rglob("*.txt.bak"):
        if bak.stat().st_size > 0:
            shutil.copy2(bak, bak.with_suffix('').with_suffix('.txt'))
            restored += 1
    print(f"[ROLLBACK] Restored {restored} label files from backups.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--restore" in sys.argv:
        restore_all_backups(DATASET_DIR)
    elif os.path.exists(DATASET_DIR):
        auto_relabel_dataset(DATASET_DIR)
    else:
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
