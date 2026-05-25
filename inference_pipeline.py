"""
2-Stage Inference Pipeline for Technical Drawing View Classification
===================================================================
Stage 1: YOLO (trained on 3 classes: figure, note, table)
         -> detects bounding boxes with class IDs 0/1/2

Stage 2: Post-processing on 'figure' (cls=0) detections
         -> classify into: front_view / side_view / section_view / isometric_view
         using pixel-based heuristics and optional OCR

Usage:
    python inference_pipeline.py --weights runs/best.pt --source path/to/image.jpg
    python inference_pipeline.py --weights runs/best.pt --source path/to/images/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

# ─── Stage-2 Sub-class labels ─────────────────────────────────────────────────
YOLO_CLASSES   = ['figure', 'note', 'table']
FIGURE_CLASSES = ['front_view', 'side_view', 'section_view', 'isometric_view']

# Colors (BGR) for visualization
PALETTE = {
    'front_view'    : (255,  80,  80),
    'side_view'     : (  0, 165, 255),
    'section_view'  : (  0, 215, 255),
    'isometric_view': (  0,   0, 220),
    'note'          : (200,   0, 200),
    'table'         : (  0, 200,  50),
}

# ─── Singleton EasyOCR reader ─────────────────────────────────────────────────
_OCR_READER    = None
_OCR_AVAILABLE = None


def _get_ocr_reader():
    global _OCR_READER, _OCR_AVAILABLE
    if _OCR_AVAILABLE is None:
        try:
            import easyocr
            _OCR_READER    = easyocr.Reader(['en'], gpu=False, verbose=False)
            _OCR_AVAILABLE = True
        except Exception:
            _OCR_AVAILABLE = False
    return _OCR_READER if _OCR_AVAILABLE else None


# ─── Stage-2 helpers ──────────────────────────────────────────────────────────

def is_isometric(crop: np.ndarray) -> bool:
    """
    Detect 3D rendered (isometric/shaded) views using pixel distribution.
    3D renders: low white_ratio (<0.65) + high mid_gray_ratio (>0.25)
    2D drawings: mostly white background (>74%), thin black lines
    """
    if crop is None or crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    if float(np.mean(s)) > 25:          # colored render
        return True
    total = max(1, v.size)
    white    = float(np.sum(v > 240)) / total
    mid_gray = float(np.sum((v > 50) & (v < 210))) / total
    return white < 0.65 and mid_gray > 0.25


def detect_section_by_ocr(crop: np.ndarray) -> bool:
    """Scan bottom 30% of crop for SECTION / A-A keywords."""
    reader = _get_ocr_reader()
    if reader is None:
        return False
    try:
        h = crop.shape[0]
        bottom = crop[int(h * 0.7):, :]
        texts  = reader.readtext(bottom, detail=0)
        joined = " ".join(texts).upper()
        return any(kw in joined for kw in ("SECTION", "SEC ", "A-A", "B-B", "DETAIL"))
    except Exception:
        return False


def classify_figures(crops_with_area: list[tuple]) -> list[str]:
    """
    Given a list of (crop_image, area) for all 'figure' detections in one image,
    return a fine-grained label for each one.

    Layout heuristic:
      1. If isometric pixel stats → 'isometric_view'
      2. If OCR finds section keyword → 'section_view'
      3. Largest remaining → 'front_view'
      4. All others → 'side_view'
    """
    if not crops_with_area:
        return []

    labels   = [None] * len(crops_with_area)
    pending  = []

    # Pass 1: isometric + section
    for i, (crop, area) in enumerate(crops_with_area):
        if is_isometric(crop):
            labels[i] = 'isometric_view'
        elif detect_section_by_ocr(crop):
            labels[i] = 'section_view'
        else:
            pending.append(i)

    # Pass 2: among remaining, largest → front_view, rest → side_view
    if pending:
        areas    = [crops_with_area[i][1] for i in pending]
        max_idx  = pending[int(np.argmax(areas))]
        for i in pending:
            labels[i] = 'front_view' if i == max_idx else 'side_view'

    return labels


# ─── Main inference ───────────────────────────────────────────────────────────

def run(weights: str, source: str, conf: float = 0.25, save_dir: str = "runs/inference"):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    model    = YOLO(weights)
    src_path = Path(source)
    out_dir  = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = (
        list(src_path.glob("*.jpg")) + list(src_path.glob("*.png"))
        if src_path.is_dir() else [src_path]
    )

    print(f"[Stage-1] Running YOLO on {len(img_paths)} image(s)...")

    for img_path in img_paths:
        buf = np.fromfile(str(img_path), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        h, w = img.shape[:2]
        results = model.predict(str(img_path), conf=conf, iou=0.45, verbose=False)
        boxes   = results[0].boxes

        figures = []   # (idx_in_boxes, crop, area)
        draws   = []   # deferred label draws

        for bi, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if cls_id == 0:  # figure → Stage-2
                crop = img[y1:y2, x1:x2]
                area = (x2 - x1) * (y2 - y1)
                figures.append((bi, crop, area, x1, y1, x2, y2))
            else:
                lbl = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else f"cls{cls_id}"
                draws.append((x1, y1, x2, y2, lbl))

        # Stage-2: classify all figures from this image together
        if figures:
            crops_areas = [(f[1], f[2]) for f in figures]
            fine_labels = classify_figures(crops_areas)
            for (bi, crop, area, x1, y1, x2, y2), lbl in zip(figures, fine_labels):
                draws.append((x1, y1, x2, y2, lbl))

        # Draw all boxes
        for (x1, y1, x2, y2, lbl) in draws:
            color = PALETTE.get(lbl, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, lbl, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"  [OK] {img_path.name} -> {out_path}")

    print(f"\n[DONE] Results saved to: {out_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-Stage Technical Drawing Inference")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights file")
    parser.add_argument("--source",  required=True, help="Image file or directory")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-dir", default="runs/inference", help="Output directory")
    args = parser.parse_args()

    run(
        weights  = args.weights,
        source   = args.source,
        conf     = args.conf,
        save_dir = args.save_dir,
    )
