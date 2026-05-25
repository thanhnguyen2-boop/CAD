"""
test_model.py — Test script for viewport-detect.pt
====================================================
Stage 1 : YOLOv10 detects figure / note / table regions
Stage 2 : Pixel heuristics classify 'figure' into:
          front_view / side_view / section_view / isometric_view

Usage:
    # Single image
    python test_model.py --source path/to/image.jpg

    # Folder of images
    python test_model.py --source path/to/folder/

    # Use a different weights file
    python test_model.py --source img.jpg --weights path/to/other.pt

    # Change confidence threshold
    python test_model.py --source img.jpg --conf 0.3

    # Save results to a specific folder
    python test_model.py --source img.jpg --out results/
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = r"C:\Users\thanh.nguyen2\Documents\project-python\cad\models\viewport-detect.pt"

YOLO_CLASSES   = ["figure", "note", "table"]

# Stage-2 fine-grained labels for 'figure'
FIGURE_LABELS  = ["front_view", "side_view", "section_view", "isometric_view"]

# BGR colors for each final label
PALETTE = {
    "front_view"    : (220,  50,  50),
    "side_view"     : (  0, 165, 255),
    "section_view"  : (  0, 210, 210),
    "isometric_view": (  0,   0, 200),
    "note"          : (190,   0, 190),
    "table"         : ( 30, 180,  30),
}

# ─── Stage-2: Pixel-based figure classifier ───────────────────────────────────

_OCR_READER    = None
_OCR_AVAILABLE = None


def _get_ocr():
    global _OCR_READER, _OCR_AVAILABLE
    if _OCR_AVAILABLE is None:
        try:
            import easyocr
            print("[OCR] Initializing EasyOCR (one-time) ...")
            _OCR_READER    = easyocr.Reader(["en"], gpu=False, verbose=False)
            _OCR_AVAILABLE = True
            print("[OCR] Ready.")
        except Exception as e:
            _OCR_AVAILABLE = False
            print(f"[OCR] Skipped ({e})")
    return _OCR_READER if _OCR_AVAILABLE else None


def _is_isometric(crop: np.ndarray) -> bool:
    """
    3D rendered view: low white_ratio (<65%) + high mid-gray (>25%).
    Colored render: high mean saturation (>25).
    2D line drawings: mostly white paper (>74%) + thin black lines (<20% mid-gray).
    """
    if crop is None or crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    if float(np.mean(s)) > 25:
        return True
    total    = max(1, v.size)
    white    = float(np.sum(v > 240)) / total
    mid_gray = float(np.sum((v > 50) & (v < 210))) / total
    return white < 0.65 and mid_gray > 0.25


def _is_section(crop: np.ndarray) -> bool:
    """Scan bottom 30% of crop for SECTION / A-A / DETAIL keywords via OCR."""
    reader = _get_ocr()
    if reader is None:
        return False
    try:
        h      = crop.shape[0]
        bottom = crop[int(h * 0.7):, :]
        texts  = reader.readtext(bottom, detail=0)
        joined = " ".join(texts).upper()
        return any(kw in joined for kw in ("SECTION", "SEC ", "A-A", "B-B", "DETAIL"))
    except Exception:
        return False


def classify_figures(figures: list[tuple]) -> list[str]:
    """
    Args:
        figures: list of (crop_np, area_px)
    Returns:
        list of fine-grained label strings
    """
    if not figures:
        return []

    labels  = [None] * len(figures)
    pending = []

    for i, (crop, _) in enumerate(figures):
        if _is_isometric(crop):
            labels[i] = "isometric_view"
        elif _is_section(crop):
            labels[i] = "section_view"
        else:
            pending.append(i)

    if pending:
        max_i = max(pending, key=lambda i: figures[i][1])
        for i in pending:
            labels[i] = "front_view" if i == max_i else "side_view"

    return labels


# ─── Drawing helper ───────────────────────────────────────────────────────────

def draw_box(img: np.ndarray, x1, y1, x2, y2, label: str, conf: float):
    color = PALETTE.get(label, (128, 128, 128))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ─── Core inference ───────────────────────────────────────────────────────────

def run_inference(model, img_path: Path, conf: float) -> tuple[np.ndarray, list[dict]]:
    """
    Returns:
        annotated image (BGR), list of detection dicts
    """
    buf = np.fromfile(str(img_path), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]

    results = model.predict(str(img_path), conf=conf, iou=0.45, verbose=False)
    boxes   = results[0].boxes

    figures   = []   # (idx, crop, area, x1,y1,x2,y2, conf)
    non_figs  = []   # (label, conf, x1,y1,x2,y2)
    detections = []

    for box in boxes:
        cls_id   = int(box.cls[0])
        score    = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if cls_id == 0:  # figure → Stage-2
            crop = img[y1:y2, x1:x2]
            area = (x2 - x1) * (y2 - y1)
            figures.append((len(figures), crop, area, x1, y1, x2, y2, score))
        else:
            lbl = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else f"cls{cls_id}"
            non_figs.append((lbl, score, x1, y1, x2, y2))

    # Stage-2 classify figures
    if figures:
        crops_areas = [(f[1], f[2]) for f in figures]
        fine_labels = classify_figures(crops_areas)
        for (_, _, _, x1, y1, x2, y2, score), lbl in zip(figures, fine_labels):
            draw_box(img, x1, y1, x2, y2, lbl, score)
            detections.append({"label": lbl, "conf": score,
                                "bbox": [x1, y1, x2, y2]})

    for (lbl, score, x1, y1, x2, y2) in non_figs:
        draw_box(img, x1, y1, x2, y2, lbl, score)
        detections.append({"label": lbl, "conf": score, "bbox": [x1, y1, x2, y2]})

    return img, detections


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test viewport-detect.pt on technical drawing images")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to .pt weights file")
    parser.add_argument("--source",  required=True,
                        help="Image file or directory")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--out",     default="test_results",
                        help="Output directory for annotated images")
    parser.add_argument("--show",    action="store_true",
                        help="Display each result in a window (requires GUI)")
    args = parser.parse_args()

    # Load YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    print(f"Model loaded. nc={model.model.nc} | "
          f"classes={model.names}")

    # Collect images
    src = Path(args.source)
    if src.is_dir():
        img_paths = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) + \
                    sorted(src.glob("*.jpeg"))
    else:
        img_paths = [src]

    if not img_paths:
        sys.exit(f"No images found in: {src}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inference loop
    total_dets = 0
    label_counts: dict[str, int] = {}
    t_start = time.perf_counter()

    print(f"\nRunning inference on {len(img_paths)} image(s) | conf={args.conf}")
    print("-" * 60)

    for img_path in img_paths:
        t0 = time.perf_counter()
        try:
            annotated, dets = run_inference(model, img_path, args.conf)
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_dets += len(dets)

        # Summary per image
        labels_here = [d["label"] for d in dets]
        for lbl in labels_here:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        print(f"  {img_path.name:<50}  {len(dets):>3} det  {elapsed_ms:5.1f}ms"
              f"  [{', '.join(labels_here)}]")

        # Save annotated image
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        # Optional: display
        if args.show:
            cv2.imshow("Result", annotated)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    t_total = time.perf_counter() - t_start

    # Summary
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Images processed : {len(img_paths)}")
    print(f"  Total detections : {total_dets}")
    print(f"  Total time       : {t_total:.2f}s  "
          f"({t_total/len(img_paths)*1000:.1f}ms/img)")
    print(f"\n  Label breakdown:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "#" * cnt
        print(f"    {lbl:<18} {cnt:>4}  {bar}")
    print(f"\n  Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
