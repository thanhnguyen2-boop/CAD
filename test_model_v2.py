"""
test_model_v2.py — Test script for viewport-detect-v2.pt
==========================================================
Stage 1 : YOLOv10 detects 4 region types in a single pass:
    0: Isometric  (kept as-is — model handles it directly)
    1: Table
    2: Text
    3: View       (passed to Stage-2 for sub-classification)

Stage 2 : Pixel heuristics classify 'View' detections into:
    front_view   — largest 2D orthographic projection
    side_view    — smaller 2D orthographic projections
    section_view — contains SECTION / A-A / B-B keywords (via OCR)

Usage:
    # Single image
    python test_model_v2.py --source path/to/image.jpg

    # Folder of images
    python test_model_v2.py --source path/to/folder/

    # Use a different weights file
    python test_model_v2.py --source img.jpg --weights path/to/other.pt

    # Change confidence threshold
    python test_model_v2.py --source img.jpg --conf 0.3

    # Save results to a specific folder
    python test_model_v2.py --source img.jpg --out results/

    # Display each result in a window
    python test_model_v2.py --source img.jpg --show
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = r"C:\Users\thanh.nguyen2\Documents\project-python\cad\models\viewport-detect-v2.pt"

# Must match training order (data.yaml nc / names)
YOLO_CLASSES = ["Isometric", "Table", "Text", "View"]

# Expected model confidence per class (for reference display)
CLASS_CONFIDENCE = {
    "Isometric": 0.9950,
    "Table":     0.9845,
    "Text":      0.8262,
    "View":      0.9807,
}

# Stage-2 fine-grained labels applied to YOLO 'View' detections
VIEW_LABELS = ["front_view", "side_view", "section_view"]

# BGR colors for each final label
PALETTE = {
    # Model-direct labels
    "Isometric"    : (  0,   0, 200),   # dark red  — 3D rendered view
    "Table"        : ( 30, 180,  30),   # green
    "Text"         : (190,   0, 190),   # purple
    # Stage-2 sub-labels (from 'View')
    "front_view"   : (220,  50,  50),   # blue
    "side_view"    : (  0, 165, 255),   # orange
    "section_view" : (  0, 210, 210),   # cyan
}

# ─── Stage-2: OCR-based View sub-classifier ──────────────────────────────────

_OCR_READER    = None
_OCR_AVAILABLE = None


def _get_ocr():
    """Lazy-initialise EasyOCR (English only, CPU). Returns reader or None."""
    global _OCR_READER, _OCR_AVAILABLE
    if _OCR_AVAILABLE is None:
        try:
            import easyocr
            print("[OCR] Initializing EasyOCR (one-time) ...")
            _OCR_READER    = easyocr.Reader(["en"], gpu=False, verbose=False)
            _OCR_AVAILABLE = True
            print("[OCR] Ready.")
        except Exception as exc:
            _OCR_AVAILABLE = False
            print(f"[OCR] Skipped ({exc})")
    return _OCR_READER if _OCR_AVAILABLE else None


def _is_section(crop: np.ndarray) -> bool:
    """Return True if the crop contains SECTION / A-A / B-B title text (OCR)."""
    reader = _get_ocr()
    if reader is None or crop is None or crop.size == 0:
        return False
    try:
        h      = crop.shape[0]
        bottom = crop[int(h * 0.7):, :]   # scan bottom 30% only
        texts  = reader.readtext(bottom, detail=0)
        joined = " ".join(texts).upper()
        return any(kw in joined for kw in ("SECTION", "SEC ", "A-A", "B-B", "DETAIL"))
    except Exception:
        return False


def classify_views(views: list[tuple]) -> list[str]:
    """
    Sub-classify YOLO 'View' detections into front / side / section.

    Args:
        views: list of (crop_np, area_px)  — one entry per 'View' box

    Returns:
        list of fine-grained label strings (same order as input)
    """
    if not views:
        return []

    labels  = [None] * len(views)
    pending = []   # indices that are NOT section_view

    for i, (crop, _) in enumerate(views):
        if _is_section(crop):
            labels[i] = "section_view"
        else:
            pending.append(i)

    if pending:
        # Largest remaining area → front_view; all others → side_view
        max_i = max(pending, key=lambda i: views[i][1])
        for i in pending:
            labels[i] = "front_view" if i == max_i else "side_view"

    return labels


# ─── Drawing helper ───────────────────────────────────────────────────────────

def draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             label: str, conf: float) -> None:
    """Draw bounding box with label and confidence on the image."""
    color = PALETTE.get(label, (128, 128, 128))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    # Background rectangle for text readability
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ─── Core inference ───────────────────────────────────────────────────────────

def run_inference(model, img_path: Path, conf: float) -> tuple[np.ndarray, list[dict]]:
    """
    Stage 1: YOLOv10 inference → detects Isometric / Table / Text / View.
    Stage 2: 'View' boxes are sub-classified into
             front_view / side_view / section_view via OCR heuristic.

    Args:
        model:    Loaded YOLO model instance.
        img_path: Path to the source image.
        conf:     Confidence threshold.

    Returns:
        annotated_img (BGR np.ndarray), list of detection dicts
        Each dict: {"label": str, "conf": float, "bbox": [x1, y1, x2, y2]}
    """
    buf = np.fromfile(str(img_path), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]

    results = model.predict(str(img_path), conf=conf, iou=0.45, verbose=False)
    boxes   = results[0].boxes

    # Separate 'View' boxes (need Stage-2) from all other boxes
    view_boxes  = []   # (crop, area, x1, y1, x2, y2, score)
    other_boxes = []   # (label, score, x1, y1, x2, y2)
    detections: list[dict] = []

    for box in boxes:
        cls_id          = int(box.cls[0])
        score           = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Resolve raw class name
        if cls_id < len(YOLO_CLASSES):
            raw_label = YOLO_CLASSES[cls_id]
        else:
            raw_label = model.names.get(cls_id, f"cls{cls_id}")

        if raw_label == "View":   # → Stage-2
            crop = img[y1:y2, x1:x2]
            area = (x2 - x1) * (y2 - y1)
            view_boxes.append((crop, area, x1, y1, x2, y2, score))
        else:
            other_boxes.append((raw_label, score, x1, y1, x2, y2))

    # ── Stage-2: sub-classify View detections ────────────────────────────────
    if view_boxes:
        crops_areas = [(vb[0], vb[1]) for vb in view_boxes]
        fine_labels = classify_views(crops_areas)
        for (_, _, x1, y1, x2, y2, score), lbl in zip(view_boxes, fine_labels):
            draw_box(img, x1, y1, x2, y2, lbl, score)
            detections.append({"label": lbl, "conf": score,
                                "bbox": [x1, y1, x2, y2]})

    # ── Draw non-View boxes directly ─────────────────────────────────────────
    for (lbl, score, x1, y1, x2, y2) in other_boxes:
        draw_box(img, x1, y1, x2, y2, lbl, score)
        detections.append({"label": lbl, "conf": score,
                            "bbox": [x1, y1, x2, y2]})

    return img, detections


# ─── Summary helpers ──────────────────────────────────────────────────────────

def _bar(count: int, max_count: int, width: int = 20) -> str:
    """Return a simple ASCII bar scaled to max_count."""
    filled = round(count / max(1, max_count) * width)
    return "█" * filled + "░" * (width - filled)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test viewport-detect-v2.pt on technical drawing images.\n"
                    "Stage-1 classes : Isometric | Table | Text | View\n"
                    "Stage-2 (View)  : front_view | side_view | section_view")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to .pt weights file  [default: viewport-detect-v2.pt]")
    parser.add_argument("--source",  required=True,
                        help="Image file or directory of images")
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="Detection confidence threshold  [default: 0.25]")
    parser.add_argument("--out",     default="test_results_v2",
                        help="Output directory for annotated images  [default: test_results_v2]")
    parser.add_argument("--show",    action="store_true",
                        help="Display each result in a window (requires GUI)")
    args = parser.parse_args()

    # ── Load YOLO ────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed.  Run:  pip install ultralytics")

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"Weights file not found: {weights}")

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    nc    = model.model.nc
    print(f"Model loaded. nc={nc} | classes={model.names}")
    print(f"\nClass confidence reference (from training):")
    for cls_name, mAP in CLASS_CONFIDENCE.items():
        bar = "█" * int(mAP * 29)
        print(f"  {cls_name:<12} {mAP:.4f}  {bar}")

    # ── Collect images ───────────────────────────────────────────────────────
    src = Path(args.source)
    if src.is_dir():
        img_paths = (
            sorted(src.glob("*.jpg"))  +
            sorted(src.glob("*.jpeg")) +
            sorted(src.glob("*.png"))  +
            sorted(src.glob("*.bmp"))  +
            sorted(src.glob("*.tif"))  +
            sorted(src.glob("*.tiff"))
        )
    elif src.is_file():
        img_paths = [src]
    else:
        sys.exit(f"Source not found: {src}")

    if not img_paths:
        sys.exit(f"No supported images found in: {src}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference loop ───────────────────────────────────────────────────────
    total_dets: int            = 0
    label_counts: dict[str, int] = {}
    t_start = time.perf_counter()

    print(f"\nRunning inference on {len(img_paths)} image(s) | conf={args.conf}")
    print("-" * 60)

    for img_path in img_paths:
        t0 = time.perf_counter()
        try:
            annotated, dets = run_inference(model, img_path, args.conf)
        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}")
            continue

        elapsed_ms  = (time.perf_counter() - t0) * 1000
        total_dets += len(dets)

        labels_here = [d["label"] for d in dets]
        for lbl in labels_here:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        print(f"  {img_path.name:<50}  {len(dets):>3} det  {elapsed_ms:6.1f}ms"
              f"  [{', '.join(labels_here)}]")

        # Save annotated image
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        # Optional: display window
        if args.show:
            cv2.imshow("Result — viewport-detect-v2", annotated)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    t_total = time.perf_counter() - t_start

    # ── Final summary ────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Model            : viewport-detect-v2.pt")
    print(f"  Images processed : {len(img_paths)}")
    print(f"  Total detections : {total_dets}")
    print(f"  Total time       : {t_total:.2f}s"
          f"  ({t_total / max(1, len(img_paths)) * 1000:.1f}ms/img)")

    if label_counts:
        max_cnt = max(label_counts.values())
        print(f"\n  Label breakdown:")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"    {lbl:<18} {cnt:>4}  {_bar(cnt, max_cnt)}")
    else:
        print("\n  No detections found.")

    print(f"\n  Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
