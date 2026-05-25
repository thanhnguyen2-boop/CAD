"""
test_pmi.py — Test PMI detector model (pmi.pt)
===============================================
Detect PMI (Product Manufacturing Information) boxes trong ảnh CAD.

Usage:
    # Single image
    python test_pmi.py --source path/to/image.jpg

    # Folder of images
    python test_pmi.py --source path/to/folder/

    # Custom confidence threshold
    python test_pmi.py --source img.jpg --conf 0.3

    # Save to specific folder
    python test_pmi.py --source img.jpg --out results/

    # Display results (requires GUI)
    python test_pmi.py --source img.jpg --show

    # Batch eval against a YOLO labels folder
    python test_pmi.py --source data/cad-drawing-pmi/valid/images --eval data/cad-drawing-pmi/valid/labels
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = r"C:\Users\thanh.nguyen2\Documents\project-python\cad\models\pmi-v2.pt"

CLASS_NAME  = "PMI"
COLOR_PMI   = (0, 200, 255)   # BGR: amber/yellow


# ─── Drawing helper ───────────────────────────────────────────────────────────

def draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             conf: float, color=(0, 200, 255)):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"PMI {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


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
    results  = model.predict(str(img_path), conf=conf, iou=0.45, verbose=False)
    boxes    = results[0].boxes
    detections = []

    for box in (boxes or []):
        score        = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        draw_box(img, x1, y1, x2, y2, score)
        detections.append({"conf": score, "bbox": [x1, y1, x2, y2]})

    # Overlay count
    count_text = f"PMI count: {len(detections)}"
    cv2.putText(img, count_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_PMI, 2)

    return img, detections


# ─── Evaluation (optional) ────────────────────────────────────────────────────

def compute_iou(a: list, b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a_area = (a[2]-a[0]) * (a[3]-a[1])
    b_area = (b[2]-b[0]) * (b[3]-b[1])
    union  = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def evaluate_vs_labels(detections: list[dict], label_file: Path,
                        img_w: int, img_h: int, iou_thr: float = 0.5) -> dict:
    """
    Tính TP / FP / FN bằng cách so với YOLO label file.
    Returns dict: tp, fp, fn, precision, recall
    """
    gt_boxes = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Class 0 = PMI
            if int(parts[0]) != 0:
                continue
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int((xc - bw/2) * img_w)
            y1 = int((yc - bh/2) * img_h)
            x2 = int((xc + bw/2) * img_w)
            y2 = int((yc + bh/2) * img_h)
            gt_boxes.append([x1, y1, x2, y2])

    matched_gt = set()
    tp = fp = 0

    for det in detections:
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(det["bbox"], gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_thr and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn        = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test PMI detector (pmi.pt) on CAD drawing images")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to pmi.pt weights file")
    parser.add_argument("--source",  required=True,
                        help="Image file or directory")
    parser.add_argument("--conf",    type=float, default=0.36,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--out",     default="test_pmi_results",
                        help="Output directory for annotated images")
    parser.add_argument("--show",    action="store_true",
                        help="Display each result (requires GUI)")
    parser.add_argument("--eval",    default=None,
                        help="Path to YOLO labels folder for precision/recall eval")
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"[ERROR] Weights not found: {weights}")

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    print(f"Model loaded | nc={model.model.nc} | classes={model.names}\n")

    # ── Collect images ──────────────────────────────────────────────────────────
    src = Path(args.source)
    if src.is_dir():
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        img_paths = []
        for ext in exts:
            img_paths.extend(sorted(src.glob(ext)))
    else:
        img_paths = [src]

    if not img_paths:
        sys.exit(f"[ERROR] No images found in: {src}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path(args.eval) if args.eval else None

    # ── Inference loop ──────────────────────────────────────────────────────────
    total_det  = 0
    total_tp = total_fp = total_fn = 0
    t_start    = time.perf_counter()

    print(f"Running inference on {len(img_paths)} image(s)  |  conf={args.conf}")
    print("─" * 70)
    header = f"{'Image':<45} {'Dets':>4}  {'ms':>6}"
    if eval_dir:
        header += f"  {'TP':>4}{'FP':>4}{'FN':>4}  {'P':>6}  {'R':>6}"
    print(header)
    print("─" * 70)

    for img_path in img_paths:
        t0 = time.perf_counter()
        try:
            annotated, dets = run_inference(model, img_path, args.conf)
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_det += len(dets)

        row = f"  {img_path.name:<43} {len(dets):>4}  {elapsed_ms:>6.1f}"

        # Optional eval
        if eval_dir:
            lbl_file = eval_dir / (img_path.stem + ".txt")
            h, w = annotated.shape[:2]
            ev = evaluate_vs_labels(dets, lbl_file, w, h)
            total_tp += ev["tp"]; total_fp += ev["fp"]; total_fn += ev["fn"]
            row += (f"  {ev['tp']:>4}{ev['fp']:>4}{ev['fn']:>4}"
                    f"  {ev['precision']:>5.2f}  {ev['recall']:>5.2f}")

        print(row)

        # Save annotated image
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        if args.show:
            # Resize for display if image is too large
            disp = annotated.copy()
            dh, dw = disp.shape[:2]
            max_dim = 1200
            if max(dh, dw) > max_dim:
                scale = max_dim / max(dh, dw)
                disp = cv2.resize(disp, (int(dw*scale), int(dh*scale)))
            cv2.imshow("PMI Detector", disp)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    t_total = time.perf_counter() - t_start

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("─" * 70)
    print(f"\nSummary")
    print(f"  Images processed : {len(img_paths)}")
    print(f"  Total PMI found  : {total_det}")
    print(f"  Avg per image    : {total_det/max(1,len(img_paths)):.1f}")
    print(f"  Total time       : {t_total:.2f}s  "
          f"({t_total/max(1,len(img_paths))*1000:.1f}ms/img)")

    if eval_dir:
        g_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        g_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1  = 2 * g_p * g_r / (g_p + g_r) if (g_p + g_r) > 0 else 0
        print(f"\n  ── Evaluation (IoU@0.5) ──")
        print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
        print(f"  Precision : {g_p:.4f}")
        print(f"  Recall    : {g_r:.4f}")
        print(f"  F1 Score  : {f1:.4f}")

    print(f"\n  Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
