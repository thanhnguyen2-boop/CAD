"""Tạo notebook fine-tune YOLOv10 cho dataset Other (Isometric, Table, Text, View)."""
import json

def cell(cell_type, source, outputs=None):
    base = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        base["outputs"] = outputs or []
        base["execution_count"] = None
    return base

def md(text): return cell("markdown", text)
def code(text): return cell("code", text)

cells = [

# ── 0. Title ──────────────────────────────────────────────────────────────────
md("""# 📐 CAD Drawing Layout Detector — YOLOv10 Fine-Tune on Kaggle
**Task**: Detect layout regions in 2D CAD / mechanical drawings  
**Classes**: `Isometric` · `Table` · `Text` · `View` (4 classes)  
**Strategy**: Fine-tune from **YOLOv10n** pretrained weights (COCO)  
**Dataset**: `cad-drawing-other` — filtered from CAD Drawing v5 (Roboflow yckim)  
**Model**: YOLOv10n — multi-class layout detector  
**Hardware**: Kaggle GPU T4 x2 (recommended)"""),

# ── 1. GPU check ──────────────────────────────────────────────────────────────
md("## 1. Environment Check"),
code("""\
import os, subprocess, platform
print("Python:", platform.python_version())
print(subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))
print("CUDA:", os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
"""),

# ── 2. Install ────────────────────────────────────────────────────────────────
md("## 2. Install Dependencies"),
code("""\
!pip install -q ultralytics roboflow
import ultralytics; ultralytics.checks()
"""),

# ── 3. Dataset ────────────────────────────────────────────────────────────────
md("""## 3. Load Dataset
Upload your **`cad-drawing-other`** folder as a Kaggle Dataset.  
Expected slug: `/kaggle/input/cad-drawing-other`"""),

code("""\
# ─── CONFIG ────────────────────────────────────────────────────────────────────
KAGGLE_DATASET_INPUT = "/kaggle/input/cad-drawing-other"
# ───────────────────────────────────────────────────────────────────────────────

import os, shutil
from pathlib import Path

WORK_DIR = Path("/kaggle/working/dataset")
WORK_DIR.mkdir(parents=True, exist_ok=True)

if os.path.exists(KAGGLE_DATASET_INPUT):
    import zipfile, glob
    zips = glob.glob(f"{KAGGLE_DATASET_INPUT}/*.zip")
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(WORK_DIR)
    else:
        shutil.copytree(KAGGLE_DATASET_INPUT, str(WORK_DIR), dirs_exist_ok=True)
else:
    raise FileNotFoundError(
        "Dataset not found!\\n"
        "Upload cad-drawing-other to Kaggle Datasets at /kaggle/input/cad-drawing-other"
    )

DATA_YAML = str(WORK_DIR / "data.yaml")
print(f"Dataset ready at: {WORK_DIR}")
"""),

# ── 4. data.yaml ──────────────────────────────────────────────────────────────
md("## 4. Verify & Fix data.yaml Paths"),
code("""\
import yaml
from pathlib import Path

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

print("Original config:")
print(yaml.dump(cfg, default_flow_style=False))

dataset_root = Path(DATA_YAML).parent
cfg["train"] = str(dataset_root / "train" / "images")
cfg["val"]   = str(dataset_root / "valid" / "images")
cfg["test"]  = str(dataset_root / "test"  / "images") if (dataset_root/"test").exists() else str(dataset_root / "valid" / "images")
cfg["nc"]    = 4
cfg["names"] = ["Isometric", "Table", "Text", "View"]

with open(DATA_YAML, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print("\\nUpdated config:")
print(yaml.dump(cfg, default_flow_style=False))

for split in ["train", "valid"]:
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"
    if img_dir.exists():
        imgs = list(img_dir.glob("*.*"))
        lbls = [f for f in lbl_dir.glob("*.txt") if f.stat().st_size > 0] if lbl_dir.exists() else []
        print(f"  {split}: {len(imgs)} images | {len(lbls)} non-empty labels")
"""),

# ── 5. EDA ────────────────────────────────────────────────────────────────────
md("## 5. Exploratory Data Analysis"),
code("""\
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2, random

CLASS_NAMES  = ["Isometric", "Table", "Text", "View"]
CLASS_COLORS = ["#3B82F6", "#10B981", "#8B5CF6", "#EF4444"]
NC = 4

dataset_root = Path(DATA_YAML).parent
lbl_dir = dataset_root / "train" / "labels"

counts = [0] * NC
for lbl_file in lbl_dir.glob("*.txt"):
    for line in lbl_file.read_text().splitlines():
        parts = line.split()
        if parts:
            cls = int(parts[0])
            if 0 <= cls < NC:
                counts[cls] += 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dataset Analysis — CAD Drawing Layout Detector (4 classes)", fontsize=14, fontweight="bold")

bars = axes[0].bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor="white", linewidth=1.2)
axes[0].set_title("Class Distribution (train set)")
axes[0].set_ylabel("Number of annotations")
axes[0].tick_params(axis="x", rotation=20)
for bar, cnt in zip(bars, counts):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, str(cnt),
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

non_zero = [(n, c, col) for n, c, col in zip(CLASS_NAMES, counts, CLASS_COLORS) if c > 0]
axes[1].pie([x[1] for x in non_zero], labels=[x[0] for x in non_zero],
            colors=[x[2] for x in non_zero], autopct="%1.1f%%", startangle=140)
axes[1].set_title("Class Distribution (pie)")

plt.tight_layout()
plt.savefig("/kaggle/working/layout_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Total annotations:", sum(counts))
"""),

code("""\
def draw_yolo_boxes(img_path, lbl_path, class_names, colors):
    img = cv2.imread(str(img_path))
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5: continue
            cls = int(parts[0])
            if cls >= len(class_names): continue
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int((xc - bw/2) * w); y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w); y2 = int((yc + bh/2) * h)
            color = tuple(int(c * 255) for c in plt.cm.tab10(cls)[:3])
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, class_names[cls], (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

img_dir = dataset_root / "train" / "images"
lbl_dir = dataset_root / "train" / "labels"
samples = random.sample(list(img_dir.glob("*.jpg")), min(6, len(list(img_dir.glob("*.jpg")))))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Sample Training Images with Ground-Truth Annotations", fontsize=13, fontweight="bold")
for ax, img_path in zip(axes.flatten(), samples):
    lbl_path = lbl_dir / (img_path.stem + ".txt")
    drawn = draw_yolo_boxes(img_path, lbl_path, CLASS_NAMES, CLASS_COLORS)
    if drawn is not None:
        ax.imshow(drawn)
    ax.axis("off")
    ax.set_title(img_path.stem[:30], fontsize=8)

plt.tight_layout()
plt.savefig("/kaggle/working/layout_sample_annotations.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

# ── 5b. Label Validation ──────────────────────────────────────────────────────
md("""## 5b. Label Validation & Auto-Fix
Checks all label files for class IDs >= nc=4. Drops invalid labels."""),
code("""\
from pathlib import Path
import yaml

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)
nc = cfg["nc"]   # 4

dataset_root = Path(DATA_YAML).parent
problems = 0

for split in ["train", "valid"]:
    lbl_dir = dataset_root / split / "labels"
    if not lbl_dir.exists(): continue
    for lbl_file in lbl_dir.glob("*.txt"):
        lines = lbl_file.read_text().splitlines()
        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            cls = int(parts[0])
            if cls >= nc:
                problems += 1
                continue
            valid_lines.append(" ".join(parts))
        lbl_file.write_text("\\n".join(valid_lines) + "\\n")

if problems == 0:
    print(f"[OK] All labels valid (nc={nc}). No fixes needed.")
else:
    print(f"[FIXED] Dropped {problems} invalid class labels.")

names = cfg.get("names", [str(i) for i in range(nc)])
for split in ["train", "valid"]:
    lbl_dir = dataset_root / split / "labels"
    counts = [0] * nc
    for lbl_file in lbl_dir.glob("*.txt"):
        for line in lbl_file.read_text().splitlines():
            parts = line.split()
            if parts:
                cls = int(parts[0])
                if 0 <= cls < nc:
                    counts[cls] += 1
    print(f"\\n[{split}] class distribution:")
    for i, (n, c) in enumerate(zip(names, counts)):
        print(f"  {i}: {n:<12} {c:>5} annotations")
"""),

# ── 6. Training Config ────────────────────────────────────────────────────────
md("""## 6. Training Configuration
**Strategy**: Fine-tune từ **YOLOv10n** pretrained trên COCO — Ultralytics tự download khi chạy  
**4 Classes**: Isometric · Table · Text · View  
**LR**: `lr0=0.01` — phù hợp fine-tune từ pretrained COCO weights"""),

code("""\
TRAIN_CONFIG = {
    "model"          : "yolov10n.pt",  # YOLOv10 Nano — pretrained COCO, auto-download
    "data"           : DATA_YAML,
    "imgsz"          : 640,
    "epochs"         : 100,
    "patience"       : 20,
    "batch"          : 16,
    "workers"        : 4,
    "optimizer"      : "SGD",
    "lr0"            : 0.01,    # standard fine-tune LR from pretrained COCO
    "lrf"            : 0.01,
    "momentum"       : 0.937,
    "weight_decay"   : 0.0005,
    "warmup_epochs"  : 3,
    "warmup_bias_lr" : 0.1,
    "warmup_momentum": 0.8,
    "amp"            : True,
    # Augmentation
    "hsv_h"          : 0.015,
    "hsv_s"          : 0.5,
    "hsv_v"          : 0.3,
    "degrees"        : 0.0,
    "translate"      : 0.1,
    "scale"          : 0.5,
    "shear"          : 0.0,
    "flipud"         : 0.0,
    "fliplr"         : 0.3,
    "mosaic"         : 0.5,
    "mixup"          : 0.0,
    "copy_paste"     : 0.0,
    # Output
    "project"        : "/kaggle/working/runs",
    "name"           : "layout_detector_finetune",
    "exist_ok"       : True,
    "plots"          : True,
    "save"           : True,
    "save_period"    : 10,
    "verbose"        : True,
    "device"         : 0,
}

for k, v in TRAIN_CONFIG.items():
    print(f"  {k:<20}: {v}")
"""),

# ── 7. Train ──────────────────────────────────────────────────────────────────
md("## 7. Training"),
code("""\
from ultralytics import YOLO
import time

print("="*60)
print("  CAD Layout Detector — YOLOv10 Fine-Tune (4 classes)")
print("="*60)

model = YOLO(TRAIN_CONFIG["model"])
print(f"Params: {sum(p.numel() for p in model.model.parameters()):,}")

t0 = time.time()
results = model.train(**TRAIN_CONFIG)

elapsed = time.time() - t0
print(f"\\nCompleted in {elapsed/60:.1f} minutes")
print(f"Best model: {results.save_dir}/weights/best.pt")
"""),

# ── 8. Validate ───────────────────────────────────────────────────────────────
md("## 8. Validation — Per-class Metrics"),
code("""\
from ultralytics import YOLO
from pathlib import Path

best_pt = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights" / "best.pt"
best_model = YOLO(str(best_pt))

metrics = best_model.val(data=DATA_YAML, imgsz=640, batch=16, verbose=True)

print("\\n" + "="*55)
print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
print(f"  Precision    : {metrics.box.mp:.4f}")
print(f"  Recall       : {metrics.box.mr:.4f}")
print("="*55)

class_names = ["Isometric", "Table", "Text", "View"]
print("\\nPer-class AP@0.5:")
for name, ap in zip(class_names, metrics.box.ap50):
    bar = "█" * int(ap * 30)
    print(f"  {name:<12} {ap:.4f}  {bar}")
"""),

# ── 9. Visualize ──────────────────────────────────────────────────────────────
md("## 9. Training Curves & Confusion Matrix"),
code("""\
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

run_dir = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"]

plots = {
    "results.png"                     : "Training Curves (Loss / mAP)",
    "confusion_matrix_normalized.png" : "Normalized Confusion Matrix",
    "PR_curve.png"                    : "Precision-Recall Curve",
    "F1_curve.png"                    : "F1-Confidence Curve",
}

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("CAD Layout Detector — Training Results", fontsize=15, fontweight="bold")

for ax, (fname, title) in zip(axes.flatten(), plots.items()):
    fpath = run_dir / fname
    if fpath.exists():
        img = mpimg.imread(str(fpath))
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, f"Not found:\\n{fname}", ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")

plt.tight_layout()
plt.savefig("/kaggle/working/layout_training_summary.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

# ── 10. Inference Demo ────────────────────────────────────────────────────────
md("## 10. Inference Demo on Validation Images"),
code("""\
import random, cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

best_pt = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights" / "best.pt"
infer_model = YOLO(str(best_pt))

val_imgs = list((Path(DATA_YAML).parent / "valid" / "images").glob("*.jpg"))
samples  = random.sample(val_imgs, min(4, len(val_imgs)))

fig, axes = plt.subplots(1, len(samples), figsize=(6*len(samples), 6))
if len(samples) == 1: axes = [axes]
fig.suptitle("Layout Detection — Inference on Validation Set (conf > 0.25)", fontsize=13, fontweight="bold")

for ax, img_path in zip(axes, samples):
    results_inf = infer_model.predict(str(img_path), conf=0.25, iou=0.5, verbose=False)
    plotted = results_inf[0].plot(line_width=2, font_size=10)
    ax.imshow(plotted[:,:,::-1])
    ax.axis("off")
    boxes = results_inf[0].boxes
    n = len(boxes) if boxes is not None else 0
    ax.set_title(f"{img_path.name[:25]}\\n{n} detections", fontsize=9)

plt.tight_layout()
plt.savefig("/kaggle/working/layout_inference_demo.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

# ── 11. Export ────────────────────────────────────────────────────────────────
md("## 11. Export Model"),
code("""\
from ultralytics import YOLO
from pathlib import Path

best_pt = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights" / "best.pt"
export_model = YOLO(str(best_pt))

onnx_path = export_model.export(format="onnx", imgsz=640, simplify=True, opset=17)
print(f"ONNX exported: {onnx_path}")
"""),

# ── 12. Package ───────────────────────────────────────────────────────────────
md("## 12. Package & Save Artifacts"),
code("""\
import zipfile
from pathlib import Path

run_dir = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"]
out_zip = Path("/kaggle/working/layout_detector.zip")

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for wt in (run_dir / "weights").glob("*.pt"):
        zf.write(wt, f"weights/{wt.name}")
    for onnx_f in run_dir.glob("*.onnx"):
        zf.write(onnx_f, f"weights/{onnx_f.name}")
    for plot_f in run_dir.glob("*.png"):
        zf.write(plot_f, f"plots/{plot_f.name}")
    for meta_f in run_dir.glob("*.yaml"):
        zf.write(meta_f, f"config/{meta_f.name}")
    for csv_f in run_dir.glob("*.csv"):
        zf.write(csv_f, f"logs/{csv_f.name}")

print(f"Artifacts saved: {out_zip}  ({out_zip.stat().st_size/1e6:.1f} MB)")
print("  best.pt   -> Layout detector (Isometric / Table / Text / View)")
print("  *.onnx    -> ONNX for deployment")
"""),

]  # end cells

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "cells": cells,
}

out_path = r"C:\Users\thanh.nguyen2\Documents\project-python\cad\train_layout_detector_yolov10.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook saved: {out_path}")
print(f"  Cells: {len(cells)}")
