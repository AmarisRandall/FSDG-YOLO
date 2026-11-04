import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def check_path(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] {name} not found: {path}")


def load_split_image_paths(data_yaml, split="val"):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = data_cfg.get("path", "")
    split_path = data_cfg.get(split, "")

    if isinstance(split_path, list):
        split_items = split_path
    else:
        split_items = [split_path]

    image_paths = []

    for item in split_items:
        item = str(item)

        if not os.path.isabs(item):
            item = os.path.join(dataset_root, item)

        item = os.path.abspath(item)

        if os.path.isfile(item) and item.endswith(".txt"):
            with open(item, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if not p:
                        continue
                    if not os.path.isabs(p):
                        p = os.path.join(dataset_root, p)
                    image_paths.append(Path(p))

        elif os.path.isdir(item):
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
                image_paths.extend(Path(item).rglob(ext))

        elif os.path.isfile(item):
            image_paths.append(Path(item))

        else:
            raise FileNotFoundError(f"[ERROR] {split} path not found: {item}")

    return sorted(list(set(image_paths)))


def image_path_to_label_path(img_path):
    img_path = Path(img_path)
    parts = list(img_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
    else:
        label_path = img_path.with_suffix(".txt")

    return label_path


def read_yolo_labels(label_path):
    labels = []

    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            labels.append([cls, x1, y1, x2, y2, w, h])

    return labels


def box_iou_norm(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(x2 - x1, 0.0)
    inter_h = max(y2 - y1, 0.0)
    inter_area = inter_w * inter_h

    area1 = max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0)
    area2 = max(box2[2] - box2[0], 0.0) * max(box2[3] - box2[1], 0.0)

    return inter_area / (area1 + area2 - inter_area + 1e-9)


def compute_ap_101(recalls, precisions):
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        if np.any(recalls >= t):
            ap += np.max(precisions[recalls >= t])
        else:
            ap += 0.0
    return ap / 101


def evaluate_small_objects_ap50(
    model,
    data_yaml,
    split="val",
    imgsz=640,
    device="0",
    conf_thres=0.001,
    nms_iou=0.7,
    match_iou=0.5,
    small_size=32,
):
    image_paths = load_split_image_paths(data_yaml, split=split)

    all_predictions = []
    total_small_gt = 0
    small_area = small_size * small_size
    gt_records = {}

    print(f"[INFO] Evaluating small objects on {len(image_paths)} {split} images...")

    for img_id, img_path in enumerate(image_paths):
        label_path = image_path_to_label_path(img_path)

        img = Image.open(img_path)
        img_w, img_h = img.size

        labels = read_yolo_labels(label_path)
        small_gts = []
        ignored_gts = []

        for lb in labels:
            cls, x1, y1, x2, y2, w, h = lb
            area_eval = (w * imgsz) * (h * imgsz)

            gt_item = {
                "cls": cls,
                "box": [x1, y1, x2, y2],
                "matched": False,
            }

            if area_eval < small_area:
                small_gts.append(gt_item)
            else:
                ignored_gts.append(gt_item)

        gt_records[img_id] = {
            "small": small_gts,
            "ignore": ignored_gts,
        }
        total_small_gt += len(small_gts)

        result = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thres,
            iou=nms_iou,
            device=device,
            verbose=False,
        )[0]

        if result.boxes is None:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        cls_arr = result.boxes.cls.cpu().numpy()
        conf_arr = result.boxes.conf.cpu().numpy()

        for box, cls, score in zip(xyxy, cls_arr, conf_arr):
            x1, y1, x2, y2 = box.tolist()
            box_norm = [
                x1 / img_w,
                y1 / img_h,
                x2 / img_w,
                y2 / img_h,
            ]

            all_predictions.append({
                "img_id": img_id,
                "cls": int(cls),
                "score": float(score),
                "box": box_norm,
            })

    if total_small_gt == 0:
        print("[WARN] No small objects found in dataset split.")
        return 0.0, 0.0, 0, 0

    all_predictions = sorted(all_predictions, key=lambda x: x["score"], reverse=True)

    tp = []
    fp = []
    matched_small = 0

    for pred in all_predictions:
        img_id = pred["img_id"]
        pred_cls = pred["cls"]
        pred_box = pred["box"]

        small_gts = gt_records[img_id]["small"]
        ignored_gts = gt_records[img_id]["ignore"]

        ignore_pred = False
        for gt in ignored_gts:
            if gt["cls"] != pred_cls:
                continue

            iou = box_iou_norm(pred_box, gt["box"])
            if iou >= match_iou:
                ignore_pred = True
                break

        if ignore_pred:
            continue

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(small_gts):
            if gt["cls"] != pred_cls:
                continue
            if gt["matched"]:
                continue

            iou = box_iou_norm(pred_box, gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= match_iou and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            small_gts[best_gt_idx]["matched"] = True
            matched_small += 1
        else:
            tp.append(0)
            fp.append(1)

    if len(tp) == 0:
        return 0.0, 0.0, total_small_gt, 0

    tp = np.array(tp)
    fp = np.array(fp)

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / (total_small_gt + 1e-9)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    ap50_small = compute_ap_101(recalls, precisions)
    recall_small = matched_small / total_small_gt

    return ap50_small, recall_small, total_small_gt, matched_small


def print_class_ap(metrics, target_names=("fire", "smoke")):
    names = metrics.names

    if isinstance(names, dict):
        names_dict = names
    else:
        names_dict = {i: n for i, n in enumerate(names)}

    ap_class_index = list(metrics.box.ap_class_index)
    class_result_dict = {}

    print("\n" + "=" * 80)
    print("[CLASS-WISE RESULTS]")

    for i, cls_id in enumerate(ap_class_index):
        cls_id = int(cls_id)
        cls_name = str(names_dict[cls_id])

        precision, recall, ap50, ap5095 = metrics.box.class_result(i)

        class_result_dict[cls_name.lower()] = {
            "precision": precision,
            "recall": recall,
            "ap50": ap50,
            "ap5095": ap5095,
        }

        print(
            f"{cls_name}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"AP50={ap50:.4f}, "
            f"AP50-95={ap5095:.4f}"
        )

    print("-" * 80)

    for target in target_names:
        key = target.lower()
        if key in class_result_dict:
            print(f"AP50-{target}: {class_result_dict[key]['ap50']:.4f}")
        else:
            print(f"[WARN] Class '{target}' not found in data.yaml names.")

    print("=" * 80 + "\n")
    return class_result_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Validate AP metrics for a trained YOLO model.")
    parser.add_argument("--model", default=str(ROOT / "weights/seed15_best.pt"), help="Path to best.pt.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML.")
    parser.add_argument("--split", default="val", help="Dataset split for validation.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics validation plots.")
    parser.add_argument("--skip-small", action="store_true", help="Skip AP50-small evaluation.")
    parser.add_argument("--small-size", type=int, default=32)
    parser.add_argument("--small-conf", type=float, default=0.001)
    parser.add_argument("--small-nms-iou", type=float, default=0.7)
    parser.add_argument("--small-match-iou", type=float, default=0.5)
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args

    check_path(args.model, "MODEL")
    check_path(args.data, "DATA_YAML")

    print("=" * 80)
    print("[INFO] Model:", args.model)
    print("[INFO] Data YAML:", args.data)
    print("[INFO] Split:", args.split)
    print("[INFO] Image size:", args.imgsz)
    print("[INFO] Batch:", args.batch)
    print("[INFO] Device:", args.device)
    print("=" * 80)

    # Register both module paths used by saved checkpoints.
    for old, current in (("FSRD", "DSDM"), ("DGFMv3", "DGFM"), ("DetectP2FireAlign", "FAGM")):
        module = import_module(f"modules.{current}")
        setattr(module, old, getattr(module, current))
        sys.modules[f"ultralytics.nn.modules.{current}"] = module
        sys.modules[f"ultralytics.nn.modules.{old}"] = module
    model = YOLO(args.model)

    val_metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        plots=args.plots,
        project=str(ROOT / "runs/evaluate"), workers=0,
        verbose=True,
    )

    class_results = print_class_ap(val_metrics, target_names=("fire", "smoke"))

    ap50_small = None
    recall_small = None

    if not args.skip_small:
        ap50_small, recall_small, total_small_gt, matched_small = evaluate_small_objects_ap50(
            model=model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            device=args.device,
            conf_thres=args.small_conf,
            nms_iou=args.small_nms_iou,
            match_iou=args.small_match_iou,
            small_size=args.small_size,
        )

        print("\n" + "=" * 80)
        print("[SMALL OBJECT RESULTS]")
        print(f"Small object definition: area < {args.small_size}x{args.small_size} under imgsz={args.imgsz}")
        print(f"Total small GT: {total_small_gt}")
        print(f"Matched small GT @IoU{args.small_match_iou}: {matched_small}")
        print(f"AP50-small: {ap50_small:.4f}")
        print(f"Recall-small@{args.small_match_iou}: {recall_small:.4f}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("[VALIDATION SUMMARY]")

    fire_ap50 = class_results.get("fire", {}).get("ap50", None)
    smoke_ap50 = class_results.get("smoke", {}).get("ap50", None)

    if fire_ap50 is not None:
        print(f"AP50-fire: {fire_ap50:.4f}")
    if smoke_ap50 is not None:
        print(f"AP50-smoke: {smoke_ap50:.4f}")

    if ap50_small is not None:
        print(f"AP50-small: {ap50_small:.4f}")
    if recall_small is not None:
        print(f"Recall-small@{args.small_match_iou}: {recall_small:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
