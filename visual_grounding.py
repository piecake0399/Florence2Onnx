import time
import requests
import psutil
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image


# ============================================================
# 1. Load RefCOCO Dataset
# ============================================================
dataset = load_dataset("jxu124/refcoco-benchmark", split="refcoco_unc_val")


# ============================================================
# 2. IoU Calculation
# ============================================================
def compute_iou(boxA, boxB):
    """
    box format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxA_area + boxB_area - interArea

    return interArea / union


# ============================================================
# 3. Benchmark Function
# ============================================================
def evaluate_refcoco(server_url="http://localhost:3000/infer", iou_threshold=0.5, limit=None):
    """
    Evaluate the Florence2 server on RefCOCO dataset.
    - Shows tqdm progress bar
    - Computes accuracy
    - Records min/max/avg inference time
    - Tracks CPU usage
    """

    total = len(dataset) if limit is None else min(limit, len(dataset))

    correct = 0
    times = []
    cpu_usages = []

    for idx in tqdm(range(total), desc="Evaluating RefCOCO"):
        sample = dataset[idx]

        # === Extract fields ===
        img_path = sample["image"]
        expression = sample["expression"]
        gt_box = sample["bbox"]   # ground-truth bbox [x1,y1,x2,y2]

        # === Send to Florence2 server ===
        payload = {
            "image_path": img_path,
            "task": "<CAPTION_TO_PHRASE_GROUNDING>",
            "expr": expression
        }

        start_time = time.time()
        cpu_usages.append(psutil.cpu_percent(interval=None))

        response = requests.post(server_url, json=payload).json()

        infer_time = time.time() - start_time
        times.append(infer_time)

        # If no prediction, count as fail
        result = response.get("result", {})
        preds = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        pred_boxes = preds.get("bboxes", [])

        if len(pred_boxes) == 0:
            continue

        # Florence2 returns normalized xywh or xyxy depending on task.
        # RefCOCO uses xyxy in pixel coordinates.
        # Florence2 grounding gives xywh normalized → convert to xyxy pixels.

        # Load image size
        img = Image.open(img_path)
        W, H = img.size

        pred_xywh = pred_boxes[0]  # only top-1
        x = pred_xywh[0] * W
        y = pred_xywh[1] * H
        w = pred_xywh[2] * W
        h = pred_xywh[3] * H

        pred_xyxy = [x, y, x + w, y + h]

        # === IoU ===
        iou = compute_iou(pred_xyxy, gt_box)

        if iou >= iou_threshold:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "total_samples": total,
        "accuracy": accuracy,
        "avg_inference_time": np.mean(times),
        "min_inference_time": np.min(times),
        "max_inference_time": np.max(times),
        "avg_cpu_usage": np.mean(cpu_usages),
    }


# ============================================================
# Run benchmark
# ============================================================
result = evaluate_refcoco(limit=100)  # run on 100 samples for testing
print("\n===== Benchmark Result =====")
for k, v in result.items():
    print(f"{k}: {v}")