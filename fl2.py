import os
import json
import numpy as np
from tqdm import tqdm
#import torch
import tempfile, subprocess, json
##import matplotlib.patches as patches

from transformers import AutoProcessor, Florence2ForConditionalGeneration
from PIL import Image
import requests
#import copy

from datasets import load_dataset
#from pycocotools import mask as maskUtils
#%matplotlib inline

# model_id = 'onnx-community/Florence-2-base'
# device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# model = Florence2ForConditionalGeneration.from_pretrained(
#     model_id,
#     dtype="q8"
# )
#model.eval()

dataset = load_dataset("jxu124/refcoco-benchmark", split="refcoco_unc_val")
COCO_IMG_ROOT = "~/coco/val2014"

# def inference_grounding_florence2(image_pil, expr):
#     task = "<CAPTION_TO_PHRASE_GROUNDING>"
#     prompts = processor.construct_prompts(task, expr)
#     inputs = processor(image_pil, prompts)
#     # ONNX generate
#     generated_ids = model.generate(
#         **inputs,
#         max_new_tokens=128
#     )
#     out_str = processor.batch_decode(
#         generated_ids,
#         skip_special_tokens=False
#     )[0]
#     result = processor.post_process_generation(
#         out_str,
#         task,
#         image_pil.size
#     )
#     grounding = result.get(task, {})
#     if "bboxes" in grounding and len(grounding["bboxes"]) > 0:
#         return grounding["bboxes"][0]
#     else:
#         return None


def inference_grounding_florence2(img: Image.Image, task, expr):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)


    proc = subprocess.run(
        ["node", "florence2.js", tmp.name, task, expr],
        capture_output=True,
        text=True,
        check=True
    )

    return json.loads(proc.stdout)


def compute_iou(boxA, boxB):
    """boxA, boxB dạng [x1,y1,x2,y2] tuyệt đối"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def evaluate_dataset(dataset, img_root, n_samples=None):
    """
    dataset: HF dataset with fields: image_id, ann(bbox), ref_list
    img_root: path containing images (not used here)
    n_samples: if None use all, if int use subset
    """
    total = 0
    correct = 0
    processed_samples = 0

    for i, sample in enumerate(tqdm(dataset)):
        if (n_samples is not None) and (processed_samples >= n_samples):
            break

        img = sample["image"].convert("RGB")
        W, H = img.size
        ref_list = sample["ref_list"]

        for ref_info in ref_list:
            ann_info = ref_info["ann_info"]
            gt_bbox = ann_info["bbox"]  # COCO format: [x, y, w, h]
            x, y, w, h = gt_bbox
            gt = [x, y, x + w, y + h]  # [x1, y1, x2, y2]

            sentences = ref_info["ref_info"]["sentences"]
            for sentence_info in sentences:
                expr = sentence_info["sent"]

                # inference via Node Florence2
                pred_xywh_norm = inference_grounding_florence2(img, "<CAPTION_TO_PHRASE_GROUNDING>", expr)

                if pred_xywh_norm is None:
                    total += 1
                    processed_samples += 1
                    continue

                # convert normalized [cx, cy, w, h] → pixel coords
                cx, cy, nw, nh = pred_xywh_norm
                px = cx * W
                py = cy * H
                pw = nw * W
                ph = nh * H

                pred = [
                    px - pw / 2,
                    py - ph / 2,
                    px + pw / 2,
                    py + ph / 2,
                ]

                # compute IoU
                iou = compute_iou(pred, gt)
                if iou >= 0.5:
                    correct += 1
                total += 1
                processed_samples += 1

    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "correct": correct, "total": total}

res = evaluate_dataset(dataset, COCO_IMG_ROOT, n_samples= None)
print("Accuracy @ IoU>=0.5 (100 samples):", res)