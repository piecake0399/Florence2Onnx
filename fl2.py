# !pip install --upgrade onnxruntime==1.20.1 transformers==4.48.3 pillow==9.5.0 datasets==2.13.1 psutil==5.9.5 tqdm==4.66.1 matplotlib==3.8.0

import os
import time
from typing import List
#from scipy.special import softmax
import numpy as np
from PIL import Image
import requests
import psutil
import onnxruntime as ort
from transformers import AutoProcessor

from tqdm import tqdm
from datasets import load_dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# WEIGHT FILES CAN BE DOWNLOADED FROM HERE: https://huggingface.co/onnx-community/Florence-2-base-ft/tree/main/onnx
import os
import time
import numpy as np
import onnxruntime as ort
from typing import List, Optional
from PIL import Image
from transformers import AutoProcessor


class Florence2OnnxModel:
    """
    Florence-2 ONNX inference wrapper
    Supports multiple Florence-2 tasks with correct prompt routing.
    """

    # ===== Task definitions =====
    NO_TEXT_TASKS = {
        "<CAPTION>",
        "<DETAILED_CAPTION>",
        "<MORE_DETAILED_CAPTION>",
        "<OD>",
        "<OCR>",
        "<OCR_WITH_REGION>",
        "<DENSE_REGION_CAPTION>",
        "<REGION_PROPOSAL>",
    }

    TEXT_TASKS = {
        "<CAPTION_TO_PHRASE_GROUNDING>",
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "<OPEN_VOCABULARY_DETECTION>",
    }

    def __init__(
        self,
        providers: Optional[List[str]] = None,
        warmup_iterations: int = 5,
    ):
        onnx_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(onnx_dir)

        if providers is None:
            providers = ["CPUExecutionProvider"]

        ROOT = "weight_files"

        self.vision_encoder = ort.InferenceSession(
            os.path.join(onnx_dir, ROOT, "vision_encoder_q4f16.onnx"),
            providers=providers,
        )
        self.text_embed = ort.InferenceSession(
            os.path.join(onnx_dir, ROOT, "embed_tokens_q4f16.onnx"),
            providers=providers,
        )
        self.encoder = ort.InferenceSession(
            os.path.join(onnx_dir, ROOT, "encoder_model_q4f16.onnx"),
            providers=providers,
        )
        self.decoder_prefill = ort.InferenceSession(
            os.path.join(onnx_dir, ROOT, "decoder_model_q4f16.onnx"),
            providers=providers,
        )
        self.decoder_decode = ort.InferenceSession(
            os.path.join(onnx_dir, ROOT, "decoder_model_merged_q4.onnx"),
            providers=providers,
        )

        processor_dir = os.path.join(onnx_dir, "processor_files")
        self.processor = AutoProcessor.from_pretrained(
            processor_dir, trust_remote_code=True
        )

        self._warmup(warmup_iterations)

    # ------------------------------------------------------------------
    # Prompt routing
    # ------------------------------------------------------------------
    def _build_prompt(self, task: str, expr: Optional[str]) -> str:
        if task in self.NO_TEXT_TASKS:
            return task

        if task in self.TEXT_TASKS:
            if not expr or not expr.strip():
                raise ValueError(f"Task {task} requires expression text")
            return f"{task}{expr}"

        raise ValueError(f"Unknown Florence-2 task: {task}")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    def _warmup(self, iterations: int):
        dummy_img = Image.new("RGB", (384, 384))
        dummy_inputs = self.processor(
            text="<MORE_DETAILED_CAPTION>",
            images=dummy_img,
            return_tensors="np",
        )

        for _ in range(iterations):
            _ = self.vision_encoder.run(
                None, {"pixel_values": dummy_inputs["pixel_values"]}
            )
            _ = self.text_embed.run(
                None, {"input_ids": dummy_inputs["input_ids"]}
            )
            _ = self.encoder.run(
                None,
                {
                    "inputs_embeds": np.zeros((1, 10, 768), dtype=np.float32),
                    "attention_mask": np.ones((1, 10), dtype=np.int64),
                },
            )

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        image: Image.Image,
        task: str,
        expr: Optional[str] = None,
        max_new_tokens: int = 128,
    ):
        prompt = self._build_prompt(task, expr)

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="np",
            do_resize=True,
        )

        start_time = time.time()

        # Vision encoder
        image_features = self.vision_encoder.run(
            None, {"pixel_values": inputs["pixel_values"]}
        )[0]
        # print("Vision feat stats:",
        #     image_features.min(),
        #     image_features.max(),
        #     image_features.mean(),
        #     image_features.std()
        # )
        #image_features[:] = 0

        # Text embed
        text_embeds = self.text_embed.run(
            None, {"input_ids": inputs["input_ids"]}
        )[0]

        # Build encoder input
        B, T_img = image_features.shape[:-1]
        img_mask = np.ones((B, T_img), dtype=np.int64)
        task_prefix_embeds = text_embeds
        task_mask = np.ones((B, task_prefix_embeds.shape[1]), dtype=np.int64)
        if task_mask.ndim == 3:
            task_mask = task_mask[:, 0]

        inputs_embeds = np.concatenate([image_features, text_embeds], axis=1)
        attention_mask = np.concatenate([img_mask, task_mask], axis=1)

        # Encoder
        encoder_hidden_states = self.encoder.run(
            None,
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            },
        )[0]

        # Decoder prefill
        # bos_id = self.processor.tokenizer.bos_token_id
        # bos_embed = self.text_embed.run(
        #     None,
        #     {"input_ids": np.array([[bos_id]], dtype=np.int64)}
        # )[0]
        next_token = self.processor.tokenizer.bos_token_id
        next_input_embeds = self.text_embed.run(None, {
            "input_ids": np.array([[next_token]], dtype=np.int64)
        })[0]
        decoder_outs = self.decoder_prefill.run(
            None,
            {
                "inputs_embeds": next_input_embeds,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask,
            },
        )

        encoder_kv = decoder_outs[1:]
        generated_tokens = []

        # Decode loop
        while len(generated_tokens) < max_new_tokens:
            logits = decoder_outs[0]
            decoder_kv = decoder_outs[1:]

            next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            generated_tokens.append(next_token)

            if next_token == 2:  # </s>
                break

            next_embed = self.text_embed.run(
                None,
                {"input_ids": np.array([[next_token]], dtype=np.int64)},
            )[0]

            decoder_outs = self.decoder_decode.run(
                None,
                {
                    "use_cache_branch": np.array([True], dtype=np.bool_),
                    "inputs_embeds": next_embed,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": attention_mask,
                    **{f"past_key_values.{i}.decoder.key": decoder_kv[i * 4]
                       for i in range(6)},
                    **{f"past_key_values.{i}.decoder.value": decoder_kv[i * 4 + 1]
                       for i in range(6)},
                    **{f"past_key_values.{i}.encoder.key": encoder_kv[i * 4 + 2]
                       for i in range(6)},
                    **{f"past_key_values.{i}.encoder.value": encoder_kv[i * 4 + 3]
                       for i in range(6)},
                },
            )

        total_time = time.time() - start_time

        text = self.processor.batch_decode(
            [generated_tokens], skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            text,
            task=task,
            image_size=(image.width*2, image.height*2),
        )

        return parsed, total_time

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def infer_from_image(
        self,
        image: Image.Image,
        task: str,
        expr: Optional[str] = None,
        max_new_tokens: int = 128,
    ):
        parsed, infer_time = self.generate(
            image=image,
            task=task,
            expr=expr,
            max_new_tokens=max_new_tokens,
        )

        task_key = list(parsed.keys())[0]
        result = parsed[task_key]

        # ------------------------------------------------
        # CASE 1: Text-only task (caption, OCR, etc.)
        # ------------------------------------------------
        if isinstance(result, str):
            return result, infer_time

        # ------------------------------------------------
        # CASE 2: Structured output (grounding, OD, ...)
        # ------------------------------------------------
        if isinstance(result, dict):
            bboxes = result.get("bboxes", [])
            labels = result.get("labels", [])

            if not bboxes:
                return None, None, infer_time

            bbox = bboxes[0]
            label = labels[0] if labels else None
            return bbox, label, infer_time

        # ------------------------------------------------
        # Fallback (should not happen)
        # ------------------------------------------------
        return None, infer_time



def compute_iou(boxA, boxB):
    """Bbox and ground truth format: [x1, y1, x2, y2]"""
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

def bbox_draw(image: Image.Image, expr: str, bbox: list, bbox2: list ,color: str = "red") -> Image.Image:
    # Vẽ bounding box bằng matplotlib
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color="red", linewidth=3)
        ax.add_patch(rect)
        x3, y3, x4, y4 = bbox2
        rect2 = plt.Rectangle((x3, y3), x4-x3, y4-y3, fill=False, color="green", linewidth=3)
        ax.add_patch(rect2)
        ax.text(x1, y1-20, expr, color="red", fontsize=12, weight='bold', backgroundcolor='white')
    else:
        plt.title("No bounding box detected")

    plt.axis("off")
    plt.show()

def evaluate_dataset(model, dataset, n_samples=None):
    """
    dataset: HF dataset with fields: image_id, ann(bbox), ref_list
    n_samples: nếu None dùng toàn bộ, nếu int dùng subset đầu
    """
    total = 0
    correct = 0
    processed_samples = 0  # Counter for processed samples
    infer_times = []
    
    process = psutil.Process()
    rss_before = process.memory_info().rss

    for i, sample in enumerate(tqdm(dataset)):
        if (n_samples is not None) and (processed_samples >= n_samples):
            break

        img_id = sample.get("image_id", f"sample_{i}")
        img = sample["image"].convert("RGB")
        ref_list = sample["ref_list"]

        for ref_info in ref_list:
            ann_info = ref_info["ann_info"]
            gt_bbox = ann_info["bbox"]  # COCO format: [x, y, w, h]
            # convert to [x1, y1, x2, y2]
            x, y, w, h = gt_bbox
            gt = [x, y, x + w, y + h]

            sentences = ref_info["ref_info"]["sentences"]
            for sentence_info in sentences:
                expr = sentence_info["sent"]

                print("\n========================")
                print(f"Image ID        : {img_id}")
                print(f"Expression      : \"{expr}\"")
                print(f"GT bbox (xyxy)  : {gt}")
                print("========================")
                
                # inference
                bbox, label, infer_time = model.infer_from_image(
                    image=img, 
                    task="<CAPTION_TO_PHRASE_GROUNDING>",
                    expr=expr,
                    max_new_tokens=256
                    )
                
                infer_times.append(infer_time)
                # peak_mems.append(peak_mem)
                if bbox is None:
                    # consider as wrong
                    # bbox_draw(img, expr, None, gt)
                    total += 1
                    processed_samples += 1
                    continue
                # else: 
                    # print("Memory used (MB):", peak_mem / 1024 / 1024)

                    # bbox_draw(img, expr, bbox, gt)

                # compute IoU
                iou = compute_iou(bbox, gt)
                if iou >= 0.5:
                    correct += 1
                total += 1
                processed_samples += 1
                print("Detected bbox:", bbox)
                print("Inference time (s):", infer_time)
                print(f"IoU: {iou:.4f} --> {'Correct' if iou >= 0.5 else 'Wrong'}")

                #input("Press Enter to continue to the next sample...")


    acc = correct / total if total > 0 else 0.0
    rss_after = process.memory_info().rss
    total_mem_used_bytes = rss_after - rss_before
    total_mem_used_mb = total_mem_used_bytes / 1024 / 1024

    print("------- Evaluation Results ------")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {acc*100:.2f}%")
    print("---------------------------------")
    print(f"Average inference time: {np.mean(infer_times):.4f} seconds")
    print(f"Minimum inference time: {np.min(infer_times):.4f} seconds")
    print(f"Maximum inference time: {np.max(infer_times):.4f} seconds")
    print("---------------------------------")
    print(f"Total memory used during benchmark: {total_mem_used_mb:.2f} MB")
    # print(f"Average peak RAM usage: {np.mean(peak_mems) / 1024 / 1024:.2f} MB")
    # print(f"Minimum peak RAM usage: {np.min(peak_mems) / 1024 / 1024:.2f} MB")
    # print(f"Maximum peak RAM usage: {np.max(peak_mems) / 1024 / 1024:.2f} MB")
    print("---------------------------------")
    #return {"accuracy": acc, "correct": correct, "total": total}

if __name__ == '__main__':
    model = Florence2OnnxModel(
        providers=["CPUExecutionProvider"],
        warmup_iterations=3
    )

    # image = Image.open("spaceshuttle.jpg")
    # expr = "The space rocket in the center"

    #response = requests.get(img_url, stream=True)

    # image = Image.open("car.jpg")
    # expr = "car"
    # result, time = model.infer_from_image(image, task="<CAPTION>", expr=None, max_new_tokens=128)
    # print("Answer:", result)
    # print(f"Inference time: {time:.4f} seconds")
    
    dataset = load_dataset("jxu124/refcoco-benchmark", split="refcoco_unc_val")
    evaluate_dataset(model, dataset, n_samples= None)



# TURNS OUT IT WAS JUST ONNXRUNTIME ERROR, NOT PIPELINE PROBLEM
# EUREKA!!!!!!!!!!
