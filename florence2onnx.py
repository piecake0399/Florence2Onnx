import os
import time
from typing import List

import numpy as np
from PIL import Image
import onnxruntime as ort
from transformers import AutoProcessor

from datasets import load_dataset
import json
import numpy as np

import psutil


so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
# WEIGHT FILES CAN BE DOWNLOADED FROM HERE: https://huggingface.co/onnx-community/Florence-2-base-ft/tree/main/onnx
class Florence2OnnxModel:
    def __init__(
        self,
        providers: List[str] = None,
        warmup_iterations: int = 10,
    ):

        # Change working directory to the provided ONNX directory
        onnx_dir: str = os.path.dirname(os.path.abspath(__file__))
        os.chdir(onnx_dir)

        processor_dir: str = os.path.join(onnx_dir, "processor_files")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.vision_encoder = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/vision_encoder_q4f16.onnx"),
            providers=providers,
            sess_options=so
        )
        self.text_embed = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/embed_tokens_q4f16.onnx"),
            providers=providers,
            sess_options=so
        )
        self.encoder = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/encoder_model_q4f16.onnx"),
            providers=providers,
            sess_options=so
        )
        self.decoder_prefill = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/decoder_model_q4f16.onnx"),
            providers=providers,
            sess_options=so
        )
        self.decoder_decode = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/decoder_model_merged_q4.onnx"),
            providers=providers,
            sess_options=so
        )

        self.processor = AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True)

        self._warmup(iterations=warmup_iterations)

    def _open_image(self, image_input):
        """
        Accept either a PIL.Image or a path string. Return PIL.Image.
        """
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        else:
            raise ValueError("image_input must be PIL.Image or path string")

    def _warmup(self, iterations: int = 10) -> None:
        dummy_image = Image.new("RGB", (384, 384))
        dummy_prompt = "<MORE_DETAILED_CAPTION>"
        dummy_inputs = self.processor(text=dummy_prompt, images=dummy_image, return_tensors="np")

        for _ in range(iterations):
            _ = self.vision_encoder.run(None, {"pixel_values": dummy_inputs["pixel_values"]})
            _ = self.text_embed.run(None, {"input_ids": dummy_inputs["input_ids"]})
            _ = self.encoder.run(None, {
                "inputs_embeds": np.zeros((1, 10, 768), dtype=np.float32),
                "attention_mask": np.zeros((1, 10), dtype=np.int64)
            })

    def generate_caption(
        self,
        image_input,
        prompt: str = "<MORE_DETAILED_CAPTION>",
        max_new_tokens: int = 1024
    ) -> (object, float):
        """
        image_input: PIL.Image or path string
        Returns: parsed_answer (could be dict or string) and elapsed_time
        """
        image = self._open_image(image_input)
        inputs = self.processor(text=prompt, images=image, return_tensors="np", do_resize=True)

        start_time = time.time()

        image_features = self.vision_encoder.run(
            None, {"pixel_values": inputs["pixel_values"]}
        )[0]

        inputs_embeds = self.text_embed.run(
            None, {"input_ids": inputs["input_ids"]}
        )[0]

        batch_size, image_token_length = image_features.shape[:-1]
        image_attention_mask = np.ones((batch_size, image_token_length), dtype=np.int64)
        task_prefix_embeds = inputs_embeds
        task_prefix_attention_mask = np.ones((batch_size, task_prefix_embeds.shape[1]), dtype=np.int64)

        if task_prefix_attention_mask.ndim == 3:
            task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

        inputs_embeds = np.concatenate([image_features, task_prefix_embeds], axis=1)
        attention_mask = np.concatenate([image_attention_mask, task_prefix_attention_mask], axis=1)

        encoder_hidden_states = self.encoder.run(
            None,
            {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        )[0]

        decoder_outs = self.decoder_prefill.run(
            None,
            {
                "inputs_embeds": inputs_embeds[:, -1:],
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask
            }
        )
        encoder_kv = decoder_outs[1:]

        generated_tokens = []
        while len(generated_tokens) < max_new_tokens:
            logits = decoder_outs[0]
            decoder_kv = decoder_outs[1:]

            next_token_logits = logits[:, -1, :]
            next_token = int(np.argmax(next_token_logits, axis=-1)[0])
            generated_tokens.append(next_token)

            # Break if the EOS token (assumed to be token id 2) is generated.
            if next_token == 2:
                break

            next_input_embeds = self.text_embed.run(
                None,
                {"input_ids": np.array([[next_token]], dtype=np.int64)}
            )[0]

            decoder_outs = self.decoder_decode.run(
                None,
                {
                    "use_cache_branch": np.array([True], dtype=np.bool_),
                    "inputs_embeds": next_input_embeds,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": attention_mask,
                    # NOTE: giữ nguyên mapping past_key_values như file ONNX của bạn
                    "past_key_values.0.decoder.key": decoder_kv[0],
                    "past_key_values.0.decoder.value": decoder_kv[1],
                    "past_key_values.0.encoder.key": encoder_kv[2],
                    "past_key_values.0.encoder.value": encoder_kv[3],
                    "past_key_values.1.decoder.key": decoder_kv[4],
                    "past_key_values.1.decoder.value": decoder_kv[5],
                    "past_key_values.1.encoder.key": encoder_kv[6],
                    "past_key_values.1.encoder.value": encoder_kv[7],
                    "past_key_values.2.decoder.key": decoder_kv[8],
                    "past_key_values.2.decoder.value": decoder_kv[9],
                    "past_key_values.2.encoder.key": encoder_kv[10],
                    "past_key_values.2.encoder.value": encoder_kv[11],
                    "past_key_values.3.decoder.key": decoder_kv[12],
                    "past_key_values.3.decoder.value": decoder_kv[13],
                    "past_key_values.3.encoder.key": encoder_kv[14],
                    "past_key_values.3.encoder.value": encoder_kv[15],
                    "past_key_values.4.decoder.key": decoder_kv[16],
                    "past_key_values.4.decoder.value": decoder_kv[17],
                    "past_key_values.4.encoder.key": encoder_kv[18],
                    "past_key_values.4.encoder.value": encoder_kv[19],
                    "past_key_values.5.decoder.key": decoder_kv[20],
                    "past_key_values.5.decoder.value": decoder_kv[21],
                    "past_key_values.5.encoder.key": encoder_kv[22],
                    "past_key_values.5.encoder.value": encoder_kv[23],
                }
            )

        end_time = time.time()
        total_time = end_time - start_time

        generated_text = self.processor.batch_decode(
            [generated_tokens], skip_special_tokens=False
        )[0]

        # post_process_generation có thể trả về dict hoặc string tuỳ task
        try:
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=prompt, image_size=(image.width, image.height)
            )
        except Exception:
            parsed_answer = generated_text

        return parsed_answer, total_time

    def generate_phrase_grounding(
        self,
        image_input,
        query: str
    ):
        prompt = f"<PHRASE_GROUNDING> {query}"
        parsed_answer, infer_time = self.generate_caption(
            image_input,
            prompt=prompt,
            max_new_tokens=256
        )
        return parsed_answer, infer_time

    def infer_from_image(
        self,
        image_input,
        prompt: str = "<MORE_DETAILED_CAPTION>",
        max_new_tokens: int = 1024
    ) -> None:

        parsed_answer, inference_time = self.generate_caption(image_input, prompt, max_new_tokens)
        print(f"Inference Time: {inference_time:.4f} seconds")
        print("Answer:", parsed_answer)

def bbox_iou(boxA, boxB):
    # box format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union = boxA_area + boxB_area - interArea

    return interArea / union if union > 0 else 0

def benchmark_refcoco(model, coco_root, sample_size=200):
    """
    Benchmark Florence2 Phrase Grounding + đo CPU usage trung bình.
    """

    ds = load_dataset("jxu124/refcoco-benchmark", split="refcoco_unc_val")

    total = 0
    avg_iou = 0.0
    avg_time = 0.0

    times = []
    cpu_records = []

    proc = psutil.Process(os.getpid())

    for idx, item in enumerate(ds):
        if idx >= sample_size:
            break

        # Dataset may provide either a PIL.Image under "image", or provide "image_id"/"image" path
        # Prefer using dataset's image if available (it's a PIL.Image)
        if isinstance(item.get("image"), Image.Image):
            image_obj = item["image"]
        else:
            # try to construct path from "image" (string) or "image_id"
            if isinstance(item.get("image"), str):
                image_path = os.path.join(coco_root, item["image"])
            elif "image_id" in item:
                image_path = os.path.join(coco_root, f"COCO_val2014_{int(item['image_id']):012d}.jpg")
            else:
                raise RuntimeError("Cannot determine image path from dataset item")
            image_obj = Image.open(image_path).convert("RGB")

        query = item.get("query") or item.get("phrase") or item.get("expression") or item.get("caption")
        if query is None:
            # fallback to a default prompt
            query = "<PHRASE_GROUNDING>"

        gt_bbox = item.get("bbox")
        if gt_bbox is None:
            # If dataset uses different field for bbox, try alternatives
            gt_bbox = item.get("box") or item.get("ground_truth_bbox")
        if gt_bbox is None:
            # skip if no GT bbox
            print(f"[{idx}] skip sample (no GT bbox)")
            continue

        # Measure CPU time of THIS process before/after inference
        cpu_before = proc.cpu_times().user + proc.cpu_times().system
        wall_before = time.time()

        # --- Inference ---
        result, t = model.generate_phrase_grounding(image_obj, query)

        wall_after = time.time()
        cpu_after = proc.cpu_times().user + proc.cpu_times().system

        elapsed_wall = wall_after - wall_before
        cpu_time_delta = cpu_after - cpu_before

        # CPU percent relative to all CPUs (system-wide %) approximated:
        cpu_iter = 0.0
        try:
            cpu_iter = (cpu_time_delta / (elapsed_wall + 1e-12)) * 100.0 / max(1, psutil.cpu_count(logical=True))
        except Exception:
            cpu_iter = 0.0

        cpu_records.append(cpu_iter)
        times.append(t)

        # === Lấy bbox AI ===
        pred_bbox = None
        # If model output is dict-like and contains bbox/polygon
        if isinstance(result, dict):
            if "bboxes" in result and len(result["bboxes"]) > 0:
                pred = result["bboxes"][0]
                # if format [x1,y1,x2,y2] -> convert to [x,y,w,h]
                if len(pred) >= 4:
                    if pred[2] > pred[0] and pred[3] > pred[1]:
                        pred_bbox = [float(pred[0]), float(pred[1]), float(pred[2]) - float(pred[0]), float(pred[3]) - float(pred[1])]
                    else:
                        pred_bbox = [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])]
            elif "polygons" in result and len(result["polygons"]) > 0:
                poly = result["polygons"][0]
                xs = poly[::2]
                ys = poly[1::2]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                pred_bbox = [minx, miny, maxx - minx, maxy - miny]
        elif isinstance(result, str):
            parsed = parse_bbox_from_string(result)
            if parsed is not None:
                pred_bbox = parsed

        if pred_bbox is None:
            print(f"[{idx}] Model returned no bbox — skipping sample")
            continue

        # ensure gt bbox numeric list [x,y,w,h]
        gt = [float(x) for x in gt_bbox]

        iou = bbox_iou(pred_bbox, gt)

        total += 1
        avg_iou += iou
        avg_time += t

        print(f"[{idx}] Query: {query}")
        print(f"GT: {gt} | Pred: {pred_bbox}")
        print(f"IoU = {iou:.3f}, Time = {t:.3f}s, CPU% = {cpu_iter:.1f}%\n")

    # === SUMMARY ===
    print("\n==== BENCHMARK COMPLETE ====")
    print(f"Total samples: {total}")
    if total > 0:
        print(f"Avg IoU: {avg_iou / total:.4f}")
        print(f"Avg time: {avg_time / total:.3f}s")
    if len(times) > 0:
        print(f"Fastest inference: {min(times):.3f}s")
        print(f"Slowest inference: {max(times):.3f}s")
    if len(cpu_records) > 0:
        print(f"Avg CPU Usage (this process): {sum(cpu_records) / len(cpu_records):.1f}%")

if __name__ == '__main__':
    model = Florence2OnnxModel(
        providers=["CPUExecutionProvider"],
        warmup_iterations=5
    )
    COCO_IMG_ROOT = "coco/val2014"
    benchmark_refcoco(model, COCO_IMG_ROOT, sample_size=200)
