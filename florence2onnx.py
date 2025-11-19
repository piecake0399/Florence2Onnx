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
        image_path: str,
        prompt: str = "<MORE_DETAILED_CAPTION>",
        max_new_tokens: int = 1024
    ) -> (str, float):


        image = Image.open(image_path)
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

        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        return parsed_answer, total_time

    def generate_phrase_grounding(
        self, 
        image_path: str, 
        query: str
    ):
        prompt = f"<PHRASE_GROUNDING> {query}"
        parsed_answer, infer_time = self.generate_caption(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=256
        )
        return parsed_answer, infer_time

    def infer_from_image(
        self,
        image_path: str,
        prompt: str = "<MORE_DETAILED_CAPTION>",
        max_new_tokens: int = 1024
    ) -> None:

        parsed_answer, inference_time = self.generate_caption(image_path, prompt, max_new_tokens)
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

    from datasets import load_dataset
    ds = load_dataset("jxu124/refcoco-benchmark", split="refcoco_unc_val")

    total = 0
    avg_iou = 0.0
    avg_time = 0.0

    times = []
    cpu_records = []

    # Reset CPU measurement
    psutil.cpu_percent(interval=None)

    for idx, item in enumerate(ds):
        if idx >= sample_size:
            break

        img_path = os.path.join(coco_root, item["image"])
        query = item["query"]
        gt_bbox = item["bbox"]  # x,y,w,h

        # đo CPU usage ngay trước khi inference (snapshot 1)
        cpu_before = psutil.cpu_percent(interval=None)

        # --- Inference ---
        result, t = model.generate_phrase_grounding(img_path, query)

        # đo CPU usage ngay sau khi inference (snapshot 2)
        cpu_after = psutil.cpu_percent(interval=None)

        # CPU usage của iteration này
        cpu_iter = (cpu_before + cpu_after) / 2
        cpu_records.append(cpu_iter)

        times.append(t)

        # === Lấy bbox AI ===
        if "bboxes" in result:
            pred = result["bboxes"][0]  # [x1,y1,x2,y2]
            pred_bbox = [
                pred[0],
                pred[1],
                pred[2] - pred[0],
                pred[3] - pred[1]
            ]
        elif "polygons" in result:
            poly = result["polygons"][0]
            xs, ys = poly[::2], poly[1::2]
            pred_bbox = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
        else:
            continue

        iou = bbox_iou(pred_bbox, gt_bbox)

        total += 1
        avg_iou += iou
        avg_time += t

        print(f"[{idx}] Query: {query}")
        print(f"GT: {gt_bbox} | Pred: {pred_bbox}")
        print(f"IoU = {iou:.3f}, Time = {t:.3f}s, CPU = {cpu_iter:.1f}%\n")

    # === SUMMARY ===

    print("\n==== BENCHMARK COMPLETE ====")
    print(f"Total samples: {total}")
    print(f"Avg IoU: {avg_iou / total:.4f}")
    print(f"Avg time: {avg_time / total:.3f}s")
    print(f"Fastest inference: {min(times):.3f}s")
    print(f"Slowest inference: {max(times):.3f}s")
    print(f"Avg CPU Usage: {sum(cpu_records) / len(cpu_records):.1f}%")

if __name__ == '__main__':
    model = Florence2OnnxModel(
        providers=["CPUExecutionProvider"],
        warmup_iterations=5
    )
    COCO_IMG_ROOT = "coco/val2014"
    benchmark_refcoco(model, COCO_IMG_ROOT, sample_size=200)
