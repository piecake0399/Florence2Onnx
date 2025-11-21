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

def evaluate_refcoco(ds, coco_root, sample_size=100):

    total = 0
    correct = 0

    cpu_samples = []
    process = psutil.Process()

    inference_times = []

    start_time = time.time()

    for idx, item in enumerate(ds):
        if idx >= sample_size:
            break

        # --- Load image ---
        img = item.get("image")
        if isinstance(img, Image.Image):
            image_obj = img
        else:
            # trường hợp không phổ biến
            if "image_info" in item:
                file_name = item["image_info"]["file_name"]
                image_path = os.path.join(coco_root, file_name)
                image_obj = Image.open(image_path).convert("RGB")
            else:
                raise RuntimeError("Cannot determine image for sample")

        # --- Iterate over ref_list (multiple GT) ---
        ref_list = item.get("ref_list")
        if ref_list is None:
            print(f"[{idx}] skip sample (no ref_list)")
            continue

        for ref_idx, ref_item in enumerate(ref_list):

            # Ground truth bbox location
            ann_info = ref_item.get("ann_info", {})
            gt_bbox = ann_info.get("bbox")

            if gt_bbox is None:
                print(f"[{idx}] skip ref {ref_idx} (no GT bbox)")
                continue

            x, y, w, h = gt_bbox
            gt_xyxy = [x, y, x + w, y + h]

            # Grounding sentences
            sentences = ref_item.get("ref_info", {}).get("sentences", [])
            if not sentences:
                print(f"[{idx}] skip ref {ref_idx} (no sentences)")
                continue

            for s in sentences:
                query = s.get("sent")
                if not query:
                    print(f"[{idx}] skip sentence (no query)")
                    continue

                # === RECORD CPU USAGE BEFORE INFERENCE ===
                cpu_percent = process.cpu_percent(interval=None)
                t0 = time.time()
                # --- RUN FLORENCE2 ---
                pred_bbox = inference_grounding_florence2(image_obj, query)

                t1 = time.time()
                infer_time = t1 - t0
                inference_timess.append(infer_time)

                # === RECORD CPU USAGE AFTER INFERENCE ===
                cpu_percent = process.cpu_percent(interval=None)
                cpu_samples.append(cpu_percent)

                total += 1

                if pred_bbox is None:
                    print(f"[{idx}] no prediction for query: {query}")
                    continue

                iou = bbox_iou(pred_bbox, gt_xyxy)
                if iou >= 0.5:
                    correct +=1

                # =====SUMMARY=====
                end_time = time.end()
                elapsed = end_time - start_time
                acc = correct / total if total > 0 else 0.0
                avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples > 0 else 0.0
                min_infer = min(inference_times) if inference_times else None
                max_infer = max(inference_times) if inference_times else None
                avg_infer = sum(inference_times) / len(inference_times) if inference_times else None

                print("\n=== COMPLETE ===")
                print(f"Accuracy: {acc:.4f}")
                print(f"Correct: {correct} / {total}")
                print(f"Avg inference time: {avg_infer:.3f}s")
                print(f"Fastest: {min_infer:.3f}s")
                print(f"Slowest: {max_infer:.3f}s")
                print(f"Avg CPU Usage: {avg_cpu:.1f}%")

                return {
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                    "avg_cpu": avg_cpu,
                    "time_sec": elapsed,
                    "min_infer_sec": min_infer,
                    "max_infer_sec": max_infer,
                    "avg_infer_sec": avg_infer,
                    }

    print("Done evaluating.")

if __name__ == '__main__':
    model = Florence2OnnxModel(
        providers=["CPUExecutionProvider"],
        warmup_iterations=5
    )
    COCO_IMG_ROOT = "coco/val2014"
    evaluate_refcoco(model, COCO_IMG_ROOT, sample_size=200)
