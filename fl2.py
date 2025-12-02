import os
import numpy as np
import time
from tqdm import tqdm
from typing import List
#import torch
##import matplotlib.patches as patches
import onnxruntime as ort
from transformers import AutoProcessor
from PIL import Image
import requests
#import copy

from datasets import load_dataset
#from pycocotools import mask as maskUtils
#%matplotlib inline

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
        )
        self.text_embed = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/embed_tokens_q4f16.onnx"),
            providers=providers,
        )
        self.encoder = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/encoder_model_q4f16.onnx"),
            providers=providers,
        )
        self.decoder_prefill = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/decoder_model_q4f16.onnx"),
            providers=providers,
        )
        self.decoder_decode = ort.InferenceSession(
            os.path.join(onnx_dir, "weight_files/decoder_model_merged_q4.onnx"),
            providers=providers,
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
        image,
        prompt: str = "<MORE_DETAILED_CAPTION>",
        expr: str = "",
        max_new_tokens: int = 1024
    ) -> (dict, float):


        #image = Image.open(image_path)
        prompt = f"{prompt} {expr}".strip()
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

    def infer_from_image(
        self,
        image,
        prompt: str = "<CAPTION_TO_PHRASE_GROUNDING>",
        expr: str = "",
        max_new_tokens: int = 1024
    ) -> None:

        parsed_answer, inference_time = self.generate_caption(image, prompt, expr, max_new_tokens)
        print(f"Inference Time: {inference_time:.4f} seconds")
        print("Answer:", parsed_answer)

dataset = load_dataset("jxu124/refcoco-benchmark", split="refcoco_unc_val")
COCO_IMG_ROOT = "~/coco/val2014"

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
                raw = inference_grounding_florence2(img, "<CAPTION_TO_PHRASE_GROUNDING>", expr)
                result = raw["<CAPTION_TO_PHRASE_GROUNDING>"]
                pred_xywh_norm = result["bboxes"][0]  # [cx, cy, w, h] normalized
                
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

#res = evaluate_dataset(dataset, COCO_IMG_ROOT, n_samples= None)
#print("Accuracy @ IoU>=0.5 (100 samples):", res)

if __name__ == '__main__':
    model = Florence2OnnxModel(
        providers=["CUDAExecutionProvider"],
        warmup_iterations=10
    )

img_url = "https://www.datocms-assets.com/53444/1687431221-testing-the-saturn-v-rocket.jpg?auto=format&w=1200"
expr = "A space rocket"
response = requests.get(img_url, stream=True)
image = Image.open(response.raw).convert("RGB")

model.infer_from_image(
    image=image,
    prompt="<CAPTION_TO_PHRASE_GROUNDING>",
    expr=expr,
    max_new_tokens=1024
)