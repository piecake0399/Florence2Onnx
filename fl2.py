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

        ROOT = "weight_files/"
        FL2BASE = ROOT + "fl2base/"
        FL2BASE_FT = ROOT + "fl2base-ft/"

        self.vision_encoder = ort.InferenceSession(
            os.path.join(onnx_dir, FL2BASE_FT + "/vision_encoder_q4f16.onnx"),
            providers=providers,
        )
        self.text_embed = ort.InferenceSession(
            os.path.join(onnx_dir, FL2BASE_FT + "/embed_tokens_q4f16.onnx"),
            providers=providers,
        )
        self.encoder = ort.InferenceSession(
            os.path.join(onnx_dir, FL2BASE_FT + "/encoder_model_q4f16.onnx"),
            providers=providers,
        )
        self.decoder_prefill = ort.InferenceSession(
            os.path.join(onnx_dir, FL2BASE_FT + "/decoder_model_q4f16.onnx"),
            providers=providers,
        )
        self.decoder_decode = ort.InferenceSession(
            os.path.join(onnx_dir, FL2BASE_FT + "/decoder_model_merged_q4.onnx"),
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
        prompt: str = "<CAPTION_TO_PHRASE_GROUNDING>",
        expr: str = "",
        max_new_tokens: int = 128
    ) -> (dict, float):


        #image = Image.open(image_path)
        prompt = f"{prompt} {expr}"
        inputs = self.processor(text=prompt, images=image, return_tensors="np", do_resize=True)
        #print("INPUT KEYS:", list(inputs.keys()))
        
        # ======== CPU USAGE START MEASURE ========
        start_time = time.time()
        # process = psutil.Process()
        # rss_before = process.memory_info().rss
        # ==========================================

        image_features = self.vision_encoder.run(
            None, {"pixel_values": inputs["pixel_values"]}
        )[0]
        #print("Original size", image.size)
        #print("Image resize", inputs["pixel_values"].shape)

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
            #probs = softmax(next_token_logits)
            next_token = np.argmax(next_token_logits, axis=-1)[0]
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

        # ==========================================
        end_time = time.time()
        total_time = end_time - start_time
        
        # ======== CPU USAGE END MEASURE ========
        # rss_after = process.memory_info().rss
        # peak_memory = rss_after - rss_before  # bytes
        # ==========================================

        #print("GENERATED TOKENS:", generated_tokens[:80])
        generated_text = self.processor.batch_decode(
            [generated_tokens], skip_special_tokens=False
        )[0]
        #print("GENERATED TEXT:", repr(generated_text))

        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task="<CAPTION_TO_PHRASE_GROUNDING>", 
            image_size=(image.width*2, image.height*2)
        )
        return parsed_answer, total_time

    def infer_from_image(
        self,
        image,
        prompt: str = "<MORE_DETAILED_CAPTION>",
        expr: str = "",
        max_new_tokens: int = 128
    ) -> (list, str, float):

        parsed_answer, inference_time = self.generate_caption(image, prompt, expr, max_new_tokens)
        
        task_key = list(parsed_answer.keys())[0]  # ví dụ "<CAPTION_TO_PHRASE_GROUNDING>"
        result = parsed_answer[task_key]

        bboxes = result.get("bboxes", [])
        labels = result.get("labels", [])

        if len(bboxes) == 0:
            return None, None, inference_time

        bbox = bboxes[0]
        label = labels[0] if len(labels) > 0 else None

        # print(f"Inference Time: {inference_time:.4f} seconds")
        # print("Bbox:", bbox)
        # print("Label:", label)
        # print(f"Peak RAM usage: {peak_mem / 1024 / 1024:.2f} MB")

        return bbox, label, inference_time

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
                    prompt="<CAPTION_TO_PHRASE_GROUNDING>",
                    expr=expr,
                    max_new_tokens=256
                    )
                
                infer_times.append(infer_time)
                # peak_mems.append(peak_mem)
                if bbox is None:
                    #consider as wrong
                    bbox_draw(img, expr, None, gt)
                    total += 1
                    processed_samples += 1
                    continue
                else: 
                    print("Detected bbox:", bbox)
                    # print("Memory used (MB):", peak_mem / 1024 / 1024)
                    print("Inference time (s):", infer_time)
                    bbox_draw(img, expr, bbox, gt)

                # compute IoU
                iou = compute_iou(bbox, gt)
                if iou >= 0.3:
                    correct += 1
                total += 1
                processed_samples += 1

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

    # img_url = "https://www.datocms-assets.com/53444/1687431221-testing-the-saturn-v-rocket.jpg?auto=format&w=1200"
    # expr = "A space rocket"

    # response = requests.get(img_url, stream=True)

    # image = Image.open("car.jpg")
    # expr = "car"
    # model.infer_from_image(image, prompt="<CAPTION_TO_PHRASE_GROUNDING>", expr=expr, max_new_tokens=32)

    dataset = load_dataset("jxu124/refcoco-benchmark", split="refcoco_unc_val")
    #COCO_IMG_ROOT = "~/coco/val2014"

    evaluate_dataset(model, dataset, n_samples= 100)