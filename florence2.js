// florence2.js
import {
    Florence2ForConditionalGeneration,
    AutoProcessor,
    load_image,
} from '@huggingface/transformers';
import { loadFlorence2, runFlorence2 } from "./florence2_module.js";
import { Buffer } from "buffer";
import { Image } from "image-js"; // or another lib to reconstruct

const MODEL_ID = "onnx-community/Florence-2-base";

let model = null;
let processor = null;

/**
 * Load Florence2 model and processor (singleton).
 */
export async function loadFlorence2() {
    if (model && processor) {
        return { model, processor };
    }

    console.log("🔄 Loading Florence-2 model...");

    model = await Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
        dtype: "fp16",
    });

    processor = await AutoProcessor.from_pretrained(MODEL_ID);

    console.log("✅ Florence-2 loaded");

    return { model, processor };
}

/**
 * Run inference on an image with a given task + expression.
 */
export async function runFlorence2(pilImage, task, expr) {
    if (!model || !processor) {
        await loadFlorence2();
    }

    // Construct prompts
    const prompts = processor.construct_prompts(task, expr);
    const inputs = await processor(pilImage, prompts);

    // Run model
    const generated_ids = await model.generate({
        ...inputs,
        max_new_tokens: 200,
    });

    // Decode
    const generated_text = processor.batch_decode(
        generated_ids,
        { skip_special_tokens: false }
    )[0];

    // Post-process output
    const result = processor.post_process_generation(
        generated_text,
        task,
        pilImage.size
    );

    return result;
}

const imagePath = process.argv[2];
const task = process.argv[3];
const expr = process.argv[4];

const image = await load_image(imagePath);
await loadFlorence2();
const result = await runFlorence2(image, task, expr);

console.log(JSON.stringify(result));

