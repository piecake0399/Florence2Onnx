// florence2_module.js
import {
    Florence2ForConditionalGeneration,
    AutoProcessor,
} from '@huggingface/transformers';

const MODEL_ID = "onnx-community/Florence-2-base";

let model = null;
let processor = null;

export async function loadFlorence2() {
    if (model && processor) return { model, processor };

    console.log("🔄 Loading Florence-2 model...");

    model = await Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
        dtype: "q8",
    });

    processor = await AutoProcessor.from_pretrained(MODEL_ID);

    console.log("✅ Florence-2 loaded");

    return { model, processor };
}

export async function runFlorence2(pilImage, task, expr) {
    if (!model || !processor) {
        await loadFlorence2();
    }

    const prompts = processor.construct_prompts(task, expr);
    const inputs = await processor(pilImage, prompts);

    const generated_ids = await model.generate({
        ...inputs,
        max_new_tokens: 200,
    });

   const generated_text = processor.batch_decode(
        generated_ids,
        { skip_special_tokens: false }
    )[0];

    return processor.post_process_generation(
        generated_text,
        task,
        pilImage.size
    );
}
