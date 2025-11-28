// florence2.js
import express from "express";
import bodyParser from "body-parser";
import {
    Florence2ForConditionalGeneration,
    AutoProcessor,
    load_image,
} from '@huggingface/transformers';

// -----------------------------
// Global model (load once)
// -----------------------------
const MODEL_ID = "onnx-community/Florence-2-base";

let model = null;
let processor = null;

async function loadModel() {
    console.log("🔄 Loading Florence-2 model...");

    model = await Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
        dtype: "fp16",
    });

    processor = await AutoProcessor.from_pretrained(MODEL_ID);

    console.log("✅ Florence-2 loaded");
}

await loadModel();

// -----------------------------
// Server
// -----------------------------
const app = express();
app.use(bodyParser.json({ limit: "10mb" }));

// Main inference endpoint
app.post("/infer", async (req, res) => {
    try {
        const { image_path, task, expr } = req.body;

        if (!image_path) {
            return res.status(400).json({ error: "Missing image_path" });
        }

        // Load image
        const image = await load_image(image_path);

        // Construct prompts
        const prompts = processor.construct_prompts(task, expr);
        const inputs = await processor(image, prompts);

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
            image.size
        );

        res.json({
            success: true,
            task,
            expr,
            result
        });

    } catch (error) {
        console.error("❌ Inference error:", error);
        res.status(500).json({ error: error.toString() });
    }
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`🚀 Florence2 server running at http://localhost:${PORT}`);
});