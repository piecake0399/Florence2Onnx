// florence2.js
import { loadFlorence2, runFlorence2 } from "./florence2_module.js";
import { load_image } from "@huggingface/transformers";

const imagePath = process.argv[2];
const task = process.argv[3];
const expr = process.argv[4];

if (!imagePath) {
  console.error("❌ Missing image path");
  process.exit(1);
}

try {
  const image = await load_image(imagePath);

  await loadFlorence2();
  const result = await runFlorence2(image, task, expr);

  console.log(JSON.stringify(result));
} catch (err) {
  console.error("❌ Inference error:", err);
  process.exit(1);
}
