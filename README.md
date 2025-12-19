# Florence2onnx
ONNX deployment for Florence 2

1. Install the dependencies: ``` pip install -r requirements.txt ```

   1.1 Install ```pip install onnxruntime-gpu``` for cuda execution (uninstall the onnxruntime)

2. Download the ONNX weight files from the following link: https://huggingface.co/onnx-community/Florence-2-base-ft/tree/main/onnx

   2.1. Copy the weight files to the weight_files folder (5 weight files are needed. Vision Encoder, Embed Tokens, Encoder Model, Decoder Model, and Decoder Model Merged).

   For example: vision_encoder_q4f16.onnx, embed_tokens_q4f16.onnx, encoder_model_q4f16.onnx, decoder_model_q4f16.onnx, decoder_model_merged_q4.onnx

4. Run the following command ``` python fl2.py ``` for <CAPTION_TO_PHRASE_GROUNDING> benchmark on RefCOCO 2014 dataset
   4.1. Run ``` python fl2_caption.py ``` for <CAPTION> benchmark on MSCOCO 2014 Captioning dataset


---------

For a more detailed caption, you can set the image size to (768,768) in the preprocessor_config.json file that is located in the processor_files folder. The image size is a trade-off between accuracy and speed. The model will generate better and extended captions but the speed is the cost!
