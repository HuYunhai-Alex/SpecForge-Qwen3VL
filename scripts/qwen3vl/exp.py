from qwen3vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwen3vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers import AutoImageProcessor, AutoTokenizer

import os
curr_path = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_TOKEN"] = "xxxx"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "nexa4ai/qwen3vl-4B", torch_dtype="auto", device_map="auto"
)

# Manual instantiation without video processor
tokenizer = AutoTokenizer.from_pretrained("nexa4ai/qwen3vl-4B")
image_processor = AutoImageProcessor.from_pretrained("nexa4ai/qwen3vl-4B")

processor = Qwen3VLProcessor(
    image_processor=image_processor,
    tokenizer=tokenizer,
    video_processor=None,
)

# Skip chat template and use direct text with image tokens
text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Tell me about this image.<|im_end|>\n<|im_start|>assistant\n"

# Process the image
from PIL import Image
import requests

image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(image_url, stream=True).raw)
# image = Image.open(os.path.join(curr_path, "modelfiles", "receipt.png"))

inputs = processor(
    text=[text],
    images=[image],
    videos=None,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.tokenizer.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("\033[1;32m" + str(output_text) + "\033[0m")
