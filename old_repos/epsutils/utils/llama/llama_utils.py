import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor


def create_model_and_processor(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def run_query(prompt, image, model, processor, max_new_tokens=30):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.decode(output[0])
