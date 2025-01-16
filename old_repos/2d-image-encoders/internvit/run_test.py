import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

from epsutils.dicom import dicom_utils

MODEL_PATH = "OpenGVLab/InternViT-300M-448px-V2_5"
IMAGE_PATH = "./sample.dcm"


def main():
    # Create the model.
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).cuda().eval()

    # Create image preprocessor.
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_PATH)

    # Preprocess image.
    image = dicom_utils.get_dicom_image(IMAGE_PATH, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    image = Image.fromarray(image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    # Run inference.
    output = model(pixel_values)

    print(f"Input tensor size: {pixel_values.shape}")
    print("Output:")
    print(output)
    print(f"Output type: {type(output)}")
    print(f"Output size: {output.pooler_output.shape}")

    print(f"First element: {output.pooler_output[0, 0]}")
    print(f"Type of first element: {type(output.pooler_output[0, 0])}")
    print(f"Data type of first element: {output.pooler_output[0, 0].dtype}")


if __name__ == "__main__":
    main()
