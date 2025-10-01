import torch

from utils_ml_inference.dicom import get_dicom_image_fail_safe
from intern_vit import InternVit

from utils_ml_inference.image import numpy_array_to_pil_image

CHECKPOINT_DIR = "/mnt/efs/models/internvl/old/internvl2.5_26b_finetune_lora_20241229_184000_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_no_label/checkpoint-58670"
IMAGE_PATH = "./samples/sample.dcm"


def main():
    # Create model.
    model = InternVit(intern_vl_checkpoint_dir=CHECKPOINT_DIR)
    model.eval()
    model = model.to("cuda")

    # Get image processor.
    image_processor = model.get_image_processor()

    # Preprocess image.
    image = get_dicom_image_fail_safe(IMAGE_PATH, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    image = numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    # Run inference.
    with torch.no_grad():
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
