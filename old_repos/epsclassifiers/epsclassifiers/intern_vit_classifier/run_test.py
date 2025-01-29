import torch
from PIL import Image

from epsutils.dicom import dicom_utils
from epsclassifiers.intern_vit_classifier import InternVitClassifier

CHECKPOINT_DIR = "/workspace/models/internvl2.5_26b_finetune_lora_20241229_184000_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_no_label/checkpoint-58670"
IMAGE_PATH = "./samples/sample.dcm"
USE_LARGER_MODEL = True
USE_TILES = True


def main():
    intern_vit_output_dim = 3200 if USE_LARGER_MODEL else 1024  # 3200 for InternVL 26B model and 1024 for InternVL 8B model.

    classifier = InternVitClassifier(num_classes=14, intern_vl_checkpoint_dir=CHECKPOINT_DIR, intern_vit_output_dim=intern_vit_output_dim, use_tiles=USE_TILES)
    classifier = classifier.to("cuda")
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # Image processor.
    image_processor = classifier.get_tile_splitting_image_processor() if USE_TILES else classifier.get_image_processor()

    # Preprocess image.
    image = dicom_utils.get_dicom_image(IMAGE_PATH, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    image = Image.fromarray(image).convert("RGB")
    pixel_values = image_processor(images=[image, image], return_tensors="pt") if USE_TILES else image_processor(images=[image, image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    # Run inference.
    output = classifier(pixel_values)

    print(f"Input tensor size: {pixel_values.shape}")
    print("Output:")
    print(output)
    print(f"Output type: {type(output)}")
    print(f"Output size: {output.shape}")


if __name__ == "__main__":
    main()
