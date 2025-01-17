import torch
from PIL import Image

from epsutils.dicom import dicom_utils
from epsclassifiers.intern_vit_classifier import InternVitClassifier

CHECKPOINT_DIR = "/mnt/training/internvl2.5_8b_finetune_lora_20241226_205132_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_tiles_6_no_labels/checkpoint-58670"
IMAGE_PATH = "./samples/sample.dcm"


def main():
    classifier = InternVitClassifier(num_classes=14, intern_vl_checkpoint_dir=CHECKPOINT_DIR)
    classifier = classifier.to("cuda")
    image_processor = classifier.get_image_processor()

    # Preprocess image.
    image = dicom_utils.get_dicom_image(IMAGE_PATH, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    image = Image.fromarray(image).convert("RGB")
    pixel_values = image_processor(images=[image, image], return_tensors="pt").pixel_values
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
