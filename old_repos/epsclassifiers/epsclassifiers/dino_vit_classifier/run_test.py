import torch
from PIL import Image

from epsutils.dicom import dicom_utils
from epsclassifiers.dino_vit_classifier import DinoVitClassifier

CHECKPOINT = "/home/andrej/work/2d-image-encoders/dino_vit/data/model_0479999.pth"
IMAGE_PATH = "./samples/sample.dcm"


def main():
    classifier = DinoVitClassifier(num_classes=14, dino_vit_checkpoint=CHECKPOINT)
    classifier = classifier.to("cuda")
    image_processor = classifier.get_image_processor()

    # Preprocess image.
    image = dicom_utils.get_dicom_image(IMAGE_PATH, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    image = Image.fromarray(image).convert("RGB")
    pixel_values = image_processor(images=[image, image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.cuda()

    # Run inference.
    output = classifier(pixel_values)

    print(f"Input tensor size: {pixel_values.shape}")
    print("Output:")
    print(output)
    print(f"Output type: {type(output)}")
    print(f"Output size: {output.shape}")


if __name__ == "__main__":
    main()
