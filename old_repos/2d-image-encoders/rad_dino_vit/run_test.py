import torch
from PIL import Image

from epsutils.dicom import dicom_utils
from rad_dino_vit import RadDinoVit

IMAGE_PATH = "./samples/sample.dcm"


def main():
    # Create model.
    model = RadDinoVit()
    model.eval()
    model = model.to("cuda")

    # Get image processor.
    image_processor = model.get_image_processor()

    # Preprocess image.
    image = dicom_utils.get_dicom_image(IMAGE_PATH, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    image = Image.fromarray(image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.cuda()

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
