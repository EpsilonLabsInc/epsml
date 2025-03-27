import base64
import json
import requests
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from epsutils.dicom import dicom_utils
from epsutils.image import image_utils


class MedImageParseSegmentor:
    def __init__(self, endpoint_url, auth_key, transfer_image_size=(512,512)):
        self.__endpoint_url = endpoint_url
        self.__auth_key = auth_key
        self.__transfer_image_size = transfer_image_size
        self.__input_images = None
        self.__preprocessed_images = None
        self.__encoded_images = None

    def segment(self, images, prompt):
        return self.__segment_impl(images, prompt)

    def segment_and_crop_images(self, images, prompt):
        return self.__segment_and_crop_images_impl(images, prompt)

    def __segment_impl(self, images, prompt):
        assert len(images) > 0

        self.__input_images = []
        self.__preprocessed_images = []
        self.__encoded_images = []

        # Create a list of input images.
        for image in images:
            if isinstance(image, str) and image.endswith(".dcm"):
                self.__input_images.append(self.__dicom_to_image(image))
            elif isinstance(image, str):
                self.__input_images.append(Image.open(image))
            elif isinstance(image, Image.Image):
                self.__input_images.append(image)
            else:
                raise ValueError("Unsupported input image")

        # Preprocess images.
        for image in self.__input_images:
            preprocessed_image = image.resize(self.__transfer_image_size) if self.__transfer_image_size else image
            self.__preprocessed_images.append(preprocessed_image)

        # Encode images.
        for image in self.__preprocessed_images:
            bytes_stream = BytesIO()
            image.save(bytes_stream, format="PNG")
            data = bytes_stream.getvalue()
            bytes_stream.close()
            encoded_image = self.__to_base64(data)
            self.__encoded_images.append(encoded_image)

        request_data = self.__generate_request_data(self.__encoded_images, prompt)
        response = self.__send_request(request_data)
        return response

    def __segment_and_crop_images_impl(self, images, prompt):
        result = self.__segment_impl(images, prompt)

        if result["error_code"]:
            return None

        input_images = result["input_images"]
        segmentation_masks = result["segmentation_masks"]
        cropped_images = []

        for i in range(len(segmentation_masks)):
            segmentation_mask = image_utils.remove_small_components(segmentation_masks[i], 100)
            rel_crop_coordinates = image_utils.min_bounding_rectangle(segmentation_mask, padding_ratio=0.0, return_relative_coordinates=True)
            cropped_image = image_utils.crop_image(image=input_images[i], crop_coordinates=rel_crop_coordinates, use_relative_coordinates=True)
            cropped_images.append(cropped_image)

        assert len(cropped_images) == len(self.__input_images)
        return cropped_images

    def __to_base64(self, data):
        return base64.b64encode(data).decode("utf-8")

    def __from_base64(self, data):
        return base64.b64decode(data)

    def __dicom_to_image(self, dicom_file):
        image = dicom_utils.get_dicom_image(dicom_file, custom_windowing_parameters={"window_center": 0, "window_width": 0})
        image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=False)
        return image

    def __generate_request_data(self, encoded_images, text):
        data = {
            "input_data": {
                "columns": ["image", "text"],
                "index": list(range(len(encoded_images))),
                "data": [[encoded_image, text] for encoded_image in encoded_images]
            }
        }

        return json.dumps(data)

    def __send_request(self, data):
        headers = {"Authorization": f"Bearer {self.__auth_key}", "Content-Type": "application/json"}
        response = requests.post(self.__endpoint_url, headers=headers, data=data)

        if response.status_code != 200:
            return {
                "input_images": self.__input_images,
                "preprocessed_images": self.__preprocessed_images,
                "segmentation_images": None,
                "segmentation_masks": None,
                "error_code": response.status_code,
                "error_text": response.text
            }

        content = response.json()
        segmentation_images = []
        segmentation_masks = []

        for item in content:
            image_features = json.loads(item["image_features"])
            image_data = image_features["data"]
            image_shape = tuple(image_features["shape"][1:])  # Skip batch dimension.
            image_dtype = image_features["dtype"]

            image_bytes = self.__from_base64(image_data)
            image_array = np.frombuffer(image_bytes, dtype=image_dtype)
            image_array = image_array.reshape(image_shape)
            segmentation_mask = image_array > 0
            segmentation_masks.append(segmentation_mask)
            segmentation_image = Image.fromarray(image_array)
            segmentation_images.append(segmentation_image)

            assert len(self.__input_images) == len(self.__preprocessed_images) == len(segmentation_images) == len(segmentation_masks)

        return {
            "input_images": self.__input_images,
            "preprocessed_images": self.__preprocessed_images,
            "segmentation_images": segmentation_images,
            "segmentation_masks": segmentation_masks,
            "error_code": None,
            "error_message": None
        }


if __name__ == "__main__":
    print("Running segmentation example")

    image_paths = ["./samples/covid_1585.png", "./samples/1.dcm"]
    prompt = "segment chest"
    endpoint_url = "https://epsilon-ml-eastus-medimageparse.eastus2.inference.ml.azure.com/score"
    auth_key = "B2CAKaGiUPuTEQ5oAUJq6sPO8uqDlChgONuZgm7XZGMW2o1ycTmwJQQJ99BCAAAAAAAAAAAAINFRAZML4PbT"

    segmentor = MedImageParseSegmentor(endpoint_url=endpoint_url, auth_key=auth_key)
    result = segmentor.segment(images=image_paths, prompt=prompt)

    if result["error_code"]:
        print("Received error response")
        print(f"Error code: {result['error_code']}")
        print(f"Error message: {result['error_message']}")
        exit()

    for index, preprocessed_image in enumerate(result["preprocessed_images"]):
        preprocessed_image.save(f"preprocessed_image_{index}.png")

    for index, segmentation_image in enumerate(result["segmentation_images"]):
        segmentation_image.save(f"segmentation_image_{index}.png")

    print("Input and segmentation images saved")
