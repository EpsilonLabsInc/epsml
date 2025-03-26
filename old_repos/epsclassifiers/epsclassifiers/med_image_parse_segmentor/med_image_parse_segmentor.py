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

    def segment(self, image, prompt):
        return self.__segment_impl(image, prompt)

    def __segment_impl(self, image, prompt):
        if isinstance(image, str):
            image = self.__dicom_to_bytes_stream(image) if image.endswith(".dcm") else open(image, "rb").read()

        encoded_image = self.__to_base64(image)
        request_data = self.__generate_request_data(encoded_image, prompt)
        response = self.__send_request(request_data)
        return response

    def __to_base64(self, data):
        return base64.b64encode(data).decode("utf-8")

    def __from_base64(self, data):
        return base64.b64decode(data)

    def __dicom_to_bytes_stream(self, dicom_file):
        image = dicom_utils.get_dicom_image(dicom_file, custom_windowing_parameters={"window_center": 0, "window_width": 0})

        if self.__transfer_image_size:
            image = cv2.resize(image, self.__transfer_image_size, interpolation=cv2.INTER_LINEAR)

        image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=False)
        bytes_stream = BytesIO()
        image.save(bytes_stream, format="PNG")
        data = bytes_stream.getvalue()
        bytes_stream.close()

        return data

    def __generate_request_data(self, encoded_image, text):
        data = {
            "input_data": {
                "columns": ["image", "text"],
                "index": [0],
                "data": [[encoded_image, text]]
            }
        }

        return json.dumps(data)

    def __send_request(self, data):
        headers = {"Authorization": f"Bearer {self.__auth_key}", "Content-Type": "application/json"}
        response = requests.post(self.__endpoint_url, headers=headers, data=data)

        if response.status_code != 200:
            return {"image": None, "error_code": response.status_code, "error_text": response.text}

        content = response.json()
        image_features = json.loads(content[0]["image_features"])
        image_data = image_features["data"]
        image_shape = tuple(image_features["shape"][1:])  # Skip batch dimension.
        image_dtype = image_features["dtype"]

        image_bytes = self.__from_base64(image_data)
        image_array = np.frombuffer(image_bytes, dtype=image_dtype)
        image_array = image_array.reshape(image_shape)
        image = Image.fromarray(image_array)

        return {"image": image, "error_code": None, "error_message": None}


if __name__ == "__main__":
    print("Running segmentation example")

    # image_path = "./samples/covid_1585.png"
    image_path = "./samples/1.dcm"
    output_file_name = "segmentation_results.png"
    prompt = "segment chest"
    endpoint_url = "https://epsilon-ml-eastus-medimageparse.eastus2.inference.ml.azure.com/score"
    auth_key = "B2CAKaGiUPuTEQ5oAUJq6sPO8uqDlChgONuZgm7XZGMW2o1ycTmwJQQJ99BCAAAAAAAAAAAAINFRAZML4PbT"

    segmentor = MedImageParseSegmentor(endpoint_url=endpoint_url, auth_key=auth_key)
    result = segmentor.segment(image=image_path, prompt=prompt)

    if result["error_code"]:
        print("Received error response")
        print(f"Error code: {result['error_code']}")
        print(f"Error message: {result['error_message']}")
        exit()

    image = result["image"]
    image.save(output_file_name)
    print(f"Image saved as {output_file_name}")
