import base64
from io import BytesIO


def encode_image_as_base64(jpeg_byte_stream: BytesIO) -> str:
    return base64.b64encode(jpeg_byte_stream.read()).decode("utf-8")


def create_user_content(user_prompt, images):
    user_content = [{"type": "text", "text": user_prompt}]

    if images is not None:
        base64_images = [encode_image_as_base64(jpeg_byte_stream=image) for image in images]

        for base64_image in base64_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    return user_content
