import base64
import json
from io import BytesIO


def encode_image_as_base64(jpeg_byte_stream: BytesIO) -> str:
    return base64.b64encode(jpeg_byte_stream.read()).decode("utf-8")


def create_gpt_request(prompt, images, request_id, deployment):
    base64_images = [encode_image_as_base64(jpeg_byte_stream=jpeg_byte_stream) for jpeg_byte_stream in images]

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    for base64_image in base64_images:
        content.append(
            {
                "type": "image_url",
                "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" }
            }
        )

    request = {
        "custom_id": request_id,
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": deployment,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": 0,
        }
    }

    return request


def find_content(response):
    if isinstance(response, dict):
        for key, value in response.items():
            if key == 'content':
                content = json.loads(value.strip("```json\\n"))
                return content
            else:
                result = find_content(value)
                if result:
                    return result
    elif isinstance(response, list):
        for item in response:
            result = find_content(item)
            if result:
                return result

    return None
