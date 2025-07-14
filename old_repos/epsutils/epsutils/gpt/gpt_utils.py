import base64
import json
import time
from io import BytesIO

from openai import AzureOpenAI


def encode_image_as_base64(jpeg_byte_stream: BytesIO) -> str:
    return base64.b64encode(jpeg_byte_stream.read()).decode("utf-8")


def create_request(prompt, images, request_id, deployment):
    assert isinstance(request_id, str)

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    if images is not None:
        base64_images = [encode_image_as_base64(jpeg_byte_stream=jpeg_byte_stream) for jpeg_byte_stream in images]

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


def save_requests_as_jsonl(requests, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for request in requests:
            line = json.dumps(request, ensure_ascii=False) + "\n"
            file.write(line)


def run_batch(input_jsonl: str, endpoint: str, api_key: str, api_version: str, check_status_interval_in_sec=60, is_content=False):
    # Create OpenAI client.
    print("Creating OpenAI client")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    # Upload file.
    if is_content:
        print("Uploading content")
        file_stream = BytesIO(input_jsonl.encode("utf-8"))
        file_stream.name = "input_jsonl"
        resp = client.files.create(file=file_stream, purpose="batch")
    else:
        print("Uploading file")
        with open(input_jsonl, "rb") as file:
            resp = client.files.create(file=file, purpose="batch")

    file_id = resp.id

    # Wait for the file to be processed.
    while True:
        status = client.files.retrieve(file_id).status.lower()
        print(f"File {file_id} status: {status}")
        if status == "error":
            raise RuntimeError("File processing error")
        elif status == "processed":
            break
        time.sleep(10)

    # Create batch job.
    print("Creating batch job")
    resp = client.batches.create(input_file_id=file_id, endpoint="/chat/completions", completion_window="24h")
    batch_id = resp.id

    # Wait for the batch to complete.
    while True:
        info = client.batches.retrieve(batch_id)
        status = info.status.lower()
        print(f"Batch {batch_id} status: {status}")
        if status in {"failed", "cancelled", "canceled", "expired"}:
            message = info.error.get("message", "No message available") if hasattr(info, "error") else str(info)
            raise RuntimeError(f"Batch processing error: {message}")
        elif status == "completed":
            break
        time.sleep(check_status_interval_in_sec)

    # Get results.
    out_id = client.batches.retrieve(batch_id).output_file_id
    content = client.files.content(out_id).text

    return content


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
