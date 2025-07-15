import json
import time
from io import BytesIO

from openai import AzureOpenAI

from epsutils.sys import sys_utils

from ._internal import create_user_content


def run_single_query(system_prompt, user_prompt, images, endpoint, api_key, api_version, deployment):
    # Create OpenAI client.
    print("Creating OpenAI client")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    # Create user content.
    user_content = create_user_content(user_prompt=user_prompt, images=images)

    # Send request.
    print("Sending request")
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0
    )

    return response.choices[0].message.content


def create_request(system_prompt, user_prompt, images, request_id, deployment):
    assert isinstance(request_id, str)

    # Create user content.
    user_content = create_user_content(user_prompt=user_prompt, images=images)

    # Create request.
    request = {
        "custom_id": request_id,
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": deployment,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0,
        }
    }

    return request


def save_requests_as_jsonl(requests,
                           output_file_name,
                           max_file_size=190 * 1024 * 1024,
                           max_entries_per_file=100_000):
    output_file_names = []
    curr_file = None

    for request in requests:
        line = json.dumps(request, ensure_ascii=False) + "\n"
        line_size = len(line.encode())

        if curr_file is None or current_file_size + line_size > max_file_size or current_entry_count + 1 > max_entries_per_file:
            if curr_file is not None:
                curr_file.close()

            file_count = len(output_file_names) + 1
            new_output_file_name = sys_utils.add_suffix_to_file_path(output_file_name, f"_{file_count}")
            curr_file = open(new_output_file_name, "w", encoding="utf-8")
            output_file_names.append(new_output_file_name)

            current_file_size = 0
            current_entry_count = 0

        curr_file.write(line)
        current_file_size += line_size
        current_entry_count += 1

    return output_file_names


def run_batch(input_jsonl, endpoint, api_key, api_version, check_status_interval_in_sec=60, is_content=False):
    # Create OpenAI client.
    print("Creating OpenAI client")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    # Upload file.
    if is_content:
        print("Uploading content")
        file_stream = BytesIO(input_jsonl.encode("utf-8"))
        file_stream.name = "input.jsonl"
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
