import json
import time

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
        temperature=0  # TODO: GPT o3 deployment does not support temperature so handle this correctly.
    )

    return response.choices[0].message.content


def create_request(system_prompt, user_prompt, images, request_id, deployment):
    assert isinstance(request_id, str)

    # Create user content.
    user_content = create_user_content(user_prompt=user_prompt, images=images)

    # Create request body.
    body = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    }

    # Only set temperature for non-o3 deployments, o3 doesn't take temperature param.
    if deployment != "o3":
        body["temperature"] = 0

    # Create request.
    request = {
        "custom_id": request_id,
        "method": "POST",
        "url": "/chat/completions",
        "body": body
    }

    return request


def save_requests_as_jsonl(requests, file_name, max_file_size=190 * 1024 * 1024, max_entries_per_file=100_000):
    file_names = []
    curr_file = None

    for request in requests:
        line = json.dumps(request, ensure_ascii=False) + "\n"
        line_size = len(line.encode())

        if curr_file is None or current_file_size + line_size > max_file_size or current_entry_count + 1 > max_entries_per_file:
            if curr_file is not None:
                curr_file.close()

            file_count = len(file_names) + 1
            new_file_name = sys_utils.add_suffix_to_file_path(file_name, f"_{file_count}")
            curr_file = open(new_file_name, "w", encoding="utf-8")
            file_names.append(new_file_name)

            current_file_size = 0
            current_entry_count = 0

        curr_file.write(line)
        current_file_size += line_size
        current_entry_count += 1

    return file_names


def run_batch(input_jsonl, output_jsonl, endpoint, api_key, api_version, check_status_interval_in_sec=60):
    try:
        # Create OpenAI client.
        print("Creating OpenAI client")
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

        # Upload file.
        print("Uploading file")
        with open(input_jsonl, "rb") as file:
            resp = client.files.create(file=file, purpose="batch")
        file_id = resp.id

        # Wait for the file to be processed.
        start_time = time.time()
        while True:
            status = client.files.retrieve(file_id).status.lower()
            elapsed = time.time() - start_time
            print(f"File {file_id} status: {status} (elapsed time: {int(elapsed)} sec)")
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
        start_time = time.time()
        while True:
            info = client.batches.retrieve(batch_id)
            status = info.status.lower()
            elapsed = time.time() - start_time
            print(f"Batch {batch_id} status: {status} (elapsed time: {int(elapsed)} sec)")
            if status in {"failed", "cancelled", "canceled", "expired"}:
                message = info.error.get("message", "No message available") if hasattr(info, "error") else str(info)
                raise RuntimeError(f"Batch processing error: {message}")
            elif status == "completed":
                break
            time.sleep(check_status_interval_in_sec)

        # Get results.
        out_id = client.batches.retrieve(batch_id).output_file_id
        content = client.files.content(out_id).text

        # Save results.
        with open(output_jsonl, "w", encoding="utf-8") as file:
            file.write(content)
    except Exception as e:
        print(f"Run batch error: {str(e)}")
        raise


def delete_files(endpoint, api_key, api_version, force=False, purpose=None):
    # Create OpenAI client.
    print("Creating OpenAI client")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    # List all files.
    if purpose is not None:
        files = client.files.list(purpose=purpose)
    else:
        files = client.files.list()
    file_list = list(files)

    if not file_list:
        print("No files found.")
        return

    # Print files to be deleted.
    print(f"Found {len(file_list)} files to delete:")
    for file in file_list[:10]:  # Show first 10 files.
        print(f"  - {file.id}")

    if len(file_list) > 10:
        print(f"  ... and {len(file_list) - 10} more files")

    # Confirm deletion
    if not force:
        confirmation = input(f"\nAre you sure you want to delete these {len(file_list)} files? (y/n): ")
        if confirmation.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return

    # Delete each file.
    deleted_count = 0
    for file in file_list:
        try:
            client.files.delete(file.id)
            print(f"Deleted file: {file.id}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting file {file.id}: {str(e)}")

    print(f"Successfully deleted {deleted_count} files.")
    return
