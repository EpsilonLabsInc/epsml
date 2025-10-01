import boto3


def split_aws_s3_uri(aws_s3_uri):
    scheme = "s3://"

    if not aws_s3_uri.startswith(scheme):
        raise ValueError(f"AWS S3 URI {aws_s3_uri} is missing {scheme} prefix")

    parts = aws_s3_uri.replace(scheme, "").split("/", 1)
    return {"scheme:": scheme, "aws_s3_bucket_name": parts[0], "aws_s3_path": parts[1]}


def is_aws_s3_uri(aws_s3_uri):
    try:
        aws_s3_data = split_aws_s3_uri(aws_s3_uri)
        return True
    except:
        return False


def download_file_as_string(aws_s3_bucket_name: str, aws_s3_file_name: str) -> str:
    """
    Downloads a file from AWS S3 bucket and returns its content as a string.
    """

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=aws_s3_bucket_name, Key=aws_s3_file_name)
    return response["Body"].read().decode("utf-8")


def download_file_as_bytes(aws_s3_bucket_name: str, aws_s3_file_name: str) -> bytes:
    """
    Downloads a file from AWS S3 bucket and returns its content as bytes.
    """

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=aws_s3_bucket_name, Key=aws_s3_file_name)
    return response["Body"].read()
