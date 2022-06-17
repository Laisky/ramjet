"""upload files to s3
"""

import os
from typing import Generator

import boto3
import botocore
from botocore.exceptions import ClientError
from kipp.utils import setup_logger

from ramjet.settings import S3_SERVER, S3_REGION, S3_BUCKET, S3_KEY, S3_SECRET

logger = setup_logger("s3_uploader")


def upload_file(s3cli, bucket: str, key: str, file_cnt: bytes):
    if not is_file_exists(s3cli, len(file_cnt), bucket, key):
        upload_file_in_mem(s3cli, file_cnt, bucket, key)


def is_file_exists(s3_client, fsize: int, bucket: str, key: str) -> bool:
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        resp_code = response["ResponseMetadata"]["HTTPStatusCode"]
        assert (
            resp_code == 200
        ), f"status code is {response['ResponseMetadata']['HTTPStatusCode']}"
        assert response["ContentLength"] == fsize, "file size mismatch"
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False

        logger.exception(f"head file {bucket}/{key}")

    return False


def _get_s3_fname(root, file_name):
    return file_name[len(root) :].lstrip("/")


def connect_s3(
    server: str, region: str, access_key: str, secret_key: str
) -> boto3.client:
    b3_session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    return b3_session.client("s3", endpoint_url=server)


def _list_all_files(dir) -> Generator:
    for root, dirs, files in os.walk(dir):
        for file in files:
            yield os.path.join(root, file)


def upload_file_in_mem(s3_client, file_cnt: bytes, bucket: str, key: str):
    """Upload a file in memory(without disk) to an S3 bucket"""

    try:
        response = s3_client.put_object(Body=file_cnt, Bucket=bucket, Key=key)
        logger.info(f"uploaded {bucket}/{key}")
    except ClientError as e:
        logger.exception(f"upload file {bucket}/{key}")
        raise e


if __name__ == "__main__":
    s3_cli = connect_s3(
        S3_SERVER,
        S3_REGION,
        S3_KEY,
        S3_SECRET,
    )
    with open(r"/home/laisky/repo/laisky/ramjet/README.md", "rb") as f:
        upload_file(s3_cli, "test", "README.md", f.read())
