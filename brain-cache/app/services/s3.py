"""File storage via S3."""
import json
import os
from io import BytesIO

import boto3

_client = None
_bucket = None


def _get_client():
    global _client, _bucket
    if _client is None:
        region = os.getenv("AWS_REGION", "us-east-1")
        _bucket = os.getenv("S3_BUCKET")
        if not _bucket:
            raise ValueError("S3_BUCKET must be set")
        # If AWS_ACCESS_KEY_ID is not set, boto3 uses the instance role automatically
        key = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        if key and secret:
            _client = boto3.client("s3", region_name=region,
                                   aws_access_key_id=key,
                                   aws_secret_access_key=secret)
        else:
            _client = boto3.client("s3", region_name=region)  # uses instance role
    return _client, _bucket

async def upload_content(source_id: str, content: str, metadata: dict) -> str:
    """Store raw content and metadata in S3; return object key."""
    client, bucket = _get_client()
    key = f"sources/{source_id}.json"
    body = json.dumps({"content": content, "metadata": metadata}).encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=BytesIO(body), ContentType="application/json")
    return key


async def get_content(source_id: str) -> tuple[str, dict]:
    """Retrieve stored content and metadata from S3."""
    client, bucket = _get_client()
    key = f"sources/{source_id}.json"
    resp = client.get_object(Bucket=bucket, Key=key)
    data = json.loads(resp["Body"].read().decode("utf-8"))
    return data["content"], data.get("metadata", {})
