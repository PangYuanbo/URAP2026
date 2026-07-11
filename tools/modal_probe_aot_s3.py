from __future__ import annotations

import json

import modal


app = modal.App("urap-aot-s3-probe-v1")
image = modal.Image.debian_slim(python_version="3.11").pip_install("boto3==1.39.4")


@app.function(image=image, timeout=600)
def probe(flight_id: str = "") -> dict:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    bucket = "airborne-obj-detection-challenge-training"
    prefix = f"part1/Images/{flight_id}/" if flight_id else "part1/"
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = client.get_paginator("list_objects_v2")
    pages = list(paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/" if not flight_id else ""))
    contents = [item for page in pages for item in page.get("Contents", [])]
    result = {
        "bucket": bucket,
        "prefix": prefix,
        "common_prefixes": [item["Prefix"] for page in pages for item in page.get("CommonPrefixes", [])],
        "objects": [
            {"key": item["Key"], "size": item["Size"], "etag": item.get("ETag")}
            for item in contents[:30]
        ],
        "key_count": len(contents),
        "total_bytes": sum(item["Size"] for item in contents),
        "truncated": any(page.get("IsTruncated") for page in pages),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(flight_id: str = "") -> None:
    print(json.dumps(probe.remote(flight_id), indent=2))
