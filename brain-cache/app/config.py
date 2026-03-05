# app/config.py
import os

import boto3


def get_secrets() -> dict:
    # In local dev, fall back to environment variables
    if os.getenv("ENV") == "local":
        return {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
            "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
        }
    ssm = boto3.client("ssm", region_name="us-east-1")
    response = ssm.get_parameters(
        Names=[
            "/brain-cache/prod/OPENAI_API_KEY",
            "/brain-cache/prod/PINECONE_API_KEY",
            "/brain-cache/prod/PINECONE_INDEX_NAME",
        ],
        WithDecryption=True,  # decrypts SecureString values
    )
    return {
        p["Name"].split("/")[-1]: p["Value"]
        for p in response["Parameters"]
    }
