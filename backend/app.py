"""
Hugging Face Spaces entry point.
HF Spaces with SDK=docker runs this via the Dockerfile.
This file is kept as a thin alias so both `uvicorn app:app` 
and `uvicorn main:app` work correctly.
"""
from main import app  # noqa: F401
