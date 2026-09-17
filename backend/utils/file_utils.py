import os
from typing import Optional

from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# File types backend.services.documents knows how to read
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"}


def get_upload_extension(filename: Optional[str]) -> str:
    """Return the lower-case extension of an allowed upload, or raise a 400 for anything else."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF, an image (PNG, JPG, WEBP) or a text file.",
        )
    return ext


def read_upload(upload_file: UploadFile) -> bytes:
    """Read an upload into memory, rejecting empty or oversized files.

    Medical reports are only needed while they are analyzed, so they are never saved.
    """
    data = upload_file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File is too large. The maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return data
