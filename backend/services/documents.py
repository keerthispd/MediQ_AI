"""Turn uploaded medical documents into text or page images the local model can read."""
import base64
import io
import re
from dataclasses import dataclass, field
from typing import List

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_DOCUMENT_CHARS = 12000  # about 3,000 tokens, leaving room in the model's context for the reply
MAX_SCANNED_PAGES = 3
MAX_IMAGE_SIDE = 1600
MIN_PDF_TEXT_CHARS = 50  # a PDF with less text than this is treated as a scan


class DocumentError(ValueError):
    """The upload could not be read. The message is safe to show to users."""


@dataclass
class Document:
    text: str = ""
    # Base64 JPEGs for scans and photos, which need the vision model to read them
    images: List[str] = field(default_factory=list)


def _encode_image(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")  # honour phone camera rotation
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _read_image(data: bytes) -> Document:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return Document(images=[_encode_image(image)])
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise DocumentError("This image could not be read. Please upload a clear PNG, JPG or WEBP file.") from e


def _read_pdf(data: bytes) -> Document:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        raise DocumentError("This PDF could not be read. It may be damaged or password-protected.") from e
    try:
        pages = []
        for index in range(len(pdf)):
            text = pdf[index].get_textpage().get_text_bounded().strip()
            if text:
                pages.append(text)
        text = "\n\n".join(pages)
        if len(text) >= MIN_PDF_TEXT_CHARS:
            return Document(text=text)
        # No text layer: render the first pages so the vision model can read them
        images = [_encode_image(pdf[i].render(scale=2).to_pil()) for i in range(min(len(pdf), MAX_SCANNED_PAGES))]
        return Document(images=images)
    finally:
        pdf.close()


def read_document(data: bytes, extension: str) -> Document:
    """Read an upload whose type was already validated (see backend.utils.file_utils)."""
    if extension == ".pdf":
        return _read_pdf(data)
    if extension == ".txt":
        return Document(text=data.decode("utf-8", errors="replace"))
    return _read_image(data)


# A lab result row: "<test name> <value> [units] <low> - <high>", e.g. "WBC Count 11.9 x10^3/uL 4.0 - 11.0".
# The test name can't contain digits and the value must be a standalone number, so dates ("01-12-2025")
# and names with numbers ("Vitamin D, 25-OH") are skipped rather than misread.
_LAB_ROW = re.compile(
    r"^\s*(?P<name>[A-Za-z][^\d\n]{0,40}?)[\s:]+(?P<value>\d+(?:\.\d+)?)(?=[\s%]|$)"
    r"[^\n]*?(?P<low>\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(?P<high>\d+(?:\.\d+)?)",
    re.MULTILINE,
)


def check_ranges(text: str) -> List[str]:
    """Compare lab values with their reference ranges.

    Small language models often misjudge numbers (e.g. calling 11.9 normal for a 4.0-11.0 range),
    so the comparison is done here and given to the model.
    """
    checks = []
    for match in _LAB_ROW.finditer(text):
        value, low, high = (float(match[k]) for k in ("value", "low", "high"))
        if low >= high:
            continue
        status = "HIGH" if value > high else "LOW" if value < low else "within range"
        name = match["name"].strip(" :,")
        checks.append(f"{name}: {match['value']} is {status} (reference {match['low']}-{match['high']})")
    return checks


def limit_text(text: str) -> str:
    """Trim document text to what fits in the model's context."""
    text = text.strip()
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    return text[:MAX_DOCUMENT_CHARS] + "\n[Document truncated: only the first part was analyzed.]"
