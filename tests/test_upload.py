import base64
import io

import pytest
from PIL import Image

from backend.routes.api import NO_TEXT_REPLY
from backend.services.documents import MAX_SCANNED_PAGES, check_ranges
from backend.utils import file_utils
from conftest import get_history, reply_text, stream_events


def make_text_pdf(text):
    """Build a minimal one-page PDF with a real text layer."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref))
    return out.getvalue()


def make_image(fmt="PNG", size=(400, 300)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, fmt)
    return buffer.getvalue()


def make_scanned_pdf(pages):
    buffer = io.BytesIO()
    images = [Image.new("RGB", (300, 400), "white") for _ in range(pages)]
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def upload(client, headers, name, data, message=None):
    form = {"message": message} if message else {}
    return client.post("/api/upload", files={"file": (name, data)}, data=form, headers=headers)


@pytest.mark.parametrize("name, data", [
    ("report.exe", b"MZ..."),
    ("../../backend/main.py", b"print('hi')"),
    ("scan.png", b"not really a png"),
    ("report.pdf", b"not really a pdf"),
    ("empty.txt", b""),
])
def test_upload_rejects_unsupported_or_unreadable_files(client, auth, fake_llm, name, data):
    response = upload(client, auth, name, data)
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)
    assert fake_llm.chats == []


def test_upload_rejects_oversized_files(client, auth, fake_llm, monkeypatch):
    monkeypatch.setattr(file_utils, "MAX_UPLOAD_BYTES", 10)
    assert upload(client, auth, "report.txt", b"x" * 11).status_code == 413


def test_text_file_is_analyzed_and_saved(client, auth, fake_llm):
    report = b"Hemoglobin 10.2 g/dL 13.5 - 17.5"
    events = stream_events(upload(client, auth, "report.txt", report, "Is this low?"))
    assert {"type": "status", "text": "Analyzing the report…"} in events
    assert reply_text(events) == "Model reply."
    assert fake_llm.transcribed == []

    prompt = fake_llm.chats[0][1]["content"]
    assert "Hemoglobin 10.2 g/dL 13.5 - 17.5" in prompt and prompt.endswith("Is this low?")
    assert "- Hemoglobin: 10.2 is LOW (reference 13.5-17.5)" in prompt

    item = get_history(client, auth)[0]
    assert item["user_message"] == "[Uploaded File: report.txt]\nQuery: Is this low?"
    assert item["assistant_reply"] == "Model reply."


def test_pdf_text_layer_is_used_directly(client, auth, fake_llm):
    pdf = make_text_pdf("Hemoglobin 10.2 g/dL  Platelets 150000 per microliter")
    stream_events(upload(client, auth, "labs.pdf", pdf))
    assert fake_llm.transcribed == []
    assert "Platelets 150000" in fake_llm.chats[0][1]["content"]


def test_image_is_read_by_the_vision_model_first(client, auth, fake_llm):
    events = stream_events(upload(client, auth, "photo.webp", make_image("WEBP", size=(4000, 3000))))
    assert {"type": "status", "text": "Reading the document…"} in events

    [images] = fake_llm.transcribed
    image = Image.open(io.BytesIO(base64.b64decode(images[0])))
    assert image.format == "JPEG" and max(image.size) == 1600  # large photos are scaled down
    assert fake_llm.transcript in fake_llm.chats[0][1]["content"]


def test_scanned_pdf_pages_are_read_by_the_vision_model(client, auth, fake_llm):
    stream_events(upload(client, auth, "scan.pdf", make_scanned_pdf(pages=5)))
    assert len(fake_llm.transcribed[0]) == MAX_SCANNED_PAGES


def test_upload_without_readable_text(client, auth, fake_llm):
    fake_llm.transcript = ""
    events = stream_events(upload(client, auth, "blank.png", make_image()))
    assert reply_text(events) == NO_TEXT_REPLY
    assert fake_llm.chats == []


def test_check_ranges_compares_values_with_reference_ranges():
    text = "\n".join([
        "Report date: 01-12-2025",
        "Patient: Sample Age: 45",
        "Hemoglobin 10.8 g/dL 13.5 - 17.5",
        "WBC Count 11.9 x10^3/uL 4.0 - 11.0",
        "Serum Creatinine: 0.9 mg/dL 0.7 to 1.3",
        "Vitamin D, 25-OH 18 ng/mL 30 - 100",  # ambiguous: skipped rather than misread
        "Cholesterol 180 mg/dL < 200",
    ])
    assert check_ranges(text) == [
        "Hemoglobin: 10.8 is LOW (reference 13.5-17.5)",
        "WBC Count: 11.9 is HIGH (reference 4.0-11.0)",
        "Serum Creatinine: 0.9 is within range (reference 0.7-1.3)",
    ]


def test_upload_query_goes_through_safety_checks(client, auth, fake_llm):
    events = stream_events(upload(client, auth, "report.txt", b"Hemoglobin 10.2", "I want to end my life"))
    assert events[0] == {"type": "meta", "blocked": True}
    assert fake_llm.chats == []

    events = stream_events(upload(client, auth, "ecg.txt", b"Sinus rhythm", "I have sudden chest pain, is this normal?"))
    assert events[0] == {"type": "meta", "redflag": True}
    assert reply_text(events).startswith("⚠️")
