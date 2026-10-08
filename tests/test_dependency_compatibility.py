"""Runtime contracts for the frozen dependency export."""

from importlib import metadata
from io import BytesIO
from pathlib import Path

import jwt
from bson import BSON, ObjectId
from multidict import CIMultiDict
from pymongo import MongoClient
from urllib3.response import HTTPResponse
from cryptography.fernet import Fernet
from packaging.requirements import Requirement
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ramjet.tasks.gptchat.llm.embeddings import split_pdf


def test_installed_dependencies_match_export() -> None:
    """test_installed_dependencies_match_export checks pins and active dependency ranges."""
    root = Path(__file__).resolve().parents[1]
    for raw_line in (root / "requirements.txt").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        distribution = metadata.distribution(requirement.name)
        assert distribution.version in requirement.specifier, line
        for dependency_line in distribution.requires or []:
            dependency = Requirement(dependency_line)
            if dependency.marker and not dependency.marker.evaluate({"extra": ""}):
                continue
            assert (
                metadata.version(dependency.name) in dependency.specifier
            ), f"{requirement.name}: {dependency_line}"


def test_pdf_reader_preserves_encrypted_document() -> None:
    """test_pdf_reader_preserves_encrypted_document checks the supported PDF parser API."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=144)
    writer.add_metadata({"/Title": "Ramjet offline PDF"})
    writer.encrypt("offline-password", algorithm="AES-256")
    data = BytesIO()
    writer.write(data)
    data.seek(0)
    reader = PdfReader(data)
    assert reader.is_encrypted
    assert reader.decrypt("offline-password")
    assert len(reader.pages) == 1
    assert reader.metadata.title == "Ramjet offline PDF"
    assert reader.pages[0].mediabox.height == 144


def test_pdf_image_conversion_preserves_pixels() -> None:
    """test_pdf_image_conversion_preserves_pixels checks the PDF-to-image Pillow dependency."""
    original = Image.new("RGB", (16, 16), (24, 48, 72))
    data = BytesIO()
    original.save(data, format="PNG")
    data.seek(0)
    with Image.open(data) as restored:
        restored.load()
        assert restored.size == original.size
        assert restored.getpixel((0, 0)) == (24, 48, 72)


def test_authenticated_encryption_round_trip() -> None:
    """test_authenticated_encryption_round_trip validates cryptography's symmetric API."""
    cipher = Fernet(Fernet.generate_key())
    plaintext = b"Ramjet dependency qualification"
    assert cipher.decrypt(cipher.encrypt(plaintext)) == plaintext


def test_pdf_ingestion_preserves_text_and_source(tmp_path) -> None:
    """test_pdf_ingestion_preserves_text_and_source checks the application's PDF pipeline."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (Ramjet dependency qualification) Tj ET")
    page.replace_contents(stream)
    path = tmp_path / "document.pdf"
    writer.write(path)
    chunks = split_pdf(
        str(path), "offline-document", max_chunks=5, chunk_size=64, chunk_overlap=8
    )
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "Ramjet dependency qualification"
    assert chunks[0].metadata["source"] == "offline-document#page=1?chunk=1"


def test_http_headers_and_streaming_keep_duplicate_values() -> None:
    """test_http_headers_and_streaming_keep_duplicate_values checks HTTP dependencies."""
    headers = CIMultiDict()
    headers.add("X-Trace", "first")
    headers.add("x-trace", "second")
    assert headers.getall("X-TRACE") == ["first", "second"]
    response = HTTPResponse(body=BytesIO(b"abcdef"), preload_content=False)
    assert b"".join(response.stream(amt=2)) == b"abcdef"


def test_bson_and_lazy_database_client_preserve_data() -> None:
    """test_bson_and_lazy_database_client_preserve_data checks supported MongoDB APIs."""
    document = {"_id": ObjectId(), "label": "offline", "nested": {"count": 3}}
    assert BSON.encode(document).decode() == document
    with MongoClient(
        "mongodb://user:pass%40word@localhost:27017/test", connect=False
    ) as client:
        assert client["test"]["notes"].name == "notes"
        assert client.options.pool_options.max_pool_size == 100


def test_padded_jwt_segments_remain_valid() -> None:
    """test_padded_jwt_segments_remain_valid prevents the PyJWT 2.15.0 padding regression."""
    import base64
    import hashlib
    import hmac
    import json

    key = "public-padding-compatibility-key-" * 4
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS512", "typ": "JWT"}).encode()
    )
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "offline-user", "scope": ["read"]}).encode()
    )
    signed = header + b"." + payload
    signature = base64.urlsafe_b64encode(
        hmac.new(key.encode(), signed, hashlib.sha512).digest()
    )
    token = (signed + b"." + signature).decode()
    assert jwt.decode(token, key, algorithms=["HS512"])["sub"] == "offline-user"
