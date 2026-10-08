"""Runtime contracts for the frozen dependency export."""

from importlib import metadata
from io import BytesIO
from pathlib import Path

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
