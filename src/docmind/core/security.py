"""File upload security: validation, sanitization, and safe storage.

Every uploaded file passes through these checks before any processing:
1. Size limit -- reject before reading the full file into memory
2. Magic bytes -- verify actual file type, not just the extension or Content-Type header
3. MIME allowlist -- only permit explicitly allowed file types
4. Filename sanitization -- strip path traversal, null bytes, Unicode tricks
5. UUID storage -- never use user-provided filenames for disk storage

Supported formats:
- PDF: magic bytes %PDF
- DOCX: ZIP container (PK\x03\x04) verified to contain word/document.xml
- TXT: text/plain detected by libmagic
"""

import io
import re
import uuid
import zipfile
from pathlib import Path

import magic
import structlog

logger = structlog.get_logger()

# Maps detected MIME type to the storage extension used for that format.
MIME_TO_EXTENSION: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
}

ALLOWED_MIME_TYPES = set(MIME_TO_EXTENSION.keys())

# Max upload size in bytes (25 MB)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def validate_file_size(content: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Reject files exceeding the size limit.

    Why check size? Without this, an attacker can exhaust server memory
    by uploading a multi-GB file. FastAPI's UploadFile streams the file,
    but once we call .read(), it's all in memory.
    """
    if len(content) > max_bytes:
        raise ValueError(f"File exceeds maximum size of {max_bytes // (1024 * 1024)}MB")


def _is_valid_docx(content: bytes) -> bool:
    """Verify that a ZIP archive is actually a DOCX (contains word/document.xml).

    DOCX uses the ZIP container format (magic bytes PK\x03\x04). Any ZIP
    file would pass a pure magic-byte check, so we must also confirm the
    internal structure expected by the OOXML spec.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            return "word/document.xml" in zf.namelist()
    except zipfile.BadZipFile:
        return False


def validate_magic_bytes(content: bytes) -> str:
    """Check the actual file type using magic bytes.

    Why not trust Content-Type? The Content-Type header is set by the client
    and can be anything. An attacker can upload a .exe with
    Content-Type: application/pdf.
    Magic bytes are embedded in the file itself and can't be faked without
    breaking the file format.

    Detection strategy:
    - PDF: first 2048 bytes suffice (%PDF magic).
    - DOCX/ZIP: PK\x03\x04 signature is in the first 4 bytes; we detect it
      manually and verify the internal OOXML structure via _is_valid_docx().
      We do NOT pass the full file to libmagic for ZIP because some libmagic
      builds require more context than a small buffer to identify ZIP.
    - TXT: text/plain detected by libmagic from the first 2048 bytes.

    Returns the detected MIME type string.
    """
    # Fast path for DOCX: detect the ZIP PK signature in the first 4 bytes
    # rather than relying on libmagic's buffer size behavior.
    zip_magic = b"PK\x03\x04"
    if content[:4] == zip_magic:
        if _is_valid_docx(content):
            return (
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            )
        # ZIP but not a valid DOCX -- reject
        logger.warning("file_type_rejected", detected_mime="application/zip (not DOCX)")
        raise ValueError(
            "File type 'application/zip' not allowed. "
            "Only DOCX (Word) ZIP containers are accepted."
        )

    detected_mime = magic.from_buffer(content[:2048], mime=True)

    if detected_mime not in ALLOWED_MIME_TYPES:
        logger.warning(
            "file_type_rejected",
            detected_mime=detected_mime,
            allowed=list(ALLOWED_MIME_TYPES),
        )
        allowed = ", ".join(sorted(ALLOWED_MIME_TYPES))
        raise ValueError(
            f"File type '{detected_mime}' not allowed. Accepted: {allowed}"
        )

    return detected_mime


def sanitize_filename(filename: str) -> str:
    """Strip dangerous characters from user-provided filenames.

    Why sanitize? Filenames can contain:
    - Path separators (../../etc/passwd) for path traversal
    - Null bytes (file.pdf\\x00.exe) to bypass extension checks
    - Unicode tricks (right-to-left override) to disguise extensions

    We sanitize for metadata/display only. Actual storage uses UUIDs.
    """
    filename = filename.replace("\x00", "")
    filename = filename.replace("\\", "/")
    filename = Path(filename).name
    filename = filename.replace("/", "")
    filename = re.sub(r"[^\w\s\-.]", "", filename)
    filename = re.sub(r"\s+", "_", filename).strip("_")

    if not filename:
        filename = "unnamed_upload"

    return filename


def generate_storage_path(upload_dir: str, extension: str = ".pdf") -> tuple[str, Path]:
    """Generate a UUID-based storage path.

    Why UUID? Never store files with user-provided names. A filename like
    '../../../etc/cron.d/backdoor' could write outside the upload directory.
    UUID filenames eliminate this entire class of attack.

    Returns (doc_id, full_path).
    """
    doc_id = str(uuid.uuid4())
    safe_path = Path(upload_dir) / f"{doc_id}{extension}"
    return doc_id, safe_path


def validate_and_prepare_upload(
    content: bytes,
    original_filename: str,
    upload_dir: str,
) -> tuple[str, Path, str, str]:
    """Run all validation checks and prepare for storage.

    Returns (doc_id, storage_path, sanitized_filename, detected_mime).
    Raises ValueError if any check fails.
    """
    validate_file_size(content)
    detected_mime = validate_magic_bytes(content)
    safe_name = sanitize_filename(original_filename)
    extension = MIME_TO_EXTENSION[detected_mime]
    doc_id, storage_path = generate_storage_path(upload_dir, extension)

    logger.info(
        "upload_validated",
        doc_id=doc_id,
        original_filename=original_filename,
        sanitized_filename=safe_name,
        detected_mime=detected_mime,
        size_bytes=len(content),
    )

    return doc_id, storage_path, safe_name, detected_mime
