import pytest

from docmind.core.security import (
    sanitize_filename,
    validate_file_size,
    validate_magic_bytes,
)


class TestFileSizeValidation:
    def test_valid_size(self):
        content = b"x" * 1000
        validate_file_size(content)  # should not raise

    def test_oversized_file(self):
        content = b"x" * (26 * 1024 * 1024)  # 26 MB
        with pytest.raises(ValueError, match="exceeds maximum size"):
            validate_file_size(content)

    def test_custom_limit(self):
        content = b"x" * 100
        with pytest.raises(ValueError):
            validate_file_size(content, max_bytes=50)


class TestMagicByteValidation:
    def test_valid_pdf(self, sample_pdf_bytes):
        mime = validate_magic_bytes(sample_pdf_bytes)
        assert mime == "application/pdf"

    def test_plain_text_now_allowed(self):
        # TXT is a supported format since Issue #6; text/plain should be accepted.
        mime = validate_magic_bytes(b"This is just plain text")
        assert mime == "text/plain"

    def test_reject_html(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_magic_bytes(b"<html><body>sneaky</body></html>")

    def test_reject_exe_disguised_as_pdf(self):
        # PE/COFF executable (MZ header) -- libmagic should detect application/x-dosexec
        with pytest.raises(ValueError):
            validate_magic_bytes(b"MZ\x90\x00" + b"\x00" * 200)


class TestFilenameSanitization:
    def test_normal_filename(self):
        assert sanitize_filename("report.pdf") == "report.pdf"

    def test_path_traversal(self):
        result = sanitize_filename("../../etc/passwd")
        assert "/" not in result
        assert ".." not in result

    def test_null_bytes(self):
        result = sanitize_filename("file.pdf\x00.exe")
        assert "\x00" not in result

    def test_empty_filename(self):
        assert sanitize_filename("") == "unnamed_upload"

    def test_windows_path(self):
        result = sanitize_filename("C:\\Users\\hacker\\malware.pdf")
        assert "\\" not in result
        assert "malware.pdf" == result

    def test_spaces_normalized(self):
        result = sanitize_filename("my   report   file.pdf")
        assert "  " not in result
