"""
Unit tests for validators module.
"""
import pytest
from pathlib import Path
import json
from backend.validators import validate_upload, sanitize_filename, ValidationError


class TestValidateUpload:
    """Test file upload validation."""

    def test_valid_json_file(self, tmp_path):
        """Test validation of valid JSON file."""
        file_path = tmp_path / "valid.json"
        file_path.write_text('{"test": "data"}')

        is_valid, error = validate_upload(file_path)

        assert is_valid
        assert error == ""

    def test_valid_json_with_unicode(self, tmp_path):
        """Test validation of JSON with unicode characters."""
        file_path = tmp_path / "unicode.json"
        file_path.write_text('{"test": "データ 🎯"}', encoding='utf-8')

        is_valid, error = validate_upload(file_path)

        assert is_valid
        assert error == ""

    def test_file_not_found(self):
        """Test validation of non-existent file."""
        is_valid, error = validate_upload(Path("/nonexistent/file.json"))

        assert not is_valid
        assert "not found" in error.lower()

    def test_file_too_large(self, tmp_path):
        """Test validation of oversized file."""
        file_path = tmp_path / "large.json"
        # Create 150MB file (over 100MB limit)
        large_data = {"data": "x" * (150 * 1024 * 1024)}
        file_path.write_bytes(json.dumps({"x": "y"}).encode() * (75 * 1024 * 1024))

        is_valid, error = validate_upload(file_path, max_size_mb=100)

        assert not is_valid
        assert "too large" in error.lower()
        assert "100" in error  # Should mention limit

    def test_file_size_custom_limit(self, tmp_path):
        """Test validation with custom size limit."""
        file_path = tmp_path / "medium.json"
        # Create 5MB file
        file_path.write_bytes(b"x" * (5 * 1024 * 1024))

        # Should fail with 2MB limit
        is_valid, error = validate_upload(file_path, max_size_mb=2)
        assert not is_valid
        assert "too large" in error.lower()

        # Should pass with 10MB limit
        is_valid, error = validate_upload(file_path, max_size_mb=10)
        # Will fail due to invalid JSON, but size check passes
        assert "too large" not in error.lower()

    def test_invalid_json(self, tmp_path):
        """Test validation of malformed JSON."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not json{{{")

        is_valid, error = validate_upload(file_path)

        assert not is_valid
        assert "invalid json" in error.lower()

    def test_invalid_json_syntax_error(self, tmp_path):
        """Test validation of JSON with syntax errors."""
        file_path = tmp_path / "bad_syntax.json"
        file_path.write_text('{"key": "value",}')  # Trailing comma

        is_valid, error = validate_upload(file_path)

        assert not is_valid
        assert "invalid json" in error.lower()

    def test_invalid_utf8(self, tmp_path):
        """Test validation of file with invalid UTF-8."""
        file_path = tmp_path / "binary.json"
        file_path.write_bytes(b"\xff\xfe\xfd")

        is_valid, error = validate_upload(file_path)

        assert not is_valid
        assert "utf-8" in error.lower()

    def test_empty_json_file(self, tmp_path):
        """Test validation of empty JSON file."""
        file_path = tmp_path / "empty.json"
        file_path.write_text("")

        is_valid, error = validate_upload(file_path)

        assert not is_valid
        assert "invalid json" in error.lower()

    def test_empty_json_object(self, tmp_path):
        """Test validation of valid empty JSON object."""
        file_path = tmp_path / "empty_obj.json"
        file_path.write_text("{}")

        is_valid, error = validate_upload(file_path)

        assert is_valid
        assert error == ""


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_basic_filename(self):
        """Test sanitization of clean filename."""
        result = sanitize_filename("test.json")
        assert result == "test.json"

    def test_filename_with_spaces(self):
        """Test sanitization of filename with spaces."""
        result = sanitize_filename("my file.json")
        # Spaces should be replaced with underscores
        assert result == "my_file.json"

    def test_path_traversal_attempt(self):
        """Test sanitization of path traversal attack."""
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        assert "etc" in result  # Filename part preserved

    def test_windows_path_separator(self):
        """Test sanitization of Windows path separators."""
        result = sanitize_filename("C:\\Windows\\System32\\config.json")
        assert "\\" not in result
        assert ":" not in result
        assert "config.json" in result

    def test_null_bytes(self):
        """Test sanitization of null bytes."""
        result = sanitize_filename("test\x00.json")
        assert "\x00" not in result
        assert "test" in result

    def test_special_characters(self):
        """Test sanitization of special characters."""
        result = sanitize_filename("test@#$%.json")
        # Should replace with underscores
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "test" in result
        assert ".json" in result

    def test_unicode_characters(self):
        """Test sanitization of unicode characters."""
        result = sanitize_filename("データ.json")
        # Non-ASCII should be replaced
        assert "json" in result

    def test_empty_filename(self):
        """Test sanitization of empty filename."""
        result = sanitize_filename("")
        assert result == "upload.json"  # Default fallback

    def test_only_special_chars(self):
        """Test sanitization when only special chars remain."""
        result = sanitize_filename("@#$%")
        assert result == "upload.json"  # Should use default

    def test_allowed_characters_preserved(self):
        """Test that allowed characters are preserved."""
        result = sanitize_filename("test-file_2024.json")
        assert result == "test-file_2024.json"
        # Dash, underscore, dot, alphanumeric should all be preserved

    def test_multiple_dots(self):
        """Test filename with multiple dots."""
        result = sanitize_filename("my.file.name.json")
        assert result == "my.file.name.json"

    def test_leading_dot(self):
        """Test filename starting with dot."""
        result = sanitize_filename(".hidden.json")
        assert result == ".hidden.json"
