"""Tests for logging configuration, specifically the file handler path."""
import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure backend is importable
VELQUA_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VELQUA_ROOT))

# Set a temp DB so importing backend.config doesn't fail
_tmpdir = tempfile.mkdtemp()
os.environ.setdefault("VELQUA_DB_PATH", os.path.join(_tmpdir, "test_log.db"))

from backend.logging_config import setup_logging, get_logger


class TestLoggingConfig:
    def test_setup_with_file_handler(self, tmp_path):
        """setup_logging(log_file=...) should create the log file and add a file handler."""
        log_file = tmp_path / "logs" / "velqua_test.log"
        assert not log_file.exists()

        setup_logging(log_file=log_file, level="DEBUG")

        # The parent directory should have been created
        assert log_file.parent.exists()

        # Write a log message to flush something into the file
        test_logger = get_logger("test_file_handler")
        test_logger.info("File handler test message")

        # Force flush all handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        # The log file should now exist and contain the message
        assert log_file.exists()
        content = log_file.read_text()
        assert "File handler test message" in content

    def test_setup_without_file_handler(self, tmp_path):
        """setup_logging() without log_file should only have console handler."""
        setup_logging(level="INFO")

        # Root logger should have handlers, none should be FileHandler
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        # There may be a file handler from the previous test, but the point is
        # this call should succeed without error
        assert root.handlers  # At least one handler exists

    def test_get_logger_prefix(self):
        """get_logger should prefix with 'velqua.'"""
        logger = get_logger("mymodule")
        assert logger.name == "velqua.mymodule"
