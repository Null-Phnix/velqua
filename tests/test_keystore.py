"""
Tests for encrypted API key storage.

Tests cover: encrypt/decrypt round-trip, missing keys, overwrite,
delete, list, and corrupt file handling.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestKeyStore:
    """Tests for the Fernet-encrypted keystore."""

    @pytest.fixture
    def keystore(self, tmp_path):
        from backend.keystore import KeyStore
        return KeyStore(tmp_path)

    def test_store_and_get(self, keystore):
        keystore.store("openai", "sk-abc123")
        assert keystore.get("openai") == "sk-abc123"

    def test_get_missing_returns_none(self, keystore):
        assert keystore.get("nonexistent") is None

    def test_overwrite_key(self, keystore):
        keystore.store("openai", "sk-old")
        keystore.store("openai", "sk-new")
        assert keystore.get("openai") == "sk-new"

    def test_delete_key(self, keystore):
        keystore.store("openai", "sk-abc123")
        assert keystore.delete("openai") is True
        assert keystore.get("openai") is None

    def test_delete_nonexistent(self, keystore):
        assert keystore.delete("nonexistent") is False

    def test_list_providers(self, keystore):
        keystore.store("openai", "sk-1")
        keystore.store("anthropic", "sk-ant-2")
        providers = keystore.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_list_empty(self, keystore):
        assert keystore.list_providers() == []

    def test_has_key(self, keystore):
        keystore.store("openai", "sk-1")
        assert keystore.has_key("openai") is True
        assert keystore.has_key("groq") is False

    def test_corrupt_file_returns_empty(self, tmp_path):
        from backend.keystore import KeyStore
        ks = KeyStore(tmp_path)
        # Write garbage to the keys file
        (tmp_path / "keys.enc").write_bytes(b"not encrypted data")
        assert ks.get("openai") is None
        assert ks.list_providers() == []

    def test_multiple_keys_independent(self, keystore):
        keystore.store("openai", "sk-openai")
        keystore.store("anthropic", "sk-anthropic")
        keystore.store("groq", "gsk-groq")
        assert keystore.get("openai") == "sk-openai"
        assert keystore.get("anthropic") == "sk-anthropic"
        assert keystore.get("groq") == "gsk-groq"

    def test_salt_persists(self, tmp_path):
        from backend.keystore import KeyStore
        ks1 = KeyStore(tmp_path)
        ks1.store("openai", "sk-test")

        # New instance should read same salt and decrypt
        ks2 = KeyStore(tmp_path)
        assert ks2.get("openai") == "sk-test"

    def test_store_special_characters(self, keystore):
        """API keys may contain special chars."""
        key = "sk-ant-api03-XyZ_123+/="
        keystore.store("test", key)
        assert keystore.get("test") == key


class TestKeyStoreMachineDerived:
    """Test the machine ID derivation."""

    def test_get_machine_id_returns_string(self):
        from backend.keystore import _get_machine_id
        mid = _get_machine_id()
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_derive_key_deterministic(self):
        from backend.keystore import _derive_key
        key1 = _derive_key("test-machine", b"salt12345678")
        key2 = _derive_key("test-machine", b"salt12345678")
        assert key1 == key2

    def test_derive_key_different_salt(self):
        from backend.keystore import _derive_key
        key1 = _derive_key("test-machine", b"salt11111111")
        key2 = _derive_key("test-machine", b"salt22222222")
        assert key1 != key2

    def test_get_machine_id_multicast_mac(self):
        """When uuid.getnode() returns a multicast (random) MAC, fall back to platform.node()."""
        from backend.keystore import _get_machine_id
        import platform
        # A MAC with the multicast bit set (bit 40 of address = LSB of first octet)
        multicast_mac = 0x010000000000  # bit 40 set
        with patch("uuid.getnode", return_value=multicast_mac):
            mid = _get_machine_id()
        # Should fall back to platform.node() or "velqua-default"
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_get_machine_id_exception(self):
        """When uuid.getnode() raises, fall back to platform.node()."""
        from backend.keystore import _get_machine_id
        with patch("uuid.getnode", side_effect=RuntimeError("no network")):
            mid = _get_machine_id()
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_keystore_no_cryptography(self):
        """KeyStore without cryptography returns None fernet and no-ops on read/write."""
        import sys
        from backend.keystore import KeyStore as _KeyStore
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate cryptography not installed
            with patch.dict(sys.modules, {"cryptography": None, "cryptography.fernet": None}):
                ks = _KeyStore(Path(tmpdir))
                # _init_fernet returns None
                ks._fernet = None
                # _read_encrypted returns {} when fernet is None
                result = ks._read_encrypted()
                assert result == {}
                # _write_encrypted is a no-op when fernet is None
                ks._write_encrypted({"openai": "sk-test"})  # should not raise
                assert not (Path(tmpdir) / "keys.enc").exists()
