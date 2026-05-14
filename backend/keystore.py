"""
Encrypted API key storage using Fernet symmetric encryption.

Keys are stored in a single encrypted JSON blob at data/keys.enc.
The encryption key is derived from a machine-specific identifier + salt,
so keys.enc is not portable between machines (which is the point — if
someone copies the file, they can't read it on a different machine).

This is NOT military-grade security. It prevents plaintext API keys
sitting on disk, which is the #1 way keys get leaked (git commits,
backups, screen shares). A determined attacker with local access can
still derive the key. But it's vastly better than a .env file.
"""
import hashlib
import json
import platform
import uuid
from pathlib import Path

from backend.logging_config import get_logger

logger = get_logger("keystore")


def _get_machine_id() -> str:
    """
    Get a stable machine identifier.

    Uses platform node (MAC address) + OS name as the base.
    This is stable across reboots but changes if the network adapter changes.
    Falls back to a fixed string if MAC detection fails.
    """
    try:
        mac = uuid.getnode()
        # uuid.getnode() returns a random value if MAC can't be determined
        # (indicated by the multicast bit being set on the first octet)
        if (mac >> 40) & 1:  # Multicast bit = random MAC
            node_id = platform.node() or "velqua-default"
        else:
            node_id = str(mac)
    except Exception:
        node_id = platform.node() or "velqua-default"

    return f"{node_id}-{platform.system()}-{platform.machine()}"


def _derive_key(machine_id: str, salt: bytes) -> bytes:
    """Derive a Fernet key from machine ID + salt using PBKDF2."""
    import base64
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        machine_id.encode("utf-8"),
        salt,
        iterations=100_000,
        dklen=32,
    )
    return base64.urlsafe_b64encode(dk)


class KeyStore:
    """
    Fernet-encrypted API key storage.

    Usage:
        ks = KeyStore(data_dir)
        ks.store("openai", "sk-abc123...")
        key = ks.get("openai")  # "sk-abc123..."
        ks.delete("openai")
    """

    SALT_FILE = ".keystore.salt"
    KEYS_FILE = "keys.enc"

    def __init__(self, data_dir: Path):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._salt_path = self._data_dir / self.SALT_FILE
        self._keys_path = self._data_dir / self.KEYS_FILE
        self._fernet = self._init_fernet()

    def _init_fernet(self):
        """Initialize Fernet cipher with machine-derived key."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            logger.warning(
                "cryptography package not installed. "
                "API keys will be stored in memory only. "
                "Install with: pip install cryptography>=41.0.0"
            )
            return None

        salt = self._load_or_create_salt()
        machine_id = _get_machine_id()
        key = _derive_key(machine_id, salt)
        return Fernet(key)

    def _load_or_create_salt(self) -> bytes:
        """Load existing salt or generate a new one."""
        if self._salt_path.exists():
            return self._salt_path.read_bytes()
        import os
        salt = os.urandom(16)
        self._salt_path.write_bytes(salt)
        return salt

    def _read_encrypted(self) -> dict:
        """Read and decrypt the keys file. Returns empty dict if missing/corrupt."""
        if not self._keys_path.exists():
            return {}
        if self._fernet is None:
            return {}
        try:
            encrypted = self._keys_path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            return json.loads(decrypted.decode("utf-8"))
        except Exception as e:
            logger.warning("Failed to read keystore: %s", e)
            return {}

    def _write_encrypted(self, data: dict) -> None:
        """Encrypt and write the keys file."""
        if self._fernet is None:
            return
        plaintext = json.dumps(data).encode("utf-8")
        encrypted = self._fernet.encrypt(plaintext)
        self._keys_path.write_bytes(encrypted)

    def store(self, provider: str, key: str) -> None:
        """Store an API key for a provider."""
        data = self._read_encrypted()
        data[provider] = key
        self._write_encrypted(data)
        logger.info("Stored API key for provider: %s", provider)

    def get(self, provider: str) -> str | None:
        """Get an API key for a provider. Returns None if not found."""
        data = self._read_encrypted()
        return data.get(provider)

    def delete(self, provider: str) -> bool:
        """Delete an API key. Returns True if it existed."""
        data = self._read_encrypted()
        if provider not in data:
            return False
        del data[provider]
        self._write_encrypted(data)
        logger.info("Deleted API key for provider: %s", provider)
        return True

    def list_providers(self) -> list[str]:
        """List providers that have stored API keys."""
        data = self._read_encrypted()
        return list(data.keys())

    def has_key(self, provider: str) -> bool:
        """Check if a provider has a stored API key."""
        return provider in self._read_encrypted()
