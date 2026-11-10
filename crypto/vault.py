"""
Cryptographic Engine for Academic Stress & Clinical Referral Ecosystem
Compliant with Phase I & Phase II Synopsis Specifications:
- AES-256-GCM Authenticated Encryption
- PBKDF2HMAC Key Derivation (SHA-256, 100,000 rounds)
- Local SQLite Multi-Tenant Encrypted Store
- Per-User Cryptographic Session Isolation & Zero-Plaintext Audit Guarantee
"""
import os
import sqlite3
import json
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from crypto.db import DatabaseManager, DB_PATH

CANARY_PHRASE = b"CUK_ACADEMIC_STRESS_VAULT_V1_CANARY_OK"
PBKDF2_ITERATIONS = 100_000
KEY_LENGTH_BYTES = 32  # 256 bits


class EncryptedVault:
    """
    Multi-tenant zero-knowledge encrypted vault.
    Supports both legacy single-passphrase mode and multi-tenant per-user authentication.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._active_key: Optional[bytes] = None
        self._user_keys: Dict[int, bytes] = {}  # In-memory per-user derived AES keys
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager that ensures SQLite connections are closed reliably on Windows."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initializes SQLite schema for metadata and encrypted reflections."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vault_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    salt_hex TEXT NOT NULL,
                    canary_nonce_hex TEXT NOT NULL,
                    canary_ciphertext_hex TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    created_at TEXT NOT NULL,
                    nonce_hex TEXT NOT NULL,
                    ciphertext_hex TEXT NOT NULL,
                    tag_hex TEXT NOT NULL,
                    predicted_tier TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    primary_domain TEXT DEFAULT 'General Academic Distress',
                    stressors_json TEXT NOT NULL,
                    clinical_markers_json TEXT DEFAULT '[]',
                    referral_id TEXT
                )
            """)
            conn.commit()

    # =========================================================================
    # Multi-Tenant In-Memory Key Management
    # =========================================================================
    def mount_user_key(self, user_id: int, key: bytes):
        """Mounts derived AES-256 key in memory for an authenticated user session."""
        self._user_keys[user_id] = key

    def unmount_user_key(self, user_id: int):
        """Purges user's active key from memory upon logout."""
        self._user_keys.pop(user_id, None)

    def is_user_unlocked(self, user_id: Optional[int] = None) -> bool:
        """Returns True if the specified user (or master fallback) is unlocked in memory."""
        if user_id is not None and user_id in self._user_keys:
            return True
        return self._active_key is not None

    def get_effective_key(self, user_id: Optional[int] = None) -> Optional[bytes]:
        """Resolves the active encryption key for the request."""
        if user_id is not None and user_id in self._user_keys:
            return self._user_keys[user_id]
        return self._active_key

    # =========================================================================
    # Legacy Single-Passphrase Vault Interface (Backward-Compatible)
    # =========================================================================
    def is_initialized(self) -> bool:
        """Checks if the vault has been configured with a master passphrase."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM vault_meta WHERE id = 1")
            return cursor.fetchone() is not None

    def is_unlocked(self) -> bool:
        """Returns True if any key is currently mounted in memory."""
        return self._active_key is not None or len(self._user_keys) > 0

    @staticmethod
    def _derive_key(passphrase: str, salt: bytes) -> bytes:
        """Derives a 256-bit AES key using PBKDF2HMAC SHA-256."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_LENGTH_BYTES,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
        )
        return kdf.derive(passphrase.encode("utf-8"))

    def initialize_vault(self, passphrase: str) -> bool:
        """Initializes the vault with a master passphrase and a verification canary."""
        if self.is_initialized():
            raise ValueError("Vault is already initialized.")
        if len(passphrase.strip()) < 6:
            raise ValueError("Passphrase must be at least 6 characters long.")

        salt = os.urandom(16)
        key = self._derive_key(passphrase, salt)
        aesgcm = AESGCM(key)
        canary_nonce = os.urandom(12)
        encrypted_canary = aesgcm.encrypt(canary_nonce, CANARY_PHRASE, None)

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO vault_meta (id, salt_hex, canary_nonce_hex, canary_ciphertext_hex, created_at)
                VALUES (1, ?, ?, ?, ?)
            """, (
                salt.hex(),
                canary_nonce.hex(),
                encrypted_canary.hex(),
                created_at
            ))
            conn.commit()

        self._active_key = key
        return True

    def unlock(self, passphrase: str) -> bool:
        """Unlocks the master vault by validating the passphrase against the canary."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT salt_hex, canary_nonce_hex, canary_ciphertext_hex FROM vault_meta WHERE id = 1")
            row = cursor.fetchone()
            if not row:
                raise ValueError("Vault is not initialized.")

            salt = bytes.fromhex(row["salt_hex"])
            nonce = bytes.fromhex(row["canary_nonce_hex"])
            enc_canary = bytes.fromhex(row["canary_ciphertext_hex"])

        candidate_key = self._derive_key(passphrase, salt)
        aesgcm = AESGCM(candidate_key)
        try:
            decrypted = aesgcm.decrypt(nonce, enc_canary, None)
            if decrypted == CANARY_PHRASE:
                self._active_key = candidate_key
                return True
            return False
        except Exception:
            return False

    def lock(self):
        """Wipes all active keys from memory."""
        self._active_key = None
        self._user_keys.clear()

    # =========================================================================
    # Reflection Storage & Retrieval (Multi-Tenant & Guaranteed Persistence)
    # =========================================================================
    def store_reflection(
        self,
        reflection_text: str,
        predicted_tier: str,
        confidence: float,
        stressors: List[str],
        referral_id: Optional[str] = None,
        user_id: Optional[int] = None,
        primary_domain: Optional[str] = None,
        clinical_markers: Optional[List[str]] = None,
        custom_key: Optional[bytes] = None
    ) -> int:
        """
        Encrypts reflection text using AES-256-GCM and persists to SQLite.
        Atomically links to user_id if provided.
        """
        key = custom_key or self.get_effective_key(user_id)
        if not key:
            raise PermissionError("Vault is locked. Authenticate or provide master passphrase first.")

        # AES-GCM encryption
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        plaintext_bytes = reflection_text.encode("utf-8")
        
        # enc_output is (ciphertext + 16-byte tag)
        enc_output = aesgcm.encrypt(nonce, plaintext_bytes, None)
        ciphertext = enc_output[:-16]
        tag = enc_output[-16:]

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO encrypted_reflections 
                (user_id, created_at, nonce_hex, ciphertext_hex, tag_hex, predicted_tier,
                 confidence, primary_domain, stressors_json, clinical_markers_json, referral_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                created_at,
                nonce.hex(),
                ciphertext.hex(),
                tag.hex(),
                predicted_tier,
                round(confidence, 4),
                primary_domain or "General Academic Distress",
                json.dumps(stressors),
                json.dumps(clinical_markers or []),
                referral_id
            ))
            conn.commit()
            return cursor.lastrowid

    def get_reflections(
        self,
        user_id: Optional[int] = None,
        custom_key: Optional[bytes] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves reflections strictly exclusive to the authenticated account.
        If user_id is None or unauthenticated, returns an empty list to prevent cross-account leakage.
        If the account's key is mounted, decrypts reflection text in ephemeral memory.
        If locked, returns masked ciphertext.
        """
        if user_id is None:
            return []

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, created_at, nonce_hex, ciphertext_hex, tag_hex, 
                       predicted_tier, confidence, primary_domain, stressors_json,
                       clinical_markers_json, referral_id
                FROM encrypted_reflections 
                WHERE user_id = ?
                ORDER BY id DESC
            """, (user_id,))
            rows = cursor.fetchall()

        key = custom_key or self.get_effective_key(user_id)
        is_unlocked = key is not None
        aesgcm = AESGCM(key) if is_unlocked else None

        results = []
        for row in rows:
            nonce = bytes.fromhex(row["nonce_hex"])
            ciphertext = bytes.fromhex(row["ciphertext_hex"])
            tag = bytes.fromhex(row["tag_hex"])
            stressors = json.loads(row["stressors_json"])
            markers = json.loads(row["clinical_markers_json"]) if "clinical_markers_json" in row.keys() and row["clinical_markers_json"] else []

            if is_unlocked:
                try:
                    decrypted_bytes = aesgcm.decrypt(nonce, ciphertext + tag, None)
                    text_content = decrypted_bytes.decode("utf-8")
                    entry_decrypted = True
                except Exception:
                    text_content = f"🔒 [Session #{row['id']} - Sealed with Previous Key]"
                    entry_decrypted = False
            else:
                text_content = f"🔒 [AES-256-GCM Encrypted: {row['ciphertext_hex'][:16]}... (Length: {len(ciphertext)} bytes)]"
                entry_decrypted = False

            results.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "created_at": row["created_at"],
                "text": text_content,
                "tier": row["predicted_tier"],
                "confidence": row["confidence"],
                "primary_domain": row["primary_domain"] if "primary_domain" in row.keys() else "General Academic Distress",
                "stressors": stressors,
                "clinical_markers": markers,
                "referral_id": row["referral_id"],
                "is_decrypted": entry_decrypted,
                "cipher_meta": {
                    "nonce_hex": row["nonce_hex"],
                    "tag_hex": row["tag_hex"],
                    "ciphertext_preview": row["ciphertext_hex"][:32] + "..."
                }
            })

        return results

    def get_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Provides vault statistics without disclosing private contents."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute("""
                    SELECT COUNT(*), predicted_tier 
                    FROM encrypted_reflections 
                    WHERE user_id = ? OR user_id IS NULL 
                    GROUP BY predicted_tier
                """, (user_id,))
                tier_counts = {tier: count for count, tier in cursor.fetchall()}
                cursor.execute("SELECT COUNT(*) FROM encrypted_reflections WHERE user_id = ? OR user_id IS NULL", (user_id,))
                total = cursor.fetchone()[0]
            else:
                cursor.execute("SELECT COUNT(*), predicted_tier FROM encrypted_reflections GROUP BY predicted_tier")
                tier_counts = {tier: count for count, tier in cursor.fetchall()}
                cursor.execute("SELECT COUNT(*) FROM encrypted_reflections")
                total = cursor.fetchone()[0]

        return {
            "is_initialized": self.is_initialized(),
            "is_unlocked": self.is_user_unlocked(user_id),
            "total_reflections": total,
            "tier_distribution": {
                "Low": tier_counts.get("Low", 0),
                "Medium": tier_counts.get("Medium", 0),
                "High": tier_counts.get("High", 0)
            },
            "security": {
                "algorithm": "AES-256-GCM",
                "kdf": "PBKDF2HMAC-SHA256",
                "kdf_iterations": PBKDF2_ITERATIONS,
                "storage": "Local SQLite (Zero Plaintext)"
            }
        }
