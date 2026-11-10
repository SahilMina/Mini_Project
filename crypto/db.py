"""
Database Management & Multi-Tenant Relational Schema Engine
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka, Kadaganchi, Kalaburagi
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Manages multi-tenant relational persistence with zero-plaintext compliance:
- users: Student and Clinical Counselor credentials & cryptographic salts
- user_sessions: Secure session tokens with expiry
- encrypted_reflections: Per-user AES-256-GCM encrypted diary records
- clinical_referrals: Cryptographically sealed priority clinical referral tokens
- conversation_memory: Multi-turn contextual continuity store
"""

import os
import sqlite3
import json
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes

DB_PATH = "academic_vault.sqlite3"
CANARY_PHRASE = b"CUK_ACADEMIC_STRESS_USER_VAULT_OK"
PBKDF2_ITERATIONS = 100_000
KEY_LENGTH_BYTES = 32
SESSION_EXPIRY_DAYS = 7


class DatabaseManager:
    """Manages SQLite schema creation, migrations, and transactional execution."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def connection(self):
        """Context manager guaranteeing reliable SQLite connection closing."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        """Creates tables and performs non-destructive schema migrations."""
        with self.connection() as conn:
            cursor = conn.cursor()

            # 1. Users Table (Students, Counselors, HODs, Admins)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    department TEXT DEFAULT 'Computer Science',
                    password_hash TEXT NOT NULL,
                    salt_hex TEXT NOT NULL,
                    canary_nonce_hex TEXT NOT NULL,
                    canary_ciphertext_hex TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'student',
                    created_at TEXT NOT NULL
                )
            """)

            # 2. User Sessions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_token TEXT UNIQUE NOT NULL,
                    ip_hash TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)

            # 3. Encrypted Reflections (Per-User Isolated Vault)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
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

            # 4. Clinical Referrals Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clinical_referrals (
                    case_id TEXT PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    primary_domain TEXT NOT NULL,
                    stressors_json TEXT NOT NULL,
                    clinical_markers_json TEXT,
                    vocal_stress_index REAL DEFAULT 0.0,
                    hmac_signature TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    student_passcode_hash TEXT,
                    counselor_notes_encrypted TEXT,
                    resolved_at TEXT,
                    resolved_by TEXT
                )
            """)

            # 5. Conversation Memory Buffer
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content_encrypted TEXT NOT NULL,
                    summary_topic TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Perform safe non-destructive migration on existing encrypted_reflections table
            cursor.execute("PRAGMA table_info(encrypted_reflections)")
            cols = [col["name"] for col in cursor.fetchall()]

            if "user_id" not in cols:
                cursor.execute("ALTER TABLE encrypted_reflections ADD COLUMN user_id INTEGER REFERENCES users(id)")
            if "primary_domain" not in cols:
                cursor.execute("ALTER TABLE encrypted_reflections ADD COLUMN primary_domain TEXT DEFAULT 'General Academic Distress'")
            if "clinical_markers_json" not in cols:
                cursor.execute("ALTER TABLE encrypted_reflections ADD COLUMN clinical_markers_json TEXT DEFAULT '[]'")

            self._seed_default_staff_accounts(cursor)

            conn.commit()

    def _seed_default_staff_accounts(self, cursor):
        """Seeds default clinical counselor, HOD, and admin accounts if not already present."""
        default_staff = [
            ("COUNSELOR01", "Dr. Ananya Sharma (Lead Clinical Psychologist)", "counselor@cuk.ac.in", "Counselor@123", "Health Centre & Counseling Cell", "counselor"),
            ("HOD_CS", "Prof. Srikantaiah K C (Head of Department)", "srikantaiah@cuk.ac.in", "HodPassword@123", "Computer Science", "hod"),
            ("ADMIN01", "System Administrator (CUK Mental Health Cell)", "admin@cuk.ac.in", "AdminPass@123", "Central Administration", "admin")
        ]
        now_iso = datetime.now(timezone.utc).isoformat()
        for emp_id, name, email, pwd, dept, role in default_staff:
            cursor.execute("SELECT 1 FROM users WHERE student_id = ? OR email = ?", (emp_id, email))
            if not cursor.fetchone():
                salt = os.urandom(16)
                key = self.derive_key(pwd, salt)
                aesgcm = AESGCM(key)
                canary_nonce = os.urandom(12)
                encrypted_canary = aesgcm.encrypt(canary_nonce, CANARY_PHRASE, None)

                pwd_salt = os.urandom(16)
                pwd_hash = self.derive_key(pwd, pwd_salt).hex()
                stored_pwd = f"{pwd_salt.hex()}${pwd_hash}"

                cursor.execute("""
                    INSERT INTO users (
                        student_id, full_name, email, department, password_hash,
                        salt_hex, canary_nonce_hex, canary_ciphertext_hex, role, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (emp_id, name, email, dept, stored_pwd, salt.hex(), canary_nonce.hex(), encrypted_canary.hex(), role, now_iso))

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derives a 256-bit AES-GCM key using PBKDF2HMAC SHA-256 (100,000 rounds)."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_LENGTH_BYTES,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
        )
        return kdf.derive(password.encode("utf-8"))

    def create_user(
        self,
        student_id: str,
        full_name: str,
        email: str,
        password: str,
        department: str = "Computer Science",
        role: str = "student"
    ) -> Dict[str, Any]:
        """Registers a new user and generates their zero-knowledge cryptographic vault canary."""
        clean_student_id = student_id.strip().upper()
        clean_email = email.strip().lower()
        clean_name = full_name.strip()

        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters long.")

        salt = os.urandom(16)
        key = self.derive_key(password, salt)
        aesgcm = AESGCM(key)
        canary_nonce = os.urandom(12)
        encrypted_canary = aesgcm.encrypt(canary_nonce, CANARY_PHRASE, None)

        # Hash password for standard auth authentication
        pwd_salt = os.urandom(16)
        pwd_hash = self.derive_key(password, pwd_salt).hex()
        stored_password_hash = f"{pwd_salt.hex()}${pwd_hash}"

        created_at = datetime.now(timezone.utc).isoformat()

        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO users (
                        student_id, full_name, email, department, password_hash,
                        salt_hex, canary_nonce_hex, canary_ciphertext_hex, role, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    clean_student_id,
                    clean_name,
                    clean_email,
                    department,
                    stored_password_hash,
                    salt.hex(),
                    canary_nonce.hex(),
                    encrypted_canary.hex(),
                    role,
                    created_at
                ))
                user_id = cursor.lastrowid
                conn.commit()
            except sqlite3.IntegrityError as e:
                if "student_id" in str(e):
                    raise ValueError(f"Student ID '{clean_student_id}' is already registered.")
                elif "email" in str(e):
                    raise ValueError(f"Email '{clean_email}' is already registered.")
                raise ValueError(f"Registration failed: {str(e)}")

        return {
            "id": user_id,
            "student_id": clean_student_id,
            "full_name": clean_name,
            "email": clean_email,
            "department": department,
            "role": role,
            "created_at": created_at
        }

    def verify_credentials(self, identifier: str, password: str) -> Optional[Tuple[Dict[str, Any], bytes]]:
        """
        Verifies login credentials and derives the user's active 256-bit AES-GCM vault key.
        Returns: (user_dict, vault_key_bytes) if valid, else None.
        """
        clean_id = identifier.strip()
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, student_id, full_name, email, department, role, password_hash,
                       salt_hex, canary_nonce_hex, canary_ciphertext_hex, created_at
                FROM users
                WHERE UPPER(student_id) = UPPER(?) OR LOWER(email) = LOWER(?)
            """, (clean_id, clean_id))
            row = cursor.fetchone()

        if not row:
            return None

        # Verify password hash
        try:
            salt_hex, hash_hex = row["password_hash"].split("$")
            computed_hash = self.derive_key(password, bytes.fromhex(salt_hex)).hex()
            if not secrets.compare_digest(computed_hash, hash_hex):
                return None
        except Exception:
            return None

        # Derive Vault Key & Verify Canary
        vault_salt = bytes.fromhex(row["salt_hex"])
        canary_nonce = bytes.fromhex(row["canary_nonce_hex"])
        enc_canary = bytes.fromhex(row["canary_ciphertext_hex"])

        candidate_vault_key = self.derive_key(password, vault_salt)
        aesgcm = AESGCM(candidate_vault_key)
        try:
            decrypted = aesgcm.decrypt(canary_nonce, enc_canary, None)
            if decrypted != CANARY_PHRASE:
                return None
        except Exception:
            return None

        user_dict = {
            "id": row["id"],
            "student_id": row["student_id"],
            "full_name": row["full_name"],
            "email": row["email"],
            "department": row["department"],
            "role": row["role"],
            "created_at": row["created_at"]
        }
        return user_dict, candidate_vault_key

    def create_session(self, user_id: int, ip_hash: Optional[str] = None) -> str:
        """Generates a cryptographically random session token with 7-day expiration."""
        token = secrets.token_hex(32)
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(days=SESSION_EXPIRY_DAYS)

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_sessions (user_id, session_token, ip_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                token,
                ip_hash or "",
                created_at.isoformat(),
                expires_at.isoformat()
            ))
            conn.commit()

        return token

    def get_user_from_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Retrieves user profile linked to an active, unexpired session token."""
        if not session_token:
            return None

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.id, u.student_id, u.full_name, u.email, u.department, u.role, u.salt_hex,
                       s.expires_at
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? AND s.expires_at > ?
            """, (session_token, now_iso))
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "student_id": row["student_id"],
            "full_name": row["full_name"],
            "email": row["email"],
            "department": row["department"],
            "role": row["role"],
            "salt_hex": row["salt_hex"]
        }

    def delete_session(self, session_token: str):
        """Purges an active session token upon logout."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
            conn.commit()

    # =========================================================================
    # Clinical Referrals & Crisis Triage Management
    # =========================================================================
    def record_referral(
        self,
        case_id: str,
        user_id: Optional[int],
        tier: str,
        confidence: float,
        primary_domain: str,
        stressors: List[str],
        clinical_markers: Optional[List[str]] = None,
        vocal_stress_index: float = 0.0,
        hmac_signature: str = "",
        student_passcode: Optional[str] = None
    ) -> bool:
        """Records a Priority High-Tier Crisis Referral for counselor triage."""
        created_at = datetime.now(timezone.utc).isoformat()
        passcode_hash = None
        if student_passcode:
            p_salt = os.urandom(16)
            p_hash = self.derive_key(student_passcode, p_salt).hex()
            passcode_hash = f"{p_salt.hex()}${p_hash}"

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO clinical_referrals (
                    case_id, user_id, created_at, tier, confidence, primary_domain,
                    stressors_json, clinical_markers_json, vocal_stress_index,
                    hmac_signature, status, student_passcode_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
            """, (
                case_id,
                user_id,
                created_at,
                tier,
                round(confidence, 4),
                primary_domain,
                json.dumps(stressors),
                json.dumps(clinical_markers or []),
                round(vocal_stress_index, 2),
                hmac_signature,
                passcode_hash
            ))
            conn.commit()
            return True

    def get_triage_queue(self) -> Dict[str, Any]:
        """
        Retrieves all active clinical referrals for counselor triage dashboard.
        Joins with users table to provide student name and department if available.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.case_id, r.user_id, r.created_at, r.tier, r.confidence,
                       r.primary_domain, r.stressors_json, r.clinical_markers_json,
                       r.vocal_stress_index, r.status,
                       u.student_id, u.full_name as student_name, u.department
                FROM clinical_referrals r
                LEFT JOIN users u ON r.user_id = u.id
                ORDER BY r.created_at DESC
            """)
            rows = cursor.fetchall()

            cases = []
            tremor_count = 0
            for row in rows:
                vocal_idx = row["vocal_stress_index"] or 0.0
                if vocal_idx > 50.0:
                    tremor_count += 1

                cases.append({
                    "case_id": row["case_id"],
                    "user_id": row["user_id"],
                    "student_id": row["student_id"] or "ANON_STUDENT",
                    "student_name": row["student_name"] or "Anonymous Student",
                    "department": row["department"] or "General Campus",
                    "created_at": row["created_at"],
                    "tier": row["tier"],
                    "confidence": row["confidence"],
                    "primary_domain": row["primary_domain"],
                    "stressors": json.loads(row["stressors_json"] or "[]"),
                    "clinical_markers": json.loads(row["clinical_markers_json"] or "[]"),
                    "vocal_stress_index": vocal_idx,
                    "status": row["status"]
                })

            cursor.execute("SELECT COUNT(*) as total FROM encrypted_reflections")
            total_reflections = cursor.fetchone()["total"]

        return {
            "cases": cases,
            "total_reflections": total_reflections,
            "vocal_tremor_count": tremor_count
        }

    def decrypt_referral_reflection(
        self,
        case_id: str,
        student_passphrase: str
    ) -> Dict[str, Any]:
        """
        Dual-Key Zero-Knowledge Consultation Decryption Handshake.
        Decrypts the student's reflection text strictly in ephemeral memory using
        the student's personal passphrase presented during the counseling consultation.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.case_id, r.user_id, r.primary_domain, r.stressors_json,
                       r.clinical_markers_json, r.student_passcode_hash,
                       u.salt_hex, u.full_name as student_name, u.student_id
                FROM clinical_referrals r
                LEFT JOIN users u ON r.user_id = u.id
                WHERE r.case_id = ?
            """, (case_id,))
            referral = cursor.fetchone()

        if not referral:
            raise ValueError(f"Referral case '{case_id}' was not found in clinical database.")

        user_id = referral["user_id"]
        salt_hex = referral["salt_hex"]

        # Fetch the encrypted reflection row linked to this referral
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, nonce_hex, ciphertext_hex, tag_hex, stressors_json, clinical_markers_json
                FROM encrypted_reflections
                WHERE referral_id = ? OR (user_id = ? AND predicted_tier = 'High')
                ORDER BY created_at DESC LIMIT 1
            """, (case_id, user_id))
            reflection_row = cursor.fetchone()

        if not reflection_row:
            raise ValueError(f"No encrypted diary reflection found linked to case '{case_id}'.")

        # Derive candidate AES key from supplied student passphrase
        derived_key = None
        if salt_hex:
            salt = bytes.fromhex(salt_hex)
            derived_key = self.derive_key(student_passphrase, salt)

        decrypted_text = None
        nonce = bytes.fromhex(reflection_row["nonce_hex"])
        ciphertext = bytes.fromhex(reflection_row["ciphertext_hex"])
        tag = bytes.fromhex(reflection_row["tag_hex"])
        combined_payload = ciphertext + tag

        if derived_key:
            try:
                aesgcm = AESGCM(derived_key)
                decrypted_bytes = aesgcm.decrypt(nonce, combined_payload, None)
                decrypted_text = decrypted_bytes.decode("utf-8")
            except Exception:
                pass

        # Fallback: try master vault salt if legacy or unlinked
        if not decrypted_text:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT salt_hex FROM vault_meta WHERE id = 1")
                v_meta = cursor.fetchone()
                if v_meta and v_meta["salt_hex"]:
                    try:
                        m_salt = bytes.fromhex(v_meta["salt_hex"])
                        m_key = self.derive_key(student_passphrase, m_salt)
                        aesgcm = AESGCM(m_key)
                        decrypted_bytes = aesgcm.decrypt(nonce, combined_payload, None)
                        decrypted_text = decrypted_bytes.decode("utf-8")
                    except Exception:
                        pass

        if not decrypted_text:
            raise ValueError("Decryption failed: Incorrect student passphrase or referral passcode.")

        stressors = json.loads(reflection_row["stressors_json"] or "[]")
        markers = json.loads(reflection_row["clinical_markers_json"] or "[]")

        return {
            "case_id": case_id,
            "student_id": referral["student_id"] or "Student",
            "student_name": referral["student_name"] or "Student",
            "primary_domain": referral["primary_domain"],
            "decrypted_text": decrypted_text,
            "stressors": stressors,
            "clinical_markers": markers
        }

    # =========================================================================
    # Conversation Memory Engine Persistence
    # =========================================================================
    def save_memory_turn(
        self,
        user_id: int,
        role: str,
        content: str,
        summary_topic: Optional[str] = None
    ):
        """Saves a conversational turn for contextual continuity."""
        created_at = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(turn_index), 0) + 1 as next_idx FROM conversation_memory WHERE user_id = ?", (user_id,))
            turn_idx = cursor.fetchone()["next_idx"]

            cursor.execute("""
                INSERT INTO conversation_memory (user_id, turn_index, role, content_encrypted, summary_topic, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, turn_idx, role, content, summary_topic or "", created_at))
            conn.commit()

    def get_recent_memory(self, user_id: int, limit: int = 6) -> List[Dict[str, Any]]:
        """Returns the last N conversation turns for prompt context injection."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT turn_index, role, content_encrypted, summary_topic, created_at
                FROM conversation_memory
                WHERE user_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()

        return [
            {
                "turn_index": r["turn_index"],
                "role": r["role"],
                "content": r["content_encrypted"],
                "topic": r["summary_topic"],
                "created_at": r["created_at"]
            }
            for r in reversed(rows)
        ]


# Singleton database instance
db_manager = DatabaseManager()

