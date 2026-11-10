"""
Unit Tests for AES-256-GCM Encrypted Vault and PBKDF2 Key Derivation
"""
import os
import unittest
import tempfile
from crypto.vault import EncryptedVault


class TestEncryptedVault(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_vault.sqlite3")
        self.vault = EncryptedVault(self.db_path)
        self.passphrase = "SecureCampusPassphrase123!"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_and_unlock(self):
        self.assertFalse(self.vault.is_initialized())
        self.assertFalse(self.vault.is_unlocked())

        # Initialize
        res = self.vault.initialize_vault(self.passphrase)
        self.assertTrue(res)
        self.assertTrue(self.vault.is_initialized())
        self.assertTrue(self.vault.is_unlocked())

        # Lock
        self.vault.lock()
        self.assertFalse(self.vault.is_unlocked())

        # Incorrect passphrase unlock
        wrong_unlock = self.vault.unlock("WrongPassword123!")
        self.assertFalse(wrong_unlock)
        self.assertFalse(self.vault.is_unlocked())

        # Correct passphrase unlock
        correct_unlock = self.vault.unlock(self.passphrase)
        self.assertTrue(correct_unlock)
        self.assertTrue(self.vault.is_unlocked())

    def test_store_and_retrieve_reflection(self):
        self.vault.initialize_vault(self.passphrase)

        secret_text = "I am deeply anxious about the dynamic programming lab exam tomorrow."
        row_id = self.vault.store_reflection(
            reflection_text=secret_text,
            predicted_tier="Medium",
            confidence=0.742,
            stressors=["Exam & Viva Anxiety"],
            referral_id=None,
            user_id=101
        )
        self.assertIsNotNone(row_id)

        # Retrieve while unlocked: should decrypt perfectly
        entries = self.vault.get_reflections(user_id=101)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], secret_text)
        self.assertEqual(entries[0]["tier"], "Medium")
        self.assertTrue(entries[0]["is_decrypted"])

        # Lock vault: should mask plaintext and expose only ciphertext to prove zero leakage
        self.vault.lock()
        locked_entries = self.vault.get_reflections(user_id=101)
        self.assertEqual(len(locked_entries), 1)
        self.assertFalse(locked_entries[0]["is_decrypted"])
        self.assertIn("AES-256-GCM Encrypted", locked_entries[0]["text"])
        self.assertNotIn(secret_text, locked_entries[0]["text"])

    def test_lock_enforcement_on_write(self):
        self.vault.initialize_vault(self.passphrase)
        self.vault.lock()

        with self.assertRaises(PermissionError):
            self.vault.store_reflection(
                reflection_text="Should fail while locked",
                predicted_tier="Low",
                confidence=0.9,
                stressors=[]
            )


if __name__ == "__main__":
    unittest.main()
