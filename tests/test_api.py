"""
FastAPI End-to-End API Test Suite
Uses FastAPI TestClient to test all REST endpoints:
- Page serving
- Vault status, initialization, unlock, and lock
- Text ingestion and local inference (<300ms SLA)
- Deterministic triage (Low, Medium, High)
- Sealed clinical referral pass generation & cryptographic verification
- Zero plaintext leakage enforcement when locked
"""
import unittest
import os
import tempfile
from fastapi.testclient import TestClient

import backend.main as app_module
from crypto.vault import EncryptedVault


class TestAcademicStressAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_db = os.path.join(cls.temp_dir.name, "api_test_vault.sqlite3")
        
        # Inject fresh isolated vault for API tests
        app_module.vault = EncryptedVault(cls.test_db)
        cls.client = TestClient(app_module.app)

        # Register and login test user
        cls.client.post("/api/auth/register", json={
            "student_id": "TEST_API_STUDENT",
            "full_name": "API Test Student",
            "email": "apitest@cuk.ac.in",
            "password": "Password@123",
            "department": "Computer Science",
            "role": "student"
        })
        login_res = cls.client.post("/api/auth/login", json={
            "identifier": "TEST_API_STUDENT",
            "password": "Password@123"
        })
        cls.session_token = login_res.json().get("session_token")
        if cls.session_token:
            cls.client.cookies.set("cuk_session_token", cls.session_token)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_01_serve_homepage(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Academic Stress & Clinical Referral Ecosystem", res.text)
        self.assertIn("Central University of Karnataka", res.text)
        self.assertIn("Sahil Mina (24UCOMP047)", res.text)

    def test_02_vault_status_and_init(self):
        # Initial status
        res = self.client.get("/api/vault/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_initialized"])
        self.assertFalse(data["is_unlocked"])

        # Initialize vault
        init_res = self.client.post("/api/vault/init", json={"passphrase": "CentralUniv2026!"})
        self.assertEqual(init_res.status_code, 200)
        self.assertTrue(init_res.json()["unlocked"])

        # Status after init
        status_res = self.client.get("/api/vault/status")
        self.assertTrue(status_res.json()["is_initialized"])
        self.assertTrue(status_res.json()["is_unlocked"])

    def test_03_analyze_low_stress(self):
        res = self.client.post("/api/analyze", json={
            "text": "Finished my operating systems mini project synopsis early, feeling calm and confident.",
            "save_to_vault": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["classification"]["predicted_tier"], "Low")
        self.assertTrue(data["classification"]["meets_latency_sla"])
        self.assertLess(data["classification"]["latency_ms"], 300.0)
        self.assertEqual(data["triage"]["tier"], "Low")
        self.assertFalse(data["triage"]["requires_clinical_escalation"])
        self.assertTrue(data["vault"]["saved"])

    def test_04_analyze_medium_stress_cbt_dispatch(self):
        res = self.client.post("/api/analyze", json={
            "text": "I have three lab exams in two days and I haven't finished the records. Panic is setting in.",
            "save_to_vault": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["classification"]["predicted_tier"], "Medium")
        self.assertTrue(data["classification"]["meets_latency_sla"])
        self.assertEqual(data["triage"]["tier"], "Medium")
        
        # Check CBT exercises include 4-7-8 breathing pacer
        exercises = data["triage"]["cbt_exercises"]
        ex_ids = [ex["id"] for ex in exercises]
        self.assertIn("breathing_478", ex_ids)
        self.assertIn("cognitive_reframing", ex_ids)

        # Check empathetic conversational reply
        self.assertIn("conversational_reply", data)
        self.assertIn("native_text", data["conversational_reply"])
        self.assertIn("english_text", data["conversational_reply"])

    def test_05_analyze_high_stress_referral_escalation(self):
        res = self.client.post("/api/analyze", json={
            "text": "I cannot take this anymore. Everything is falling apart and I want all of this pain to just end.",
            "save_to_vault": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["classification"]["predicted_tier"], "High")
        self.assertTrue(data["triage"]["requires_clinical_escalation"])
        
        # Verify sealed referral token
        token = data["referral_token"]
        self.assertIsNotNone(token)
        self.assertTrue(token["case_id"].startswith("CUK-REF-"))
        self.assertIn("Tele-MANAS", token["emergency_helplines"]["tele_manas"])

        # Cryptographically verify token signature via API
        verify_res = self.client.post("/api/referral/verify", json={
            "case_id": token["case_id"],
            "timestamp_str": token["issued_at"],
            "confidence": data["classification"]["confidence"],
            "signature": token["verification_signature"]
        })
        self.assertEqual(verify_res.status_code, 200)
        self.assertTrue(verify_res.json()["valid"])

    def test_06_vault_decryption_and_zero_leakage_enforcement(self):
        # Unlocked: should retrieve decrypted records
        unlocked_res = self.client.get("/api/vault/entries")
        self.assertEqual(unlocked_res.status_code, 200)
        entries = unlocked_res.json()["entries"]
        self.assertEqual(len(entries), 3)
        self.assertTrue(unlocked_res.json()["is_unlocked"])
        self.assertIn("mini project", entries[2]["text"])

        # Lock vault
        lock_res = self.client.post("/api/vault/lock")
        self.assertEqual(lock_res.status_code, 200)
        self.assertFalse(lock_res.json()["unlocked"])

        # Locked: entries must be masked and strictly hide raw plaintext
        locked_res = self.client.get("/api/vault/entries")
        self.assertFalse(locked_res.json()["is_unlocked"])
        locked_entries = locked_res.json()["entries"]
        self.assertIn("AES-256-GCM Encrypted", locked_entries[0]["text"])
        self.assertNotIn("mini project", locked_entries[2]["text"])

    def test_07_metrics_endpoint(self):
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["meets_accuracy_goal"])
        self.assertTrue(data["meets_latency_sla"])

    def test_08_analyze_hindi_indic_translation_and_triage(self):
        """Verifies Hindi query translation to English ensures accurate semantic triage."""
        res = self.client.post("/api/analyze", json={
            "text": "कल मेरा कंपाइलर का एग्जाम है और मुझे बहुत टेंशन हो रही है",
            "save_to_vault": False
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(data["is_translated"])
        self.assertIn("exam", data["translated_text"].lower())
        # Calibrated accurately to Medium (instead of false Low)
        self.assertEqual(data["classification"]["predicted_tier"], "Medium")
        self.assertIn("original_text", data)


if __name__ == "__main__":
    unittest.main()
