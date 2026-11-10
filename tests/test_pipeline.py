"""
End-to-End Pipeline Integration Test
Verifies the complete flow as specified in the Phase I synopsis:
Text Ingestion -> Preprocessing -> Inference (<300ms) -> AES-256-GCM Vault -> Deterministic Triage / Sealed Referral.
"""
import unittest
import tempfile
import os

from core.preprocessor import TextPreprocessor
from core.classifier import StressClassifier
from triage.dispatcher import TriageDispatcher
from triage.referral import ReferralTokenGenerator
from crypto.vault import EncryptedVault


class TestAcademicStressPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "pipeline_vault.sqlite3")
        self.vault = EncryptedVault(self.db_path)
        self.vault.initialize_vault("EngineeringPassphrase2026!")

        self.preprocessor = TextPreprocessor()
        self.classifier = StressClassifier()
        self.classifier.classify("warmup")
        self.triage = TriageDispatcher()
        self.referral_gen = ReferralTokenGenerator()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_low_stress_pipeline(self):
        raw_text = "Submitted my mini project synopsis on time. Feeling confident for the viva."
        
        # 1. Ingestion & Preprocessing
        proc = self.preprocessor.process(raw_text)
        self.assertLessEqual(proc["tokens"]["token_count"], 128)
        
        # 2. Local Inference
        result = self.classifier.classify(raw_text)
        self.assertEqual(result["predicted_tier"], "Low")
        self.assertTrue(result["meets_latency_sla"])
        self.assertLess(result["latency_ms"], 300.0)

        # 3. Triage Dispatch
        triage_res = self.triage.dispatch(result["predicted_tier"], result["confidence"], result["stressors"])
        self.assertEqual(triage_res["tier"], "Low")
        self.assertFalse(triage_res["requires_clinical_escalation"])
        self.assertTrue(len(triage_res["cbt_exercises"]) > 0)

        # 4. Vault Storage
        row_id = self.vault.store_reflection(
            reflection_text=raw_text,
            predicted_tier=result["predicted_tier"],
            confidence=result["confidence"],
            stressors=result["stressors"]
        )
        self.assertIsNotNone(row_id)

    def test_complete_medium_stress_pipeline(self):
        raw_text = "I have three lab exams in two days and I haven't finished the records. Panic is setting in."
        
        # Ingestion
        proc = self.preprocessor.process(raw_text)
        self.assertTrue(any("Exam" in s for s in proc["stressors"]))

        # Inference
        result = self.classifier.classify(raw_text)
        self.assertEqual(result["predicted_tier"], "Medium")
        self.assertTrue(result["meets_latency_sla"])

        # Triage Dispatch
        triage_res = self.triage.dispatch(result["predicted_tier"], result["confidence"], result["stressors"])
        self.assertEqual(triage_res["tier"], "Medium")
        self.assertFalse(triage_res["requires_clinical_escalation"])
        
        # Interactive CBT exercises present (4-7-8 breathing, cognitive reframing, 5-4-3-2-1)
        exercise_ids = [ex["id"] for ex in triage_res["cbt_exercises"]]
        self.assertIn("breathing_478", exercise_ids)
        self.assertIn("cognitive_reframing", exercise_ids)
        self.assertIn("grounding_54321", exercise_ids)

    def test_complete_high_stress_escalation_and_referral(self):
        raw_text = "I cannot take this anymore. Everything is falling apart and I want all of this pain to just end."
        
        # Inference
        result = self.classifier.classify(raw_text)
        self.assertEqual(result["predicted_tier"], "High")
        self.assertTrue(result["meets_latency_sla"])

        # Triage Dispatch
        triage_res = self.triage.dispatch(result["predicted_tier"], result["confidence"], result["stressors"])
        self.assertEqual(triage_res["tier"], "High")
        self.assertTrue(triage_res["requires_clinical_escalation"])
        self.assertTrue(len(triage_res["crisis_helplines"]) > 0)

        # Sealed Referral Token Generation
        referral_token = self.referral_gen.generate_token(result["stressors"], result["confidence"])
        self.assertTrue(referral_token["case_id"].startswith("CUK-REF-"))
        self.assertIn("Tele-MANAS", referral_token["emergency_helplines"]["tele_manas"])
        
        # Cryptographic Verification of Referral Token
        is_authentic = self.referral_gen.verify_token(
            case_id=referral_token["case_id"],
            timestamp_str=referral_token["issued_at"],
            confidence=result["confidence"],
            signature=referral_token["verification_signature"]
        )
        self.assertTrue(is_authentic)

        # Store in Vault with sealed referral link
        row_id = self.vault.store_reflection(
            reflection_text=raw_text,
            predicted_tier=result["predicted_tier"],
            confidence=result["confidence"],
            stressors=result["stressors"],
            referral_id=referral_token["case_id"]
        )
        self.assertIsNotNone(row_id)

    def test_mild_situational_exam_anxiety_is_medium_not_high(self):
        """
        Critical calibration test:
        Common pre-exam anxiety ("tomorrow is exam i'm anxious") must be classified
        as Medium stress (CBT guidance), NOT High stress (Crisis escalation).
        """
        raw_text = "tomorrow is exam i'm anxious"
        result = self.classifier.classify(raw_text)
        
        self.assertEqual(result["predicted_tier"], "Medium", f"Expected Medium for mild exam anxiety, got {result['predicted_tier']}")
        self.assertLess(result["probabilities"]["High"], 0.20, "High stress probability should be low for normal exam jitters")
        
        triage_res = self.triage.dispatch(result["predicted_tier"], result["confidence"], result["stressors"])
        self.assertFalse(triage_res["requires_clinical_escalation"], "Mild anxiety should not trigger clinical escalation")
        
        # CBT exercises should be available
        cbt_ids = [ex["id"] for ex in triage_res["cbt_exercises"]]
        self.assertIn("breathing_478", cbt_ids)


if __name__ == "__main__":
    unittest.main()
