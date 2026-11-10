"""
Unit Tests for Conversational Response Generator & Bidirectional Translation
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka
"""

import unittest
from core.generator import ConversationalResponseGenerator
from core.translator import MultilingualTranslator


class TestConversationalResponder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = ConversationalResponseGenerator()
        cls.translator = MultilingualTranslator()

    def test_generator_pure_english_contract(self):
        """Validates generator operates strictly with English prompt context (no language parameter)."""
        res = self.generator.generate(
            query_english="Tomorrow is my compiler design lab viva and I am panicking over backlogs.",
            stress_tier="Medium",
            stressors=["Exam & Viva Anxiety"],
            recommended_cbt="15-minute chunking"
        )
        self.assertIn("english_reply", res)
        self.assertIsInstance(res["english_reply"], str)
        self.assertTrue(len(res["english_reply"]) > 20)
        self.assertIn("Compiler", res["english_reply"])
        self.assertLess(res["latency_ms"], 100.0)

    def test_reverse_translation_hindi(self):
        """Validates English reply is translated accurately to Hindi."""
        english_reply = "Take a deep breath and start with one topic at a time. You can do this."
        trans_res = self.translator.translate_from_english(english_reply, target_language="hi-IN")
        self.assertTrue(trans_res["is_translated"])
        self.assertEqual(trans_res["target_language"], "hi-IN")
        self.assertTrue(any('\u0900' <= char <= '\u097F' for char in trans_res["translated_text"]))

    def test_reverse_translation_english_passthrough(self):
        """Validates English target bypasses translation overhead."""
        english_reply = "Your stress levels are low and well managed."
        trans_res = self.translator.translate_from_english(english_reply, target_language="en-IN")
        self.assertFalse(trans_res["is_translated"])
        self.assertEqual(trans_res["translated_text"], english_reply)


if __name__ == "__main__":
    unittest.main()

