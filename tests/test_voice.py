"""
AI4Bharat Voice Pipeline Test Suite
Tests:
- Acoustic feature extraction (F0, Jitter %, Shimmer %, Speech Rate, Pause Ratio, Stress Index)
- AI4Bharat ASR transcription across Indic languages
- Multimodal sentiment + vocal prosody fusion
- CBT Audio guidance generation and harmonic WAV synthesis
- REST API voice endpoints (/api/voice/models, /api/voice/transcribe, /api/voice/analyze, etc.)
"""
import unittest
import os
import io
import tempfile
import base64
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import backend.main as app_module
from voice import (
    VoicePipeline,
    VoiceGuidanceGenerator,
    AcousticStressExtractor,
    AI4BharatASR,
    SUPPORTED_LANGUAGES
)
from crypto.vault import EncryptedVault


class TestVoicePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_db = os.path.join(cls.temp_dir.name, "voice_test_vault.sqlite3")
        app_module.vault = EncryptedVault(cls.test_db)
        app_module.voice_pipeline.vault = app_module.vault
        app_module.classifier.classify("warmup pipeline session")
        cls.client = TestClient(app_module.app)

        # Authenticate test student
        cls.client.post("/api/auth/register", json={
            "student_id": "TEST_VOICE_STUDENT",
            "full_name": "Voice Test Student",
            "email": "voicetest@cuk.ac.in",
            "password": "Password@123",
            "department": "Computer Science",
            "role": "student"
        })
        login_res = cls.client.post("/api/auth/login", json={
            "identifier": "TEST_VOICE_STUDENT",
            "password": "Password@123"
        })
        session_token = login_res.json().get("session_token")
        if session_token:
            cls.client.cookies.set("cuk_session_token", session_token)

        # Generate a test 16kHz sine wave audio clip (3 seconds, 220Hz)
        cls.sr = 16000
        t = np.linspace(0, 3, cls.sr * 3)
        # Add micro-tremor and amplitude envelope
        tremor = 0.05 * np.sin(2 * np.pi * 15 * t)
        signal = (0.5 + tremor) * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.randn(len(t))
        bio = io.BytesIO()
        sf.write(bio, signal.astype(np.float32), cls.sr, format='WAV')
        cls.test_wav_bytes = bio.getvalue()
        cls.test_b64_audio = base64.b64encode(cls.test_wav_bytes).decode("utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_01_acoustic_extraction(self):
        extractor = AcousticStressExtractor(target_sample_rate=16000)
        data, sr = sf.read(io.BytesIO(self.test_wav_bytes), dtype='float32')
        features = extractor.extract_features(data, sr)

        self.assertIn("vocal_stress_index", features)
        self.assertIn("jitter_percent", features)
        self.assertIn("shimmer_percent", features)
        self.assertIn("speech_rate_wpm", features)
        self.assertIn("pause_ratio_percent", features)
        self.assertGreater(features["duration_sec"], 2.5)
        self.assertGreater(features["mean_f0_hz"], 100.0)
        self.assertLess(features["mean_f0_hz"], 350.0)

    def test_02_asr_transcription_languages(self):
        asr = AI4BharatASR()
        # English
        res_en = asr.transcribe(self.test_wav_bytes, language="en-IN")
        self.assertEqual(res_en["language"], "en-IN")
        self.assertIn("AI4Bharat", res_en["model_used"])
        self.assertTrue(len(res_en["transcription"]) > 10)

        # Hindi
        res_hi = asr.transcribe(self.test_wav_bytes, language="hi-IN")
        self.assertEqual(res_hi["language"], "hi-IN")
        self.assertIn("AI4Bharat", res_hi["model_used"])
        self.assertTrue(len(res_hi["transcription"]) > 5)

        # Kannada (Karnataka campus)
        res_kn = asr.transcribe(self.test_wav_bytes, language="kn-IN")
        self.assertEqual(res_kn["language"], "kn-IN")
        self.assertIn("AI4Bharat", res_kn["model_used"])
        self.assertTrue(len(res_kn["transcription"]) > 5)

    def test_03_voice_guidance_and_harmonic_audio(self):
        guidance = VoiceGuidanceGenerator()
        # Metadata check
        meta_en = guidance.get_guidance_metadata("breathing_478", "en-IN")
        self.assertEqual(meta_en["exercise_id"], "breathing_478")
        self.assertTrue(len(meta_en["steps"]) >= 4)

        meta_kn = guidance.get_guidance_metadata("breathing_478", "kn-IN")
        self.assertEqual(meta_kn["language"], "kn-IN")

        # Harmonic pacing audio generation
        wav_bytes = guidance.generate_harmonic_pacer_wav("breathing_478")
        self.assertTrue(len(wav_bytes) > 1000)
        # Verify valid WAV format
        with io.BytesIO(wav_bytes) as bio:
            data, sr = sf.read(bio)
            self.assertEqual(sr, 16000)
            self.assertGreater(len(data), 16000 * 15)  # ~19 seconds cycle

    def test_04_api_voice_models(self):
        res = self.client.get("/api/voice/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("supported_models", data)
        self.assertIn("languages", data)
        self.assertIn("en-IN", data["languages"])
        self.assertIn("hi-IN", data["languages"])
        self.assertIn("kn-IN", data["languages"])

    def test_05_api_voice_transcribe(self):
        res = self.client.post("/api/voice/transcribe", json={
            "audio_base64": self.test_b64_audio,
            "language": "en-IN",
            "model_choice": "auto",
            "save_to_vault": False
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("transcription", data)
        self.assertIn("acoustic_biomarkers", data)

    def test_06_api_voice_analyze(self):
        res = self.client.post("/api/voice/analyze", json={
            "audio_base64": self.test_b64_audio,
            "language": "en-IN",
            "model_choice": "auto",
            "save_to_vault": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("fused_tier", data)
        self.assertIn("triage", data)
        self.assertIn("acoustic_biomarkers", data)
        self.assertIn("total_latency_ms", data)
        # Verify CPU latency SLA (<300ms)
        self.assertLess(data["total_latency_ms"], 300.0)

    def test_07_api_guidance_audio_endpoint(self):
        res = self.client.get("/api/voice/guidance/breathing_478/audio")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "audio/wav")
        self.assertGreater(len(res.content), 5000)

    def test_08_api_voice_analyze_with_client_transcript(self):
        spoken_text = "tomorrow is exam i am anxious about clearing the papers"
        res = self.client.post("/api/voice/analyze", json={
            "audio_base64": self.test_b64_audio,
            "language": "en-IN",
            "model_choice": "auto",
            "save_to_vault": True,
            "text_transcript": spoken_text
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["transcription"], spoken_text)
        self.assertIn("classification", data)
        self.assertEqual(data["classification"]["predicted_tier"], "Medium")
        self.assertIn("triage", data)
        self.assertIn("acoustic_biomarkers", data)
        self.assertLess(data["total_latency_ms"], 300.0)

