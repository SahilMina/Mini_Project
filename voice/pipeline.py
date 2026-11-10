"""
Unified Voice Pipeline for Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka, Kadaganchi, Kalaburagi
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Compliant with Phase I Synopsis:
- AI4Bharat Indic Speech Recognition (IndicWav2Vec / IndicConformer / Dhruva API / Edge)
- Autonomic Acoustic Stress Biomarker Extraction
- Multimodal Sentiment & Prosody Fusion
- Integration with AES-256-GCM Vault & Deterministic Triage
"""
import time
from typing import Dict, Any, Optional

from voice.models import (
    SUPPORTED_LANGUAGES,
    TARGET_SAMPLE_RATE,
    AcousticBiomarkers,
    VoicePipelineResult
)
from voice.acoustic import AcousticStressExtractor
from voice.asr import AI4BharatASR, decode_base64_audio
from core.classifier import StressClassifier
from core.translator import MultilingualTranslator
from core.generator import ConversationalResponseGenerator
from triage.dispatcher import TriageDispatcher
from triage.referral import ReferralTokenGenerator
from crypto.vault import EncryptedVault


class VoicePipeline:
    """
    Multimodal voice and speech pipeline tailored for Indian engineering students.
    Seamlessly combines AI4Bharat ASR transcription, acoustic biomarker tracking,
    transformer classification, and AES-256-GCM encrypted persistence.
    """

    def __init__(
        self,
        classifier: Optional[StressClassifier] = None,
        triage: Optional[TriageDispatcher] = None,
        referral_gen: Optional[ReferralTokenGenerator] = None,
        vault: Optional[EncryptedVault] = None,
        generator: Optional[ConversationalResponseGenerator] = None
    ):
        self.asr = AI4BharatASR()
        self.acoustic_extractor = AcousticStressExtractor(TARGET_SAMPLE_RATE)
        self.classifier = classifier or StressClassifier()
        self.translator = MultilingualTranslator()
        self.triage = triage or TriageDispatcher()
        self.referral_gen = referral_gen or ReferralTokenGenerator()
        self.vault = vault
        self.generator = generator or ConversationalResponseGenerator()

    def process_voice_reflection(
        self,
        audio_bytes: bytes,
        language: str = "en-IN",
        model_choice: str = "auto",
        save_to_vault: bool = True,
        text_transcript: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end voice screening pipeline:
        1. AI4Bharat Speech Transcription
        2. Acoustic Prosodic Biomarker Extraction
        3. Text Classification via Distilled Transformer Engine
        4. Multimodal Fusion (Text + Vocal Stress Telemetry)
        5. Deterministic Triage & Clinical Escalation (if indicated)
        6. AES-256-GCM Local Vault Persistence
        """
        t_start = time.perf_counter()

        # Step 1: AI4Bharat Speech Recognition
        asr_result = self.asr.transcribe(
            audio_bytes=audio_bytes,
            language=language,
            model_choice=model_choice,
            client_transcript=text_transcript
        )
        transcription_text = asr_result["transcription"]
        audio_array = asr_result["audio_array"]
        sr = asr_result["sample_rate"]

        # Step 2: Extract Vocal Acoustic Biomarkers
        acoustic_meta = self.acoustic_extractor.extract_features(audio_array, sr)

        # Step 2.5: Multilingual Indic Translation to English (ensures zero semantic loss on transformer)
        translation_res = self.translator.translate_to_english(transcription_text, language)
        eval_text = translation_res["translated_text"]

        # Step 3: Text Classification on English semantics
        classification = self.classifier.classify(eval_text)

        # Step 4: Multimodal Decision Fusion
        fused_tier = self._fuse_modalities(
            text_tier=classification["predicted_tier"],
            text_confidence=classification["confidence"],
            vocal_stress_index=acoustic_meta["vocal_stress_index"]
        )

        active_stressors = list(classification["stressors"])
        if acoustic_meta["vocal_stress_index"] >= 75.0:
            active_stressors.append(f"Vocal Micro-Tremor ({acoustic_meta['vocal_stress_index']:.0f}/100)")
        classification["stressors"] = active_stressors

        # Step 5: Deterministic Triage Dispatch
        triage_result = self.triage.dispatch(
            tier=fused_tier,
            confidence=classification["confidence"],
            stressors=active_stressors
        )

        # Step 6: Anonymous Clinical Referral Token (if High Tier)
        referral_token = None
        if triage_result["requires_clinical_escalation"]:
            referral_token = self.referral_gen.generate_token(
                stressors=classification["stressors"] + [f"Vocal Stress Index: {acoustic_meta['vocal_stress_index']:.1f}/100"],
                confidence=classification["confidence"]
            )

        # Step 6.5: Conversational Response Generation (Strictly Language-Agnostic Model Input)
        cbt_title = triage_result.get("cbt_exercise", {}).get("title") if triage_result.get("cbt_exercise") else None
        gen_res = self.generator.generate(
            query_english=eval_text,
            stress_tier=fused_tier,
            stressors=classification["stressors"],
            recommended_cbt=cbt_title,
            clinical_escalation=triage_result["requires_clinical_escalation"]
        )
        english_reply = gen_res["english_reply"]

        # Step 6.6: Reverse Translation to Student Spoken Native Language
        reverse_trans = self.translator.translate_from_english(english_reply, target_language=language)
        native_reply = reverse_trans["translated_text"]

        conversational_reply = {
            "native_text": native_reply,
            "english_text": english_reply,
            "is_translated": reverse_trans["is_translated"],
            "target_language": language,
            "model_name": gen_res["model_name"],
            "latency_ms": gen_res["latency_ms"]
        }

        # Step 7: Encrypted Storage in AES-256-GCM Vault
        vault_entry_id = None
        vault_status_message = "Vault is locked or not provided. Audio transcription was processed in-memory."
        
        if save_to_vault and self.vault:
            if self.vault.is_user_unlocked(user_id):
                # Formulate structured voice entry payload with original query + English translation
                vault_payload = f"[Voice Reflection - {asr_result['model_used']} - {language}]\nOriginal: {transcription_text}\n"
                if translation_res["is_translated"]:
                    vault_payload += f"English Translation: {eval_text}\n"
                vault_payload += f"[Acoustic Biomarkers: F0={acoustic_meta['mean_f0_hz']}Hz, Jitter={acoustic_meta['jitter_percent']}%, WPM={acoustic_meta['speech_rate_wpm']}, VocalStress={acoustic_meta['vocal_stress_index']}/100]"
                
                vault_entry_id = self.vault.store_reflection(
                    reflection_text=vault_payload,
                    predicted_tier=fused_tier,
                    confidence=classification["confidence"],
                    stressors=classification["stressors"],
                    referral_id=referral_token["case_id"] if referral_token else None,
                    user_id=user_id,
                    primary_domain=classification.get("primary_domain", "General Academic Distress"),
                    clinical_markers=classification.get("clinical_markers", [])
                )
                vault_status_message = "Voice reflection & acoustic telemetry successfully encrypted with AES-256-GCM in local SQLite vault."
            elif not self.vault.is_initialized():
                vault_status_message = "Vault not initialized. Reflection analyzed ephemerally without persistence."
            else:
                vault_status_message = "Vault is locked. Unlock vault to persist encrypted voice reflections."

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "status": "SUCCESS",
            "transcription": transcription_text,
            "translated_text": eval_text,
            "is_translated": translation_res["is_translated"],
            "language": language,
            "model_used": asr_result["model_used"],
            "total_latency_ms": round(total_latency_ms, 2),
            "acoustic_biomarkers": acoustic_meta,
            "classification": classification,
            "fused_tier": fused_tier,
            "triage": triage_result,
            "referral_token": referral_token,
            "conversational_reply": conversational_reply,
            "vault": {
                "saved": vault_entry_id is not None,
                "entry_id": vault_entry_id,
                "message": vault_status_message
            }
        }

    def _fuse_modalities(
        self,
        text_tier: str,
        text_confidence: float,
        vocal_stress_index: float
    ) -> str:
        """
        Combines textual transformer semantics with vocal prosody:
        - If textual stress is 'High' (crisis indicators/severe hopelessness), it is strictly maintained.
        - If textual stress is 'Medium' (situational academic distress/exam anxiety), it is maintained
          as 'Medium' (guided CBT self-intervention). Autonomic acoustic arousal (vocal tremor, rapid cadence)
          reflects exam jitters, NOT a psychiatric emergency; it must NEVER trigger false clinical escalation.
        - If textual stress is 'Low' but acoustic stress is notably high (>75/100, masked distress),
          modality fusion safely elevates to 'Medium' for preventative CBT guidance.
        """
        text_tier = text_tier.strip().capitalize()

        if text_tier == "High":
            return "High"

        if text_tier == "Medium":
            return "Medium"

        # text_tier is Low: safely elevate to Medium only if acoustic markers demonstrate masked distress
        if vocal_stress_index >= 75.0:
            return "Medium"

        return "Low"

