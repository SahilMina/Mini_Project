"""
AI4Bharat Voice Pipeline - Models & Language Specifications
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka, Kadaganchi, Kalaburagi
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Compliant with Phase I Synopsis (Ref [4] Javed et al. IndicWav2Vec / IndicConformer)
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Supported AI4Bharat Model Checkpoints and Architectures
MODEL_INDIC_WAV2VEC = "ai4bharat/indicwav2vec-hindi"
MODEL_INDIC_WAV2VEC_MULTI = "ai4bharat/indicwav2vec-indic-multilingual"
MODEL_INDIC_CONFORMER_EN = "ai4bharat/indicconformer_all_en"
MODEL_INDIC_CONFORMER_MULTI = "ai4bharat/indic-conformer-600m-multilingual"
MODEL_DHRUVA_API_ENDPOINT = "https://api.dhruva.ai4bharat.org"

# Target Audio Standard (AI4Bharat acoustic models mandate 16 kHz, single channel 16-bit PCM)
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION_SEC = 120
FRAME_LENGTH_MS = 25
FRAME_STEP_MS = 10

# Supported Indic Languages in the Ecosystem (reflecting Central University of Karnataka student demographics)
SUPPORTED_LANGUAGES: Dict[str, Dict[str, Any]] = {
    "en-IN": {
        "code": "en-IN",
        "name": "Indian English",
        "native_name": "Indian English (Campus Dialect)",
        "ai4bharat_service_id": "ai4bharat/indicconformer_all_en",
        "common_campus_keywords": [
            "viva", "backlogs", "cgpa", "arrears", "internal exam",
            "lab records", "placement", "aptitude", "faculty", "synopsis"
        ]
    },
    "hi-IN": {
        "code": "hi-IN",
        "name": "Hindi",
        "native_name": "हिन्दी",
        "ai4bharat_service_id": "ai4bharat/indicwav2vec-hindi",
        "common_campus_keywords": [
            "परीक्षा", "तनाव", "बैकलॉग", "प्लेसमेंट", "घबराहट",
            "फेल", "नींद नहीं आ रही", "चिंता", "प्रोजेक्ट"
        ]
    },
    "kn-IN": {
        "code": "kn-IN",
        "name": "Kannada",
        "native_name": "ಕನ್ನಡ (Karnataka Region)",
        "ai4bharat_service_id": "ai4bharat/indic-conformer-600m-multilingual",
        "common_campus_keywords": [
            "ಪರೀಕ್ಷೆ", "ಆತಂಕ", "ಫಲಿತಾಂಶ", "ಅಂಕಗಳು", "ವಿವಾ",
            "ಓದಲು ಆಗುತ್ತಿಲ್ಲ", "ಭಯ", "ಒತ್ತಡ"
        ]
    },
    "te-IN": {
        "code": "te-IN",
        "name": "Telugu",
        "native_name": "తెలుగు",
        "ai4bharat_service_id": "ai4bharat/indic-conformer-600m-multilingual",
        "common_campus_keywords": [
            "పరీక్షలు", "ఒత్తిడి", "ప్లేస్‌మెంట్", "భయం", "నిద్ర పట్టడం లేదు"
        ]
    },
    "ta-IN": {
        "code": "ta-IN",
        "name": "Tamil",
        "native_name": "தமிழ்",
        "ai4bharat_service_id": "ai4bharat/indic-conformer-600m-multilingual",
        "common_campus_keywords": [
            "தேர்வு", "பயம்", "வேலைவாய்ப்பு", "மன அழுத்தம்"
        ]
    }
}


class VoiceAnalysisRequest(BaseModel):
    """Voice reflection payload submitted as Base64 encoded audio or preset."""
    audio_base64: str = Field(..., description="Base64-encoded WAV, WebM, or PCM audio")
    language: str = Field("en-IN", description="Indic language code (e.g. en-IN, hi-IN, kn-IN)")
    model_choice: str = Field("auto", description="Model backend: 'indicwav2vec', 'indicconformer', 'dhruva', or 'auto'")
    save_to_vault: bool = Field(True, description="Whether to encrypt and store in AES-256-GCM vault")
    text_transcript: Optional[str] = Field(None, description="Optional live recognized speech transcript from client")


class AcousticBiomarkers(BaseModel):
    """Clinical & prosodic acoustic stress indicators computed from the voice stream."""
    duration_sec: float
    sample_rate: int
    mean_f0_hz: float
    jitter_percent: float
    shimmer_percent: float
    speech_rate_wpm: float
    pause_ratio_percent: float
    rms_energy_db: float
    vocal_stress_index: float
    stress_severity: str
    indicators: List[str]


class VoicePipelineResult(BaseModel):
    """Structured response from the multimodal voice + NLP triage pipeline."""
    transcription: str
    language: str
    model_used: str
    acoustic_biomarkers: AcousticBiomarkers
    classification: Dict[str, Any]
    fused_tier: str
    triage: Dict[str, Any]
    referral_token: Any
    vault_status: Dict[str, Any]

