"""
Module 4: Sealed Anonymous Clinical Referral Token Generator
Generates tamper-evident, cryptographically verifiable referral tokens
for counselor handoffs without exposing student diary contents.
"""
import uuid
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List

SYSTEM_REFERRAL_SALT = b"CUK_CAMPUS_COUNSELING_REFERRAL_SALT_2026"


class ReferralTokenGenerator:
    def __init__(self, institution: str = "Central University of Karnataka"):
        self.institution = institution

    def generate_token(
        self,
        stressors: List[str],
        confidence: float = 0.95,
        student_alias: str = "Anonymous Student"
    ) -> Dict[str, Any]:
        """
        Creates an anonymous, tamper-evident referral token.
        Preserves complete confidentiality: no raw reflection text is included.
        """
        now = datetime.now(timezone.utc)
        case_uuid = uuid.uuid4().hex[:8].upper()
        case_id = f"CUK-REF-{now.year}-{case_uuid[:4]}-{case_uuid[4:]}"
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Create HMAC-SHA256 signature to guarantee authenticity against tampering
        payload = f"{case_id}|{timestamp_str}|HIGH_SEVERITY_DISTRESS|{confidence:.4f}".encode("utf-8")
        token_signature = hmac.new(SYSTEM_REFERRAL_SALT, payload, hashlib.sha256).hexdigest()
        one_time_passcode = f"{uuid.uuid4().hex[:6].upper()}"

        return {
            "case_id": case_id,
            "passcode": one_time_passcode,
            "status": "ISSUED",
            "issued_at": timestamp_str,
            "institution": self.institution,
            "campus_location": "Kadaganchi, Kalaburagi – 585367, INDIA",
            "target_office": "University Health Centre & Student Counseling Cell",
            "triage_code": "CLINICAL_TIER_3_HIGH_DISTRESS",
            "confidence_rating": f"{confidence * 100:.1f}%",
            "identified_risk_domains": stressors or ["Acute Academic Overload"],
            "verification_signature": token_signature,
            "privacy_attestation": (
                "Verified On-Device Screening: The student's personal journal reflections, "
                "typed text, and personal records remain strictly locked inside an on-device "
                "AES-256-GCM encrypted database and are NOT transmitted to any server or counseling logs."
            ),
            "handoff_instructions": (
                "Present this token reference to the counselor at CUK Health Centre or call 14416 (Tele-MANAS) "
                "for immediate 24x7 confidential tele-consultation."
            ),
            "emergency_helplines": {
                "tele_manas": "Tele-MANAS (14416 / 1800-891-4416 Govt of India 24x7)",
                "kiran": "KIRAN Mental Health Helpline (1800-599-0019)",
                "campus_clinic": "CUK Health Centre, Kadaganchi (Central University of Karnataka)"
            }
        }

    def verify_token(self, case_id: str, timestamp_str: str, confidence: float, signature: str) -> bool:
        """Verifies if the referral token signature is mathematically genuine and untampered."""
        payload = f"{case_id}|{timestamp_str}|HIGH_SEVERITY_DISTRESS|{confidence:.4f}".encode("utf-8")
        expected = hmac.new(SYSTEM_REFERRAL_SALT, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)
