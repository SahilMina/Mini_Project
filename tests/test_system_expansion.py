"""
Comprehensive End-to-End System Expansion Verification Test
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))

from crypto.db import db_manager
from crypto.vault import EncryptedVault
from backend.auth import AuthManager, RegisterRequest, LoginRequest
from core.classifier import StressClassifier
from core.generator import ConversationalResponseGenerator
from core.memory import memory_manager
from triage.dispatcher import TriageDispatcher
from triage.referral import ReferralTokenGenerator
from voice.pipeline import VoicePipeline

def test_full_pipeline():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE SYSTEM EXPANSION VERIFICATION SUITE")
    print("=" * 70)

    vault = EncryptedVault()
    auth = AuthManager(vault)
    classifier = StressClassifier()
    triage = TriageDispatcher()
    referral_gen = ReferralTokenGenerator()
    generator = ConversationalResponseGenerator()

    # 1. Register a student
    print("\n[1] Testing Student Registration...")
    reg_id = "24UCOMP047_TEST"
    reg_pass = "SecurePass123"
    try:
        user_info = db_manager.create_user(
            student_id=reg_id,
            full_name="Sahil Mina (Test Student)",
            email="sahil.mina.test@cuk.ac.in",
            password=reg_pass,
            department="Computer Science",
            role="student"
        )
        print(f" -> User registered: ID={user_info['id']}, StudentID={user_info['student_id']}")
    except ValueError as e:
        print(f" -> User already exists ({e}), retrieving credentials...")
        with db_manager.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, student_id FROM users WHERE student_id = ?", (reg_id,))
            row = cursor.fetchone()
            user_info = {"id": row["id"], "student_id": row["student_id"]}

    user_id = user_info["id"]

    # 2. Authenticate & Mount Vault Key
    print("\n[2] Testing Authentication & Session Vault Mounting...")
    auth_result = db_manager.verify_credentials(reg_id, reg_pass)
    assert auth_result is not None, "Authentication failed!"
    user_dict, vault_key = auth_result
    vault.mount_user_key(user_id, vault_key)
    assert vault.is_user_unlocked(user_id), "User vault key failed to mount!"
    print(f" -> Vault Key Mounted in RAM for user_id={user_id}. Unlocked={vault.is_user_unlocked(user_id)}")

    # 3. Test High-Severity Escalation Analysis
    print("\n[3] Testing High-Tier Crisis Utterance Analysis & Persistence...")
    test_query = "I didn't got any placement it would be better to die than get humiliated like this"
    
    # Classification
    res = classifier.classify(test_query)
    print(f" -> Predicted Tier: {res['predicted_tier']} (Conf: {res['confidence']})")
    print(f" -> Clinical Markers: {res['clinical_markers']}")
    print(f" -> Primary Domain: {res['primary_domain']}")
    assert res['predicted_tier'] == "High", f"Expected High, got {res['predicted_tier']}"

    # Triage
    triage_res = triage.dispatch(res['predicted_tier'], res['confidence'], res['stressors'])
    assert triage_res['requires_clinical_escalation'], "Expected clinical escalation!"

    # Referral Generation
    referral = referral_gen.generate_token(res['stressors'], res['confidence'])
    print(f" -> Referral Token Generated: Case ID={referral['case_id']}, Passcode={referral['passcode']}")

    # Encrypted Reflection Storage
    entry_id = vault.store_reflection(
        reflection_text=test_query,
        predicted_tier=res['predicted_tier'],
        confidence=res['confidence'],
        stressors=res['stressors'],
        referral_id=referral['case_id'],
        user_id=user_id,
        primary_domain=res['primary_domain'],
        clinical_markers=res['clinical_markers']
    )
    assert entry_id > 0, "Reflection failed to save to vault!"
    print(f" -> Reflection encrypted and persisted into SQLite! Entry ID={entry_id}")

    # Record in Clinical Referrals Table
    db_manager.record_referral(
        case_id=referral['case_id'],
        user_id=user_id,
        tier=res['predicted_tier'],
        confidence=res['confidence'],
        primary_domain=res['primary_domain'],
        stressors=res['stressors'],
        clinical_markers=res['clinical_markers'],
        vocal_stress_index=0.0,
        hmac_signature=referral['verification_signature'],
        student_passcode=referral['passcode']
    )
    print(f" -> Referral case registered in clinical queue!")

    # Record Memory Turn
    memory_manager.record_turn(
        user_id=user_id,
        user_text=test_query,
        assistant_reply="I hear how painful this placement outcome feels...",
        primary_domain=res['primary_domain']
    )
    print(" -> Conversation turn recorded in context memory.")

    # 4. Test Counselor Triage Queue & Dual-Key Decryption
    print("\n[4] Testing Counselor Triage Desk & Dual-Key Zero-Knowledge Decryption...")
    queue = db_manager.get_triage_queue()
    print(f" -> Total Cases in Triage Queue: {len(queue['cases'])}")
    target_case = next((c for c in queue['cases'] if c['case_id'] == referral['case_id']), None)
    assert target_case is not None, "Target referral not found in triage queue!"
    print(f" -> Found Case: {target_case['case_id']} from {target_case['student_name']} ({target_case['student_id']})")

    # Dual-Key Decryption with Student's Passphrase
    decrypted = db_manager.decrypt_referral_reflection(referral['case_id'], reg_pass)
    print(f" -> Dual-Key Decryption Successful! Decrypted Text:\n    \"{decrypted['decrypted_text']}\"")
    assert decrypted['decrypted_text'] == test_query, "Decrypted text mismatch!"

    # 5. Test Memory Context Retention in Subsequent Turn
    print("\n[5] Testing Context Memory Retention in Generator...")
    context = memory_manager.format_context_for_prompt(user_id)
    print(f" -> Formatted Context for Prompt:\n{context}")
    assert "placement" in context.lower(), "Context missing placement reference!"

    gen_reply = generator.generate(
        query_english="I am trying to calm down now, what should I do next?",
        stress_tier="Medium",
        stressors=["Placement & Career Uncertainty"],
        clinical_escalation=False,
        conversation_context=context
    )
    print(f" -> Contextual Empathetic Reply:\n    \"{gen_reply['english_reply']}\"")
    assert "earlier" in gen_reply['english_reply'].lower() or "mind" in gen_reply['english_reply'].lower() or "counselor" in gen_reply['english_reply'].lower(), "Contextual cue missing!"

    # 6. Test Unmount on Logout
    print("\n[6] Testing Vault Unmount on Session Logout...")
    vault.unmount_user_key(user_id)
    assert not vault.is_user_unlocked(user_id), "Key should be purged from memory!"
    print(" -> Vault key successfully purged from memory. Status is now LOCKED.")

    print("\n" + "=" * 70)
    print("ALL 6 EXPANSION VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_full_pipeline()
