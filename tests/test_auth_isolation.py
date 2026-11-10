"""
Test Suite: Strict Authentication Enforcement & Multi-Tenant Vault Isolation
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from backend.main import app
from crypto.db import db_manager

def test_auth_enforcement_and_isolation():
    client = TestClient(app, follow_redirects=False)

    print("=" * 70)
    print("VERIFYING STRICT AUTHENTICATION & MULTI-TENANT ISOLATION")
    print("=" * 70)

    # 1. Verify Unauthenticated Home Page Access is Blocked
    print("\n[1] Testing Unauthenticated Access to '/' ...")
    r_home = client.get("/")
    assert r_home.status_code == 303, f"Expected 303 redirect, got {r_home.status_code}"
    assert r_home.headers["location"] == "/login", f"Expected redirect to /login, got {r_home.headers['location']}"
    print(" -> PASSED: Unauthenticated '/' redirects to '/login' (HTTP 303)")

    # 2. Verify Unauthenticated Vault Entries Access is Blocked
    print("\n[2] Testing Unauthenticated Access to '/api/vault/entries' ...")
    r_entries = client.get("/api/vault/entries")
    assert r_entries.status_code == 401, f"Expected 401, got {r_entries.status_code}"
    print(" -> PASSED: Unauthenticated access to '/api/vault/entries' returned 401 Unauthorized")

    # 3. Create User A and User B
    print("\n[3] Creating Two Distinct Accounts (User A and User B) ...")
    uid_a = "TEST_STUDENT_ALPHA"
    pwd_a = "PassAlpha123"
    try:
        user_a = db_manager.create_user(uid_a, "Alpha Student", "alpha@cuk.ac.in", pwd_a, "CSE", "student")
    except ValueError:
        pass

    uid_b = "TEST_STUDENT_BETA"
    pwd_b = "PassBeta123"
    try:
        user_b = db_manager.create_user(uid_b, "Beta Student", "beta@cuk.ac.in", pwd_b, "ECE", "student")
    except ValueError:
        pass

    # 4. Log in as User A
    print("\n[4] Logging in as User A and Storing Reflections ...")
    client_a = TestClient(app, follow_redirects=False)
    login_a = client_a.post("/api/auth/login", json={"identifier": uid_a, "password": pwd_a})
    assert login_a.status_code == 200, f"Login A failed: {login_a.text}"

    # Verify User A can now access '/'
    r_home_a = client_a.get("/")
    assert r_home_a.status_code == 200, "User A failed to access '/'"
    print(" -> PASSED: Authenticated User A accesses '/' (HTTP 200)")

    # User A submits reflection
    analyze_a = client_a.post("/api/analyze", json={
        "text": "Alpha Reflection: I am anxious about tomorrow's machine learning lab exam",
        "save_to_vault": True
    })
    assert analyze_a.status_code == 200, f"Analyze A failed: {analyze_a.text}"
    entries_a = client_a.get("/api/vault/entries").json()
    assert len(entries_a["entries"]) >= 1, "User A should have entries!"
    print(f" -> User A has {len(entries_a['entries'])} reflections in personal vault.")

    # 5. Log in as User B
    print("\n[5] Logging in as User B and Verifying Strict Session Isolation ...")
    client_b = TestClient(app, follow_redirects=False)
    login_b = client_b.post("/api/auth/login", json={"identifier": uid_b, "password": pwd_b})
    assert login_b.status_code == 200, f"Login B failed: {login_b.text}"

    # User B checks vault entries BEFORE saving anything
    entries_b_before = client_b.get("/api/vault/entries").json()
    # User B must see ZERO of User A's entries!
    assert all(e["user_id"] != entries_a["entries"][0]["user_id"] for e in entries_b_before["entries"]), "LEAK DETECTED: User B sees User A reflections!"
    print(" -> PASSED: User B cannot see any of User A's encrypted vault reflections.")

    # User B submits own reflection
    analyze_b = client_b.post("/api/analyze", json={
        "text": "Beta Reflection: Preparing for campus placement technical interview round",
        "save_to_vault": True
    })
    assert analyze_b.status_code == 200
    entries_b_after = client_b.get("/api/vault/entries").json()
    for e in entries_b_after["entries"]:
        assert "Alpha" not in e.get("text", ""), "LEAK DETECTED: User A content found in User B vault!"
    print(f" -> PASSED: User B vault strictly contains only User B entries ({len(entries_b_after['entries'])} entries).")

    # 6. Verify Admin Portal Link Button in Login Page
    print("\n[6] Verifying Admin Direct Button in '/login' ...")
    r_login_page = client.get("/login")
    assert r_login_page.status_code == 200
    assert 'href="/admin"' in r_login_page.text, "Admin link missing from /login!"
    assert 'btn-admin-portal' in r_login_page.text, "Admin button class missing!"
    print(" -> PASSED: '/login' includes prominent direct button to '/admin' portal.")

    print("\n" + "=" * 70)
    print("ALL ISOLATION & AUTH ENFORCEMENT TESTS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_auth_enforcement_and_isolation()

