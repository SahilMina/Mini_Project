"""
Application Server Launcher
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka - Department of Computer Science
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C
"""
import sys
import os
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  ACADEMIC STRESS & CLINICAL REFERRAL ECOSYSTEM - (Phase II)")
    print("  Central University of Karnataka, Kadaganchi, Kalaburagi")
    print("  Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C")
    print("=" * 70)
    print("[*] Privacy Boundary : 100% Client-Side On-Device Processing")
    print("[*] Student Portal   : http://127.0.0.1:8000/login")
    print("[*] Counselor Portal : http://127.0.0.1:8000/admin")
    print("[*] Main Interface   : http://127.0.0.1:8000")
    print("[*] Encryption       : Multi-Tenant AES-256-GCM + PBKDF2HMAC (100k rounds)")
    print("[*] Inference SLA    : Quantized ONNX MiniLM (<25ms on CPU)")
    print("[*] Context Memory   : SQLite Rolling Multi-Turn Buffer & Longitudinal Tracker")
    print("[*] Local Database   : academic_vault.sqlite3")
    print("=" * 70)

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
