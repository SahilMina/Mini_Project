# ACADEMIC STRESS & CLINICAL REFERRAL ECOSYSTEM

**Author:** Sahil Mina (24UCOMP047)  
**Guide:** Prof. Srikantaiah K C, Department of Computer Science  
**Institution:** Central University of Karnataka, Kadaganchi, Kalaburagi – 585367, INDIA  
**Degree / Phase:** Mini Project Semester 5 (Phase II Architecture)  
**Status:** 100% Local On-Device Boundary | 26 Automated Test Suites Passing (100% Green)

---

## 1. Project Overview

The **Academic Stress & Clinical Referral Ecosystem** is a privacy-first, on-device academic stress screening assistant, longitudinal memory engine, and clinical triage portal designed specifically for university engineering students. 

Traditional mental health applications transmit personal disclosures and journal logs to cloud servers, deterring students from seeking timely intervention due to social stigma or fear of disciplinary penalties. This system enforces a **strict 100% on-device boundary**:
- **On-Device Multi-Tenant Text & Speech Ingestion:** Normalizes campus slang (*"viva"*, *"backlogs"*, *"CGPA"*, *"placement"*, *"arrears"*) and ingests speech across Indian English (`en-IN`) and regional Indic languages (Hindi `hi-IN`, Kannada `kn-IN`, Telugu `te-IN`, Tamil `ta-IN`).
- **Edge Transformer & Acoustic Screening:** Predicts clinical stress tiers (**Low**, **Medium**, **High**) with **83.0% - 86.0% accuracy** and a CPU inference latency of **17.58 ms** (exceeding the >80% accuracy target and 17x faster than the <300ms SLA).
- **Multi-Tenant AES-256-GCM Vault:** User master passphrases derived with **PBKDF2HMAC-SHA256 (100,000 iterations)**. Reflections are strictly isolated per student account (`WHERE user_id = ?`) with zero plaintext writes to disk.
- **Context Memory Engine:** Rolling 6-turn multi-turn conversational buffer stored in SQLite (`conversation_memory`) providing contextual continuity and longitudinal distress tracking.
- **Empathetic Dialogue Generator & Multilingual Translation:** Local dialogue generator synthesizes clinical CBT responses, translated into the student's native spoken dialect.
- **Deterministic Triage & Dual-Key Clinical Handoff:**
  - *Low Stress:* Milestone pacing and structured reflective exercises.
  - *Medium Stress:* Interactive CBT cards (4-7-8 parasympathetic breathing pacer with 432Hz harmonic acoustic tones, cognitive reframing thought records, 5-4-3-2-1 sensory grounding).
  - *High Stress:* Generates a cryptographically sealed, tamper-evident **Anonymous Clinical Referral Token** with student passcode for confidential consultation at the university health cell.
- **Separate Counselor & Admin Portal (`/admin`):** Dedicated portal for counselors, HODs, and administrators with a priority crisis queue and **dual-key zero-knowledge decryption** (counselors cannot view diary entries until the student presents their consultation passcode).

---

## 2. System Architecture Flow

```text
[Student Laptop (Strict On-Device Boundary)]
 ┌────────────────────────────────────────────────────────┐
 │ Student Interface (Typed Reflection / Indic Speech)    │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ Module 1: Ingestion, Normalization & Voice Pipeline    │
 │ - Campus Slang Regularizer (exam & viva vocabulary)    │
 │ - AI4Bharat Indic ASR (en-IN, hi-IN, kn-IN, te-IN)     │
 │ - Autonomic Acoustic Biomarkers (F0 Jitter, Shimmer)   │
 │ - Bidirectional Translation Layer (Indic <-> English)  │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ Module 2: On-Device Screening Inference Engine         │
 │ - Quantized all-MiniLM-L6-v2 ONNX (<25ms on CPU)       │
 │ - Multimodal Fusion (Text Semantics + Vocal Prosody)   │
 │ - Stratification: Low | Medium | High Stress           │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ Module 3: Multi-Tenant AES-256-GCM Encrypted Vault     │
 │ - PBKDF2HMAC-SHA256 (100,000 rounds) Key Derivation    │
 │ - Session-Scoped Ephemeral Key Mounting in RAM         │
 │ - Strict Account Isolation (Zero Cross-Student Leak)   │
 └─────────────┬────────────────────────────┬─────────────┘
               │                            │
               ▼                            ▼
 ┌──────────────────────────┐  ┌──────────────────────────┐
 │ Module 4: Context Memory │  │ Module 5: Triage Engine  │
 │ - Rolling 6-Turn Buffer  │  │ - Low/Med: CBT Exercises │
 │ - Empathetic Dialogue    │  │ - High: Sealed Referral  │
 │ - Longitudinal Tracking  │  │   Token + HMAC-SHA256    │
 └──────────────────────────┘  └────────────┬─────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ Module 6: Clinical Counselor & Admin Portal (/admin)   │
 │ - Priority Crisis Triage Queue for Clinical Psychologists│
 │ - Dual-Key Zero-Knowledge Consultation Decryption      │
 │   (Student Referral Passcode + Counselor Session Key)  │
 └────────────────────────────────────────────────────────┘
```

---

## 3. Directory Layout

```
├── synopsis.md                         # Phase I Synopsis & Research Specification
├── model_comparison.xlsx               # Model comparison spreadsheet (Formula-driven)
├── academic_stress_model_comparison.xlsx # Multi-sheet benchmark & hardware SLA workbook
├── run.py                              # Dev server launcher (FastAPI + Uvicorn hot-reload)
├── academic_vault.sqlite3              # Multi-tenant encrypted SQLite database
├── core/
│   ├── __init__.py
│   ├── preprocessor.py                 # Module 1: Campus slang regularizer
│   ├── classifier.py                   # Module 2: ONNX MiniLM / Calibrated screening head
│   ├── translator.py                   # Module 1: Bidirectional multilingual translation
│   ├── generator.py                    # Module 4: Empathetic CBT dialogue generator
│   └── memory.py                       # Module 4: Rolling 6-turn context memory buffer
├── crypto/
│   ├── __init__.py
│   ├── db.py                           # SQLite multi-tenant DB manager & RBAC schema
│   └── vault.py                        # Module 3: AES-256-GCM vault & ephemeral mounting
├── triage/
│   ├── __init__.py
│   ├── dispatcher.py                   # Module 5: Deterministic triage rule engine
│   └── referral.py                     # Module 5: Sealed referral token generator
├── ml/
│   ├── __init__.py
│   ├── dataset.py                      # Multi-source dataset harmonizer
│   └── train.py                        # ML training pipeline & accuracy benchmark
├── backend/
│   ├── __init__.py
│   ├── auth.py                         # PBKDF2HMAC authentication & RBAC manager
│   └── main.py                         # FastAPI REST endpoints & route guards
├── voice/
│   ├── __init__.py                     # Multimodal voice screening pipeline
│   ├── models.py                       # AI4Bharat language & model specifications
│   ├── acoustic.py                     # Acoustic stress biomarkers (Jitter, Shimmer, WPM)
│   ├── asr.py                          # AI4Bharat speech recognition engine
│   ├── guidance.py                     # 432Hz harmonic tone & breathing audio synthesizer
│   └── pipeline.py                     # Multimodal acoustic + text decision fusion
├── templates/
│   ├── index.html                      # Student screening & chat interface
│   ├── login.html                      # Student login, registration & portal navigator
│   └── admin.html                      # Clinical Counselor, HOD & Admin triage desk
├── static/
│   ├── css/
│   │   └── app.css                     # Clinical dark theme & glassmorphic UI styles
│   └── js/
│       └── app.js                      # Client app controller & robust error parser
├── tests/
│   ├── test_auth_isolation.py          # Multi-tenant account isolation tests
│   ├── test_system_expansion.py        # System expansion & dual-key decryption tests
│   ├── test_api.py                     # FastAPI REST API endpoint test suite
│   ├── test_voice.py                   # AI4Bharat speech & acoustic biomarker tests
│   ├── test_vault.py                   # AES-256-GCM cryptographic vault tests
│   ├── test_pipeline.py                # End-to-end screening integration tests
│   └── test_conversational_responder.py# Dialogue generation tests
└── models/
    ├── distilled_stress_engine.joblib  # Trained baseline classifier weights
    ├── model_metrics.json              # Model evaluation metrics & latency report
    └── minilm_onnx/                    # Quantized INT8 ONNX transformer artifacts
```

---

## 4. Default Portal Credentials

For institutional evaluation and grading, default accounts are pre-seeded in the database:

### A. Student Portal (`/login` → `/`)
Students can either register a new account on `/login` or use the primary student account:
- **Student ID:** `24UCOMP047`
- **Email:** `sahilmina55555@gmail.com`
- **Password:** `Password@123`
- **Role:** `student`

### B. Clinical Counselor & Staff Portal (`/admin`)
Accessible via the **"Clinical Triage Portal"** button on `/login` or directly at `http://127.0.0.1:8000/admin`:

| Portal Role | Staff / Employee ID | Email | Password | Scope & Privileges |
| :--- | :--- | :--- | :--- | :--- |
| **Clinical Counselor** | `COUNSELOR01` | `counselor@cuk.ac.in` | `Counselor@123` | Priority crisis queue, student passcode decryption |
| **HOD (Head of Dept)** | `HOD_CS` | `srikantaiah@cuk.ac.in` | `HodPassword@123` | Departmental aggregated escalation oversight |
| **System Admin** | `ADMIN01` | `admin@cuk.ac.in` | `AdminPass@123` | Full audit logs, session management, vault health |

*(Quick auto-fill buttons are provided directly on the `/admin` login screen for fast testing).*

---

## 5. Getting Started & Running Locally

### Step 1: Environment Setup
Ensure Python 3.11+ is installed. Activate the project virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2: Run the Complete Automated Test Suite
Run all 26 unit, integration, and security test suites:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
*(All 26 tests will pass with `OK` in ~5 seconds).*

### Step 3: Launch the Application Server
Run the application server with hot-reload enabled:
```powershell
.\.venv\Scripts\python.exe run.py
```

### Step 4: Access the Application
- **Student Portal:** [`http://127.0.0.1:8000/login`](http://127.0.0.1:8000/login) *(Unauthenticated access to `/` automatically redirects here)*
- **Main Chat & Screening App:** [`http://127.0.0.1:8000`](http://127.0.0.1:8000) *(Accessible once logged in)*
- **Counselor & Admin Portal:** [`http://127.0.0.1:8000/admin`](http://127.0.0.1:8000/admin)

---

## 6. Empirical Model Benchmarks

Empirical performance evaluated across **900 held-out test reflections** (300 Low, 300 Medium, 300 High):

| Model Architecture | Features | Slang Normalized Accuracy | Raw Text Accuracy | Relative Points vs Baseline | CPU Latency | Model Size | Production Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (TF-IDF)** | Textual semantics | 78.1% | 77.6% | 0.0 pts | 0.04 ms | 0.4 MB | Baseline |
| **Random Forest (300 trees)** | Textual semantics | 75.4% | 74.9% | -2.7 pts | 0.16 ms | 42.5 MB | Evaluated |
| **Gradient Boosting** | Textual semantics | 76.3% | 75.3% | -1.8 pts | 0.05 ms | 1.8 MB | Evaluated |
| 🔬 **DistilBERT-base-uncased (Ours)** | **Textual semantics** | **82.1%** | **80.1%** | **+4.0 pts** | **145.20 ms** | **268.0 MB** | **Phase I Model** |
| 🌟 **Quantized all-MiniLM ONNX (Ours)**| **Textual semantics** | **83.0%** | **81.0%** | **+4.9 pts** | **17.58 ms** | **22.9 MB** | **Phase II Deployed** |
| **Logistic Regression (TF-IDF)** | + vocal acoustic biomarkers | 80.6% | 79.1% | 0.0 pts | 0.04 ms | 0.4 MB | Baseline |
| **Random Forest (300 trees)** | + vocal acoustic biomarkers | 79.8% | 78.2% | -0.8 pts | 0.16 ms | 42.5 MB | Evaluated |
| **Gradient Boosting** | + vocal acoustic biomarkers | 80.2% | 78.8% | -0.3 pts | 0.05 ms | 1.8 MB | Evaluated |
| 🔬 **DistilBERT-base-uncased (Ours)** | **+ vocal acoustic biomarkers** | **85.1%** | **82.7%** | **+4.6 pts** | **145.20 ms** | **268.0 MB** | **Phase I Model** |
| 🌟 **Quantized all-MiniLM ONNX (Ours)**| **+ vocal acoustic biomarkers** | **86.0%** | **83.6%** | **+5.4 pts** | **17.58 ms** | **22.9 MB** | **Phase II Deployed** |

*(Detailed formula-driven spreadsheet available in [`academic_stress_model_comparison.xlsx`](file:///d:/Coding%20Resouces%20&%20Files/Mini%20Project%20Sem%205/academic_stress_model_comparison.xlsx)).*

---

## 7. Key Privacy & Security Controls

1. **Multi-Tenant PBKDF2HMAC Key Derivation:**
   Each student account has a unique 16-byte cryptographic salt (`salt_hex`). The 256-bit AES encryption key is derived dynamically upon sign-in using 100,000 PBKDF2HMAC-SHA256 iterations and mounted ephemerally in RAM.
2. **Authenticated AES-256-GCM Ciphertext:**
   Each reflection is encrypted with a unique 12-byte initialization vector (nonce) and 16-byte authentication tag. Any tampering triggers immediate authentication failure.
3. **Dual-Key Zero-Knowledge Clinical Decryption:**
   When a student is escalated to the High Stress triage tier, the referral is registered in the counselor's queue with masked ciphertext. The counselor **cannot** decrypt the reflection unless the student presents their 6-character referral passcode during their in-person consultation.
4. **Session Key Memory Purging:**
   Logging out destroys the session token and purges the active AES key from RAM. Locked vaults expose only ciphertext previews.

---

## 8. Bibliography

- **[1] J. Gratch et al.**, *"The Distress Analysis Interview Corpus,"* LREC '14, 2014.
- **[2] K. Yang et al.**, *"MentaLLaMA: Interpretable mental health analysis on social media,"* WWW '24, 2024.
- **[3] A. G. Jaber**, *"Privacy-preserving and secure AI-based stress monitoring,"* IJRIAS, 2026.
- **[4] T. Javed et al.**, *"Towards building open-domain speech systems for Indic languages,"* ACL 2022.
- **[5] S. Ji et al.**, *"MentalBERT: Pretrained language models for mental healthcare,"* LOUHI 2022.
- **[6] J. Groth**, *"On the size of pairing-based non-interactive arguments,"* EUROCRYPT 2016.
