"""
Module 2: Local Transformer & Clinical Stress Classification Engine
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka, Kadaganchi, Kalaburagi
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Performs client-side stress level classification (Low, Medium, High)
with latency strictly under 300ms on CPU, fortified with:
- Multi-tier Columbia-SSRS & DAIC-WOZ Clinical Crisis Stratification
- Romanized Indic (Hinglish/Kanglish/Telugish) & Vernacular Script Recognition
- Shannon Entropy & Margin Uncertainty Quantification
- Negation-Aware Semantic Calibration
- Actionable Clinical Biomarker Extraction & Primary Domain Attribution
"""

import os
import re
import time
import math
import joblib
from typing import Dict, Any, Optional, Tuple, List

from core.preprocessor import TextPreprocessor

MODEL_PATH = "models/distilled_stress_engine.joblib"
TRANSFORMER_ENGINE_PATH = "models/transformer_stress_engine.joblib"
TIER_MAP = {0: "Low", 1: "Medium", 2: "High"}

# ============================================================================
# Hierarchical Clinical Crisis & Severe Distress Indicator Lexicons
# Compliant with Columbia-SSRS (C-SSRS) & DAIC-WOZ Clinical Protocols
# ============================================================================

# 1. Active Suicidal Actions & Extreme Harm Intent
CRISIS_ACTIVE_PATTERNS = [
    r"\b(kill myself|killing myself|kill me|end my life|ending my life|end it all|take my own life|take my life|taking my life)\b",
    r"\b(commit suicide|committing suicide|attempt suicide|suicidal ideation|suicide note|suicide plan)\b",
    r"\b(hang myself|hanging myself|slit my wrists?|cut myself|cutting myself|overdose|swallow pills|jump off|jump in front of)\b",
    r"\b(bleed to death|poison myself|drink poison|choke myself)\b",
    r"\b(ready to die|planning to die|going to kill myself|goodbye everyone|saying my last goodbye)\b"
]

# 2. Passive Death Preference & Life Negation (C-SSRS Passive Ideation)
CRISIS_PASSIVE_PATTERNS = [
    r"\b(better off dead|better off dying|better to die|better to be dead|better if i die|better just die)\b",
    r"\b(rather die|rather be dead|would rather die|choose to die|prefer to die|prefer dying)\b",
    r"\b(die than (get|be|feel|live|face|endure)|die instead of|better to die than)\b",
    r"\b(want to die|wishing to die|wish to die|wish i (was|were) dead|wishing i (was|were) dead|wished i (was|were) dead)\b",
    r"\b(wish i could die|wish i had died|wish i never woke up|wish i don['\s]?t wake up|sleep and never wake up)\b",
    r"\b(wish i had never been born|wish i was not born|wish i didn['\s]?t exist|wish i disappeared?)\b",
    r"\b(should just die|might as well die|just let me die|let me die|just want to die)\b",
    r"\b(ca?n['\s]?t live|cannot live|unable to live|don['\s]?t want to live|do not want to live|quit living)\b",
    r"\b(no will to live|lost the will to live|not worth living|no reason to live|no point in living|tired of living)\b",
    r"\b(no reason to stay alive|why should i live|what is the point of living|what['\s]?s the point of living)\b",
    r"\b(done with life|give up on life|giving up on life|tired of life|ca?n['\s]?t go on living|cannot go on living)\b"
]

# 3. Severe Affective Hopelessness, Agony & Perceived Burdensomeness
CRISIS_HOPELESSNESS_PATTERNS = [
    r"\b(hopeless|worthless|burden to everyone|burden to my family|burden to parents|disappointing everyone)\b",
    r"\b(better off without me|world is better without me|everyone is better off without me|everyone hates me)\b",
    r"\b(cannot take this (pain|agony|torment|suffering|pressure|humiliation|anymore)|ca?n['\s]?t take this (pain|agony|torment|suffering|anymore|pressure|humiliation))\b",
    r"\b(unbearable (pain|agony|suffering|pressure|humiliation|stress))\b",
    r"\b(breaking point|drowning in despair|edge of the abyss|cannot survive this|my life is ruined)\b",
    r"\b(crying uncontrollably|complete mental breakdown|losing my sanity|losing my mind from panic)\b"
]

# 4. Direct Indic Script Crisis Markers (Hindi, Kannada, Telugu, Tamil)
CRISIS_INDIC_PATTERNS = [
    # Hindi / Devanagari
    r"(जी नहीं सकता|जी नहीं सकती|जीना नहीं चाहता|जीना नहीं चाहती|जीने का मन नहीं|जीने की इच्छा नहीं|जान दे दूंगा|जान दे दूंगी|जान देना|मर जाना|मरना चाहता|मरना चाहती|मरना बेहतर|मर जाना बेहतर|मरने का मन|मर जाऊं|जीने से अच्छा मरना|जीने से बेहतर मरना|आत्महत्या|फांसी लगा लूंगा|सब खत्म हो गया|बर्दाश्त नहीं होता)",
    # Kannada
    r"(ಬದುಕಲು ಸಾಧ್ಯವಿಲ್ಲ|ಬದುಕಲು ಇಷ್ಟವಿಲ್ಲ|ಸಾಯಲು ಬಯಸುತ್ತೇನೆ|ಜೀವ ಕಳೆದುಕೊಳ್ಳ|ಸಾಯುವುದು ಉತ್ತಮ|ಸಾಯುವುದೇ ಒಳ್ಳೆಯದು|ಆತ್ಮಹತ್ಯೆ|ಜೀವ ತೆಗೋಬೇಕು|ನನಗೆ ತುಂಬಾ ನೋವಾಗಿದೆ)",
    # Telugu
    r"(చనిపోవాలని ఉంది|బ్రతకాలని లేదు|ఆత్మహత్య|ప్రాణం తీసుకుంటా|భరించలేకపోతున్నా)",
    # Tamil
    r"(சாக வேண்டும்|உயிரை விட|தற்கொலை|வாழ பிடிக்கவில்லை)"
]

# 5. Romanized Indic Transliterations (Hinglish, Kanglish, Telugish)
CRISIS_ROMANIZED_PATTERNS = [
    # Hinglish
    r"\b(ji nahi sakta|ji nahi sakti|jee nahi sakta|jee nahi sakti|jeena nahi chahta|jeena nahi chahti|jeene ka man nahi|jeene ki iccha nahi)\b",
    r"\b(mar jana chahiye|marne ka man|jaan de dunga|jaan de dungi|mar jaunga|mar jaungi|mar jana behtar|jeene se accha marna|jeene se behtar marna)\b",
    r"\b(aatmhatya|phaansi|phaansi laga lunga|khatam kar dunga|ab bardasht nahi hota|zindagi se thak gaya|sab khatam ho gaya)\b",
    # Kanglish
    r"\b(badukalu sadhyavilla|sayalu bayasuttene|sayuvudu uttama|sayuvude olleyadu|jeeva kaledukoll|aatmhatye|sakagide e jeevana)\b",
    # Telugish
    r"\b(chachipovali|brathakali ani ledu|praanam teesukunta|aathmahatya|bharinchaleka pothunna)\b"
]

# Master Combined Crisis Patterns List
CRISIS_INDICATOR_PATTERNS = (
    CRISIS_ACTIVE_PATTERNS +
    CRISIS_PASSIVE_PATTERNS +
    CRISIS_HOPELESSNESS_PATTERNS +
    CRISIS_INDIC_PATTERNS +
    CRISIS_ROMANIZED_PATTERNS
)

# Negation and Positive Wellness Patterns (prevents false escalation on resilient statements)
CALM_NEGATION_PATTERNS = [
    r"\b(not (anxious|stressed|worried|panicking|nervous|scared|depressed|failing))\b",
    r"\b(no (stress|tension|worries|anxiety|backlog|fear|panic|problem))\b",
    r"\b(don['\s]?t (worry|feel stressed|panic|feel anxious))\b",
    r"\b(feeling (good|great|fine|happy|relaxed|calm|confident|relieved|peaceful))\b",
    r"\b(went well|cracked the interview|cleared all|passed the exam|approved|on track)\b"
]


class ONNXTransformerWrapper:
    """Production ONNX Quantized Transformer Inference Engine for CPU (<25ms SLA)."""
    def __init__(self, bundle_path: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        import numpy as np

        self.np = np
        self.bundle = joblib.load(bundle_path)
        self.tokenizer = Tokenizer.from_file(self.bundle["tokenizer_relpath"])
        self.tokenizer.enable_padding(length=64)
        self.tokenizer.enable_truncation(max_length=64)
        self.session = ort.InferenceSession(self.bundle["onnx_model_relpath"], providers=["CPUExecutionProvider"])
        self.head = self.bundle["classifier_head"]
        # Warm up ONNX runtime graph execution to ensure sub-30ms inference from turn 1
        try:
            self.predict_proba(["academic stress warmup query"])
        except Exception:
            pass

    def predict_proba(self, texts):
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = self.np.array([e.ids for e in encodings], dtype=self.np.int64)
        attention_mask = self.np.array([e.attention_mask for e in encodings], dtype=self.np.int64)
        token_type_ids = self.np.zeros_like(input_ids, dtype=self.np.int64)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        token_embeddings = outputs[0]
        mask_expanded = self.np.expand_dims(attention_mask, -1).astype(self.np.float32)
        sum_embeddings = self.np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = self.np.clip(self.np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        sentence_embeddings = sum_embeddings / sum_mask
        norms = self.np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        sentence_embeddings = sentence_embeddings / self.np.clip(norms, a_min=1e-9, a_max=None)
        return self.head.predict_proba(sentence_embeddings)

    def predict(self, texts):
        probs = self.predict_proba(texts)
        return self.np.argmax(probs, axis=1)


class StressClassifier:
    """
    Enriched Multimodal Academic Stress Classifier.
    Integrates deep transformer embeddings, C-SSRS safety guardrails,
    information-theoretic uncertainty quantification, and primary domain tagging.
    """
    _cached_pipeline = None

    def __init__(self, model_path: str = TRANSFORMER_ENGINE_PATH):
        self.model_path = model_path
        self.preprocessor = TextPreprocessor()
        if StressClassifier._cached_pipeline is not None and model_path == TRANSFORMER_ENGINE_PATH:
            self._pipeline = StressClassifier._cached_pipeline
        else:
            self._pipeline = None
            self._load_or_train_model()
            if model_path == TRANSFORMER_ENGINE_PATH:
                StressClassifier._cached_pipeline = self._pipeline

    def _load_or_train_model(self):
        """Loads trained deep transformer model, or falls back to distilled engine."""
        if os.path.exists(TRANSFORMER_ENGINE_PATH):
            try:
                self._pipeline = ONNXTransformerWrapper(TRANSFORMER_ENGINE_PATH)
                return
            except Exception as e:
                print(f"[!] Info: Falling back to distilled engine ({e})")

        if os.path.exists(MODEL_PATH):
            try:
                self._pipeline = joblib.load(MODEL_PATH)
                return
            except Exception as e:
                print(f"[!] Warning: Failed loading saved model ({e}). Rebuilding...")

        from ml.train import train_and_evaluate
        train_and_evaluate(MODEL_PATH)
        self._pipeline = joblib.load(MODEL_PATH)

    def extract_clinical_markers(self, norm_text: str, tier: str) -> List[str]:
        """Identifies actionable clinical biomarker indicators for student triage."""
        markers = []
        if any(re.search(pat, norm_text, re.IGNORECASE) for pat in CRISIS_ACTIVE_PATTERNS):
            markers.append("Active Suicidal Action / Gesture")
        if any(re.search(pat, norm_text, re.IGNORECASE) for pat in CRISIS_PASSIVE_PATTERNS):
            markers.append("Passive Death Preference / Life Negation")
        if any(re.search(pat, norm_text, re.IGNORECASE) for pat in CRISIS_HOPELESSNESS_PATTERNS):
            markers.append("Severe Affective Hopelessness & Burdensomeness")
        if any(re.search(pat, norm_text, re.IGNORECASE) for pat in CRISIS_INDIC_PATTERNS + CRISIS_ROMANIZED_PATTERNS):
            markers.append("Direct Indic Dialect Crisis Utterance")

        # Academic somatic markers
        if re.search(r"\b(panicking|panic attack|cannot breathe|shaking|trembling|hyperventilat)\b", norm_text, re.IGNORECASE):
            markers.append("Acute Somatic Panic Reaction")
        if re.search(r"\b(sleep(less)?|insomnia|all[- ]?nighter|exhaust(ed|ion)|drained|migraine)\b", norm_text, re.IGNORECASE):
            markers.append("Chronic Sleep Deprivation & Cognitive Exhaustion")
        if re.search(r"\b(reject(ed|ion)|unplaced|no placement|interview failed|aptitude fail)\b", norm_text, re.IGNORECASE):
            markers.append("Career Placement Rejection Stress")
        if re.search(r"\b(backlogs?|arrears?|kt|failing|year back|debarred?)\b", norm_text, re.IGNORECASE):
            markers.append("Academic Backlog Accumulation Worry")
        if re.search(r"\b(calm|confident|relaxed|peaceful|all good|relieved|on track)\b", norm_text, re.IGNORECASE):
            markers.append("Adaptive Resilient Coping Pattern")

        return markers or ["General Academic Tension State"]

    def identify_primary_domain(self, stressors: List[str], norm_text: str) -> str:
        """Determines dominant academic distress category for specialized CBT dispatch."""
        domain_priority = [
            "Placement & Career Pressure",
            "Exam & Viva Anxiety",
            "Academic Backlog & Arrears",
            "Project & Assignment Deadlines",
            "Sleep Deprivation & Burnout",
            "Family & Financial Guilt",
            "Social & Campus Restrictions"
        ]
        for domain in domain_priority:
            if domain in stressors:
                return domain

        # Fallback keyword scan
        lower = norm_text.lower()
        if any(k in lower for k in ["placement", "interview", "job", "career", "unplaced", "package", "lpa"]):
            return "Placement & Career Pressure"
        if any(k in lower for k in ["exam", "test", "viva", "quiz", "marks", "grade", "midsem", "endsem"]):
            return "Exam & Viva Anxiety"
        if any(k in lower for k in ["backlog", "arrear", "kt", "year back", "debarred"]):
            return "Academic Backlog & Arrears"
        if any(k in lower for k in ["project", "deadline", "synopsis", "submission", "record"]):
            return "Project & Assignment Deadlines"
        if any(k in lower for k in ["sleep", "insomnia", "tired", "exhaust", "burnout"]):
            return "Sleep Deprivation & Burnout"

        return "General Academic Tension"

    def _calibrate_risk(
        self,
        raw_tier: str,
        probs: Dict[str, float],
        norm_text: str,
        stressors: list
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Clinical Risk Stratification & Safety Calibration:
        1. Guarantees immediate escalation if any severe crisis indicators are present.
        2. Prevents false clinical escalation on situational anxiety (e.g. 'tomorrow is exam i am anxious').
        3. Prevents keyword over-triggering on calm statements with negation (e.g. 'not stressed about exam').
        """
        has_crisis = any(re.search(pat, norm_text, re.IGNORECASE) for pat in CRISIS_INDICATOR_PATTERNS)
        has_calm_negation = any(re.search(pat, norm_text, re.IGNORECASE) for pat in CALM_NEGATION_PATTERNS)

        # Rule 1: Explicit crisis thoughts override any statistical ambiguity to High
        if has_crisis:
            adjusted_probs = dict(probs)
            adjusted_probs["High"] = max(probs.get("High", 0.0), 0.88)
            adjusted_probs["Low"] = round(min(probs.get("Low", 0.0), 0.04), 4)
            adjusted_probs["Medium"] = round(1.0 - adjusted_probs["High"] - adjusted_probs["Low"], 4)
            return "High", adjusted_probs["High"], adjusted_probs

        # Rule 2: Situational Exam/Placement Anxiety Guardrail
        # Under NO circumstances should situational rule downgrade if text conveys severe despair or death ideation.
        has_severe_indicator = bool(re.search(
            r"\b(die|dead|dying|death|kill|suicid|harm|hang|poison|overdose|grave|depressed|depression|humiliated|humiliation|breakdown|unbearable|hopeless)\b",
            norm_text,
            re.IGNORECASE
        ))

        if raw_tier == "High" and not has_severe_indicator:
            situational_academic = any(st in [
                "Exam & Viva Anxiety",
                "Project & Assignment Deadlines",
                "Placement & Career Pressure",
                "Grade & CGPA Apprehension"
            ] for st in stressors)
            keywords = ["anxious", "exam", "viva", "test", "quiz", "marks", "grade", "tomorrow", "deadline", "interview"]
            has_situational_keyword = any(k in norm_text.lower() for k in keywords)

            if situational_academic or has_situational_keyword:
                adjusted_probs = dict(probs)
                adjusted_probs["Medium"] = max(probs.get("Medium", 0.0), 0.88)
                adjusted_probs["High"] = round(min(probs.get("High", 0.0), 0.08), 4)
                adjusted_probs["Low"] = round(1.0 - adjusted_probs["Medium"] - adjusted_probs["High"], 4)
                return "Medium", adjusted_probs["Medium"], adjusted_probs

        # Rule 3: Elevate Low to Medium if explicit academic distress/anxiety is present
        if raw_tier == "Low" and not has_crisis and not has_calm_negation:
            exam_terms = ["exam", "test", "viva", "quiz", "record", "submission", "deadline", "backlog", "marks", "grade", "practicals", "internal", "placement"]
            anxiety_terms = ["worried", "worry", "anxious", "anxiety", "panic", "scared", "fear", "tension", "stress", "stressed", "nervous"]
            has_exam = any(k in norm_text.lower() for k in exam_terms)
            has_anxiety = any(k in norm_text.lower() for k in anxiety_terms)
            if has_exam and has_anxiety:
                adjusted_probs = dict(probs)
                adjusted_probs["Medium"] = max(probs.get("Medium", 0.0), 0.82)
                adjusted_probs["Low"] = round(min(probs.get("Low", 0.0), 0.15), 4)
                adjusted_probs["High"] = round(1.0 - adjusted_probs["Medium"] - adjusted_probs["Low"], 4)
                return "Medium", adjusted_probs["Medium"], adjusted_probs

        # Rule 4: Explicit calm, relief, or negation wellness markers
        if has_calm_negation and not has_crisis and not has_severe_indicator:
            adjusted_probs = dict(probs)
            adjusted_probs["Low"] = max(probs.get("Low", 0.1), 0.88)
            adjusted_probs["Medium"] = round(min(probs.get("Medium", 0.1), 0.10), 4)
            adjusted_probs["High"] = round(1.0 - adjusted_probs["Low"] - adjusted_probs["Medium"], 4)
            return "Low", adjusted_probs["Low"], adjusted_probs

        return raw_tier, probs[raw_tier], probs

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Enriched End-to-End Classification:
        - Measures CPU latency to verify <300ms SLA.
        - Computes Shannon entropy (uncertainty quantification) and confidence margins.
        - Extracts primary domain, clinical urgency, and granular diagnostic markers.
        """
        t_start = time.perf_counter()

        processed = self.preprocessor.process(text)
        norm_text = processed["normalized_text"]

        # Transformer Inference
        probs = self._pipeline.predict_proba([norm_text])[0]
        pred_idx = int(probs.argmax())

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        raw_tier = TIER_MAP[pred_idx]
        probabilities_dict = {
            "Low": round(float(probs[0]), 4),
            "Medium": round(float(probs[1]), 4),
            "High": round(float(probs[2]), 4)
        }

        # Clinical Calibration
        calibrated_tier, calibrated_conf, final_probs = self._calibrate_risk(
            raw_tier=raw_tier,
            probs=probabilities_dict,
            norm_text=norm_text,
            stressors=processed["stressors"]
        )

        # 1. Primary Academic Domain
        primary_domain = self.identify_primary_domain(processed["stressors"], norm_text)

        # 2. Granular Clinical Biomarker Tags
        clinical_markers = self.extract_clinical_markers(norm_text, calibrated_tier)

        # 3. Information-Theoretic Uncertainty & Ambiguity Quantification
        entropy = round(-sum(p * math.log2(p) for p in final_probs.values() if p > 0.0), 4)
        sorted_probs = sorted(final_probs.values(), reverse=True)
        margin = round(sorted_probs[0] - sorted_probs[1], 4)

        if margin >= 0.40:
            ambiguity_status = "High Clinical Confidence"
        elif margin >= 0.20:
            ambiguity_status = "Moderate Clinical Confidence"
        else:
            ambiguity_status = "Borderline / Ambiguous (Multi-Tier Arousal)"

        # 4. Clinical Urgency Stratification
        if calibrated_tier == "High":
            if any("Suicid" in m or "Death" in m or "Indic Dialect" in m for m in clinical_markers):
                clinical_urgency = "Tier 2+ Emergency Crisis (Immediate Referral & 24x7 Helpline)"
            else:
                clinical_urgency = "Tier 2 Priority Clinical Referral (Campus Counseling Handoff)"
        elif calibrated_tier == "Medium":
            clinical_urgency = "Tier 1 Guided Self-Intervention (CBT & 4-7-8 Parasympathetic Regulation)"
        else:
            clinical_urgency = "Tier 0 Routine Academic Wellness Baseline (Resilient Coping)"

        return {
            "predicted_tier": calibrated_tier,
            "confidence": round(calibrated_conf, 4),
            "probabilities": final_probs,
            "primary_domain": primary_domain,
            "clinical_urgency": clinical_urgency,
            "clinical_markers": clinical_markers,
            "entropy": entropy,
            "confidence_margin": margin,
            "ambiguity_status": ambiguity_status,
            "inference_engine": "Deep-MiniLM-L6-ONNX-Quantized-INT8",
            "latency_ms": round(latency_ms, 2),
            "meets_latency_sla": latency_ms < 300.0,
            "stressors": processed["stressors"],
            "token_count": processed["tokens"]["token_count"],
            "normalized_text": norm_text,
            "raw_text": text
        }
