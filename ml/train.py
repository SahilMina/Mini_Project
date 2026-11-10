"""
Training & Evaluation Pipeline: Academic Stress Classification
Trains and evaluates the distilled academic stress classifier on Dreaddit/Campus data,
validates >80% accuracy objective, and benchmarks CPU inference latency (<300ms SLA).
"""
import os
import time
import json
import joblib
from typing import Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from ml.dataset import get_train_test_split, TIER_NAMES
from core.preprocessor import TextPreprocessor


def train_and_evaluate(model_save_path: str = "models/distilled_stress_engine.joblib") -> Dict[str, Any]:
    print("=" * 65)
    print("  ACADEMIC STRESS & CLINICAL REFERRAL - ML TRAINING PIPELINE")
    print("=" * 65)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    preprocessor = TextPreprocessor()

    train_texts, train_labels, test_texts, test_labels = get_train_test_split(test_ratio=0.20, seed=42)
    print(f"[*] Training samples: {len(train_texts)} | Testing samples: {len(test_texts)}")

    # Preprocess all texts with campus slang normalization
    norm_train_texts = [preprocessor.normalize(t) for t in train_texts]
    norm_test_texts = [preprocessor.normalize(t) for t in test_texts]

    # Pipeline: Word + Bigram TF-IDF with Calibrated Linear Support Vector Classifier
    # class_weight='balanced' ensures rare minority expressions aren't swamped by priors
    base_svc = LinearSVC(C=1.0, class_weight='balanced', dual=True, random_state=42, max_iter=3000)
    calibrated_clf = CalibratedClassifierCV(estimator=base_svc, cv=5)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("clf", calibrated_clf)
    ])

    print("[*] Training distilled classification head on campus distress corpus...")
    t_start = time.perf_counter()
    pipeline.fit(norm_train_texts, train_labels)
    train_duration = (time.perf_counter() - t_start) * 1000
    print(f"[+] Model trained in {train_duration:.2f} ms")

    # Evaluation
    predictions = pipeline.predict(norm_test_texts)
    accuracy = accuracy_score(test_labels, predictions)
    target_names = [TIER_NAMES[i] for i in range(3)]
    report = classification_report(test_labels, predictions, target_names=target_names, output_dict=True)

    print("\n" + "-" * 40)
    print(f"  ACCURACY ACHIEVED: {accuracy * 100:.2f}% (Target: >80%)")
    print("-" * 40)
    print(classification_report(test_labels, predictions, target_names=target_names))

    # CPU Latency Benchmark over 100 runs
    benchmark_samples = norm_test_texts * 10
    latencies = []
    for sample in benchmark_samples[:100]:
        t0 = time.perf_counter()
        _ = pipeline.predict_proba([sample])[0]
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    print("-" * 40)
    print(f"  CPU LATENCY BENCHMARK (<300 ms SLA)")
    print(f"  - Average Latency : {avg_latency:.2f} ms")
    print(f"  - 95th Percentile : {p95_latency:.2f} ms")
    print(f"  - Max Latency     : {max_latency:.2f} ms")
    print(f"  - SLA Status      : {'PASSED (Extremely Fast)' if avg_latency < 300 else 'FAILED'}")
    print("-" * 40)

    # Save artifact
    joblib.dump(pipeline, model_save_path)
    print(f"[+] Model saved successfully to '{model_save_path}'")

    metadata = {
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy * 100:.2f}%",
        "avg_latency_ms": round(avg_latency, 2),
        "meets_accuracy_goal": accuracy >= 0.80,
        "meets_latency_sla": avg_latency < 300.0,
        "classification_report": report,
        "model_file": model_save_path
    }

    meta_path = "models/model_metrics.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    train_and_evaluate()
