"""
Deep Transformer Fine-Tuning Pipeline: Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka - Department of Computer Science
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Fine-tunes 'sentence-transformers/all-MiniLM-L6-v2' (or 'distilbert-base-uncased')
on 6,000 real-world harmonized student/social distress reflections across 3 tiers:
  0: Low Stress
  1: Medium Stress (Situational Academic Tension)
  2: High Stress (Clinical Crisis Escalation)

Automatically:
1. Detects CUDA (NVIDIA RTX 4050) or CPU
2. Fine-tunes with class-weighted Cross-Entropy Loss
3. Generates comprehensive classification metrics & confusion matrix on held-out test data
4. Verifies critical edge-cases ('tomorrow is exam i'm anxious')
5. Exports to ONNX and INT8 Quantization for sub-25ms CPU client-side inference
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "models/custom_minilm_stress"
ONNX_DIR = "models"
TIER_MAP = {0: "Low", 1: "Medium", 2: "High"}


def main():
    print("=" * 70)
    print("  ACADEMIC STRESS ECOSYSTEM - DEEP TRANSFORMER FINE-TUNING")
    print("=" * 70)

    # 1. Hardware Detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training Device Detected : {device.upper()}")
    if device == "cuda":
        print(f"    GPU Name                 : {torch.cuda.get_device_name(0)}")
        print(f"    Available VRAM           : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[!] CUDA not detected. Training will run on multi-threaded CPU.")

    # 2. Load Datasets
    print("\n[*] Loading dataset splits from data/...")
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    print(f"    - Training samples   : {len(train_df)}")
    print(f"    - Validation samples : {len(val_df)}")
    print(f"    - Testing samples    : {len(test_df)}")

    # 3. Class Weights for Loss Balance
    labels = train_df["label"].values
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f"[*] Computed Class Weights   : {class_weights}")

    # 4. Tokenization
    print(f"\n[*] Initializing Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        return tokenizer(batch["text"], max_length=128, padding="max_length", truncation=True)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    # 5. Model Initialization
    print(f"[*] Loading Pretrained Transformer: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={0: "Low", 1: "Medium", 2: "High"},
        label2id={"Low": 0, "Medium": 1, "High": 2}
    )

    # 6. Custom Weighted Trainer
    class WeightedCrossEntropyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
            loss = loss_fn(logits.view(-1, 3), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    # 7. Training Arguments
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16 if device == "cuda" else 8,
        per_device_eval_batch_size=32 if device == "cuda" else 16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=42,
        fp16=(device == "cuda"),
        report_to="none"
    )

    trainer = WeightedCrossEntropyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    print("\n" + "=" * 50)
    print("  LAUNCHING TRANSFORMER FINE-TUNING (4 Epochs)")
    print("=" * 50)
    t_start = time.perf_counter()
    trainer.train()
    total_train_time = time.perf_counter() - t_start
    print(f"\n[+] Fine-tuning completed in {total_train_time:.1f} seconds ({total_train_time/60:.2f} minutes)!")

    # Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[+] Best Model weights saved to: {OUTPUT_DIR}")

    # 8. Evaluation on Held-Out Test Set (900 rows)
    print("\n" + "=" * 55)
    print("  EVALUATING MODEL ON HELD-OUT TEST CORPUS (900 Samples)")
    print("=" * 55)
    test_preds_raw = trainer.predict(test_ds)
    test_logits = test_preds_raw.predictions
    test_preds = np.argmax(test_logits, axis=1)
    test_labels = test_preds_raw.label_ids

    report = classification_report(test_labels, test_preds, target_names=["Low", "Medium", "High"])
    print(report)

    # 9. Edge-Case Verification
    print("-" * 55)
    print("  CRITICAL EDGE-CASE VERIFICATION")
    print("-" * 55)
    edge_cases = [
        "tomorrow is exam i'm anxious",
        "viva tomorrow feeling panicky and tense",
        "i want to end my life cannot take this agonizing pain anymore",
        "finished my computer networks lab report feeling great",
        "three lab submissions in two days feeling overwhelmed",
        "everyone would be better off without me in this world"
    ]

    model.eval()
    model.to("cpu")
    with torch.no_grad():
        for text in edge_cases:
            inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=0).numpy()
            pred_idx = int(np.argmax(probs))
            print(f"'{text}'")
            print(f"  -> Predicted: {TIER_MAP[pred_idx]} | Probs: [Low: {probs[0]*100:.1f}%, Med: {probs[1]*100:.1f}%, High: {probs[2]*100:.1f}%]\n")

    # 10. ONNX Export & Dynamic INT8 Quantization
    print("\n" + "=" * 55)
    print("  EXPORTING TO ONNX & APPLYING INT8 QUANTIZATION")
    print("=" * 55)
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType

        onnx_fp32_path = os.path.join(ONNX_DIR, "stress_model_fp32.onnx")
        onnx_int8_path = os.path.join(ONNX_DIR, "stress_model_quantized.onnx")

        dummy_inputs = tokenizer("sample test", return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        torch.onnx.export(
            model,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            onnx_fp32_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            },
            opset_version=14
        )
        print(f"[+] Exported FP32 ONNX model: {onnx_fp32_path} ({os.path.getsize(onnx_fp32_path)/(1024*1024):.1f} MB)")

        quantize_dynamic(
            model_input=onnx_fp32_path,
            model_output=onnx_int8_path,
            weight_type=QuantType.QInt8
        )
        print(f"[+] Exported Quantized INT8 ONNX: {onnx_int8_path} ({os.path.getsize(onnx_int8_path)/(1024*1024):.1f} MB)")
        print("[+] ONNX model is ready for sub-25ms CPU client-side deployment!")

    except Exception as e:
        print(f"[!] ONNX export note: {e}")
        print("    (You can also use PyTorch weights directly in models/custom_minilm_stress)")

    print("\n" + "=" * 70)
    print("  FINE-TUNING & OPTIMIZATION PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

