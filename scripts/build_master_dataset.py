"""
Master Dataset Builder & Harmonizer: Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka - Department of Computer Science
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Harmonizes multi-source raw corpora into a balanced 3-Tier Clinical Ground Truth:
- Tier 0 (Low Stress): Normal, calm, productive routine, everyday life (Dreaddit non-stress, Mental Health Normal, SAD non-stress).
- Tier 1 (Medium Stress): Situational academic anxiety, exam panic, deadline pressure, sleep loss, backlog worry (Dreaddit stress, SAD School/Work, Mental Health Anxiety).
- Tier 2 (High Stress): Clinical crisis, depression, hopelessness, self-harm / suicidal thoughts (Suicide Watch, Mental Health Suicidal/Depression).
"""
import os
import sys
import re
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RANDOM_SEED = 42
TARGET_SAMPLES_PER_TIER = 2000  # 2,000 per tier = 6,000 total high-quality samples

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def clean_text(text: str) -> str:
    """Cleans text: removes URLs, Reddit markup, handles whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove Reddit formatting / artifacts
    text = re.sub(r"\[(deleted|removed)\]", "", text)
    text = re.sub(r"r/\w+", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    # Remove email addresses and phone numbers (PII)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{10,}\b", "", text)
    # Normalize excessive punctuation / spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def filter_by_length(texts, min_words=8, max_words=75):
    """Filters reflections to realistic student reflection length."""
    valid = []
    for t in texts:
        words = t.split()
        if min_words <= len(words) <= max_words:
            valid.append(t)
    return valid

def main():
    print("=" * 70)
    print("  ACADEMIC STRESS ECOSYSTEM - MASTER DATASET HARMONIZATION")
    print("=" * 70)
    os.makedirs("data", exist_ok=True)

    tier_0_texts = []
    tier_1_texts = []
    tier_2_texts = []

    # -------------------------------------------------------------------------
    # 1. Harvest from Mental Health Multi-Category Dataset
    # -------------------------------------------------------------------------
    mh_path = "data/raw/mental_health_text/mental_heath_feature_engineered.csv"
    if os.path.exists(mh_path):
        print("[*] Processing Mental Health dataset...")
        df_mh = pd.read_csv(mh_path, usecols=["text", "status"]).dropna()
        
        # Normal -> Tier 0
        mh_normal = [clean_text(t) for t in df_mh[df_mh["status"] == "Normal"]["text"]]
        tier_0_texts.extend(filter_by_length(mh_normal))

        # Anxiety -> Tier 1 (Situational distress)
        mh_anxiety = [clean_text(t) for t in df_mh[df_mh["status"] == "Anxiety"]["text"]]
        tier_1_texts.extend(filter_by_length(mh_anxiety))

        # Suicidal / Severe Depression -> Tier 2 (Clinical Crisis)
        mh_crisis = [clean_text(t) for t in df_mh[df_mh["status"] == "Suicidal"]["text"]]
        tier_2_texts.extend(filter_by_length(mh_crisis))
        print(f"    Added from Mental Health: Tier 0={len(tier_0_texts)}, Tier 1={len(tier_1_texts)}, Tier 2={len(tier_2_texts)}")

    # -------------------------------------------------------------------------
    # 2. Harvest from Dreaddit Dataset (Reddit Stress & Social Anxiety)
    # -------------------------------------------------------------------------
    dreaddit_train_path = "data/raw/dreaddit-train.csv"
    dreaddit_test_path = "data/raw/dreaddit-test.csv"
    if os.path.exists(dreaddit_train_path):
        print("[*] Processing Dreaddit social stress corpora...")
        df_dt = pd.read_csv(dreaddit_train_path, usecols=["text", "label"]).dropna()
        if os.path.exists(dreaddit_test_path):
            df_dte = pd.read_csv(dreaddit_test_path, usecols=["text", "label"]).dropna()
            df_dt = pd.concat([df_dt, df_dte], ignore_index=True)

        # Split long posts into sentences/chunks
        d_nonstress = [clean_text(t) for t in df_dt[df_dt["label"] == 0]["text"]]
        d_stress = [clean_text(t) for t in df_dt[df_dt["label"] == 1]["text"]]

        tier_0_texts.extend(filter_by_length(d_nonstress))
        tier_1_texts.extend(filter_by_length(d_stress))
        print(f"    Added from Dreaddit: Tier 0 Total={len(tier_0_texts)}, Tier 1 Total={len(tier_1_texts)}")

    # -------------------------------------------------------------------------
    # 3. Harvest from SAD (Stress Annotated Dataset - Everyday & School Stressors)
    # -------------------------------------------------------------------------
    sad_path = "data/raw/SAD/SAD_v1.xlsx"
    if os.path.exists(sad_path):
        print("[*] Processing SAD (Stress Annotated Dataset)...")
        df_sad = pd.read_excel(sad_path)
        sad_sentences = df_sad["sentence"].astype(str).dropna()
        
        # SAD non-stress -> Tier 0
        sad_nonstress = [clean_text(t) for t in sad_sentences[df_sad["is_stressor"] == 0]]
        tier_0_texts.extend(filter_by_length(sad_nonstress))

        # SAD school & moderate stressors -> Tier 1
        sad_school = [clean_text(t) for t in sad_sentences[(df_sad["School"] == 1) | (df_sad["is_stressor"] == 1)]]
        tier_1_texts.extend(filter_by_length(sad_school))
        print(f"    Added from SAD: Tier 0 Total={len(tier_0_texts)}, Tier 1 Total={len(tier_1_texts)}")

    # -------------------------------------------------------------------------
    # 4. Harvest from Suicide Watch Dataset (Tier 2 High Crisis)
    # -------------------------------------------------------------------------
    sw_path = "data/raw/suicide_watch/Suicide_Detection.csv"
    if os.path.exists(sw_path):
        print("[*] Processing Suicide Watch (Crisis class)...")
        # Read in chunks for speed
        chunks = []
        for chunk in pd.read_csv(sw_path, usecols=["text", "class"], chunksize=20000):
            suicide_chunk = chunk[chunk["class"] == "suicide"]["text"].dropna()
            cleaned = [clean_text(t) for t in suicide_chunk]
            valid_chunk = filter_by_length(cleaned, min_words=10, max_words=60)
            chunks.extend(valid_chunk)
            if len(chunks) >= 4000:
                break
        tier_2_texts.extend(chunks)
        print(f"    Added from Suicide Watch: Tier 2 Total={len(tier_2_texts)}")

    # -------------------------------------------------------------------------
    # 5. Inject Campus-Specific Ground Truth (Central University of Karnataka)
    # -------------------------------------------------------------------------
    print("[*] Injecting Indian engineering campus domain anchors...")
    from ml.dataset import ACADEMIC_STRESS_SAMPLES
    for text, label in ACADEMIC_STRESS_SAMPLES:
        if label == 0:
            tier_0_texts.append(text)
        elif label == 1:
            tier_1_texts.append(text)
        elif label == 2:
            tier_2_texts.append(text)

    # -------------------------------------------------------------------------
    # 6. Deduplicate & Balance to Target Sample Size
    # -------------------------------------------------------------------------
    tier_0_clean = list(set([t for t in tier_0_texts if len(t.split()) >= 6]))
    tier_1_clean = list(set([t for t in tier_1_texts if len(t.split()) >= 6]))
    tier_2_clean = list(set([t for t in tier_2_texts if len(t.split()) >= 6]))

    random.shuffle(tier_0_clean)
    random.shuffle(tier_1_clean)
    random.shuffle(tier_2_clean)

    n_samples = min(TARGET_SAMPLES_PER_TIER, len(tier_0_clean), len(tier_1_clean), len(tier_2_clean))
    print(f"\n[+] Available unique reflections per tier:")
    print(f"    - Tier 0 (Low Stress)   : {len(tier_0_clean)} available -> Selected: {n_samples}")
    print(f"    - Tier 1 (Medium Stress): {len(tier_1_clean)} available -> Selected: {n_samples}")
    print(f"    - Tier 2 (High Stress)  : {len(tier_2_clean)} available -> Selected: {n_samples}")

    balanced_0 = tier_0_clean[:n_samples]
    balanced_1 = tier_1_clean[:n_samples]
    balanced_2 = tier_2_clean[:n_samples]

    master_records = []
    for t in balanced_0:
        master_records.append({"text": t, "label": 0, "tier_name": "Low"})
    for t in balanced_1:
        master_records.append({"text": t, "label": 1, "tier_name": "Medium"})
    for t in balanced_2:
        master_records.append({"text": t, "label": 2, "tier_name": "High"})

    master_df = pd.DataFrame(master_records)
    master_df = master_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    master_path = "data/academic_stress_master.csv"
    master_df.to_csv(master_path, index=False)
    print(f"\n[+] Master balanced corpus saved to: {master_path} ({len(master_df)} rows)")

    # -------------------------------------------------------------------------
    # 7. Stratified Train / Val / Test Splits (70% Train, 15% Val, 15% Test)
    # -------------------------------------------------------------------------
    train_df, temp_df = train_test_split(master_df, test_size=0.30, stratify=master_df["label"], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED)

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"[+] Splits created successfully:")
    print(f"    - Train Split : data/train.csv ({len(train_df)} rows) [70%]")
    print(f"    - Val Split   : data/val.csv   ({len(val_df)} rows) [15%]")
    print(f"    - Test Split  : data/test.csv  ({len(test_df)} rows) [15%]")
    print("=" * 70)

if __name__ == "__main__":
    main()
