# evaluate_sbert_model.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# CONFIG
# ----------------------------
DATASET_FILE = "dataset.json"
MODEL_DIR = "fine_tuned_sbert"

print("📌 Loading fine-tuned SBERT model...")
model = SentenceTransformer(MODEL_DIR)

# ----------------------------
# Load + Clean Dataset
# ----------------------------
print("📂 Loading dataset...")
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

clean_data = []
for item in raw_data:
    q = item.get("input")
    a = item.get("output")

    if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
        clean_data.append(item)

print(f"🔥 Clean data size: {len(clean_data)} (removed {len(raw_data)-len(clean_data)})")

questions = [d["input"] for d in clean_data]
answers = [d["output"] for d in clean_data]

# ----------------------------
# Embed Questions & Answers
# ----------------------------
print("⚙️ Generating embeddings...")
q_emb = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
a_emb = model.encode(answers, convert_to_numpy=True, show_progress_bar=True)

# ----------------------------
# Metric 1 — Top-1 Retrieval Accuracy
# ----------------------------
print("\n🎯 Calculating Top-1 Accuracy...")

correct_top1 = 0

for i, q in enumerate(q_emb):
    sims = cosine_similarity([q], a_emb)[0]
    top1 = np.argmax(sims)

    if top1 == i:
        correct_top1 += 1

top1_accuracy = correct_top1 / len(q_emb)

# ----------------------------
# Metric 2 — Recall@3
# ----------------------------
print("🔍 Calculating Recall@3...")

correct_top3 = 0

for i, q in enumerate(q_emb):
    sims = cosine_similarity([q], a_emb)[0]
    top3 = sims.argsort()[-3:][::-1]

    if i in top3:
        correct_top3 += 1

recall3 = correct_top3 / len(q_emb)

# ----------------------------
# Metric 3 — Cosine Similarity Stats
# ----------------------------
print("📐 Calculating cosine similarity distances...")

same_sims = []
diff_sims = []

for i, q in enumerate(q_emb):
    sim_same = cosine_similarity([q], [a_emb[i]])[0][0]
    same_sims.append(sim_same)

    # random wrong pair
    j = np.random.randint(0, len(a_emb))
    sim_diff = cosine_similarity([q], [a_emb[j]])[0][0]
    diff_sims.append(sim_diff)

avg_same = np.mean(same_sims)
avg_diff = np.mean(diff_sims)
gap = avg_same - avg_diff

# ----------------------------
# FINAL REPORT
# ----------------------------
print("\n==============================")
print("📊   SBERT MODEL EVALUATION")
print("==============================")

print(f"🎯 Top-1 Accuracy:        {top1_accuracy*100:.2f}%")
print(f"🔍 Recall@3:             {recall3*100:.2f}%")
print(f"📈 Avg SAME-pair sim:    {avg_same:.4f}")
print(f"📉 Avg DIFF-pair sim:    {avg_diff:.4f}")
print(f"⚡ Similarity GAP:       {gap:.4f}")

print("==============================")
print("✨ Evaluation Complete!")
print("==============================")