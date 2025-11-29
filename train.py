import json
import os
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize


json_file = 'dataset.json'
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = [item['input'] for item in data]
answers = [item['output'] for item in data]


print("🤖 Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


print("💾 Generating embeddings...")
embeddings = model.encode(questions, show_progress_bar=True, convert_to_numpy=True)
embeddings = normalize(embeddings)  # cosine similarity ke liye normalization

# ----------------------------
# Step 4: Save embeddings, questions, answers
# ----------------------------
if not os.path.exists('ai_models'):
    os.makedirs('ai_models')

np.save('ai_models/embeddings.npy', embeddings)
joblib.dump(questions, 'ai_models/questions.pkl')
joblib.dump(answers, 'ai_models/answers.pkl')
joblib.dump(model, 'ai_models/sbert_model.pkl')

print("✅ Training complete! Models saved in 'ai_models/' folder")
print(f"Total Q/A pairs trained: {len(questions)}")