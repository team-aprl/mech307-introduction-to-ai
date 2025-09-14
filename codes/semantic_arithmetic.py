import torch
from sentence_transformers import SentenceTransformer, util

# 1. Load a pre-trained SentenceTransformer model (small & fast MiniLM version)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Function to extract embedding for a given word
def get_embedding(word: str) -> torch.Tensor:
    emb = model.encode(word, convert_to_tensor=True, normalize_embeddings=True)
    return emb

# 3. Perform vector arithmetic: king - man + woman
king = get_embedding("king")
man = get_embedding("man")
woman = get_embedding("woman")

query = king - man + woman
query = query / query.norm()  # normalize the result vector

# 4. Define candidate words to compare against the query vector
candidates = ["queen", "monarch", "princess", "royal", "empress",
              "dog", "car", "computer", "dgist", "aprl"]

# Compute embeddings for all candidates
candidate_embeddings = model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)

# Compute cosine similarity between query vector and candidate embeddings
scores = util.cos_sim(query, candidate_embeddings).squeeze()

# 5. Print results in descending order of similarity
print("king - man + woman ≈ ?")
for idx in scores.argsort(descending=True):
    print(f" {candidates[idx]}: (similarity: {scores[idx]:.4f})")

"""
# Example output:

    king - man + woman ≈ ?
    queen: (similarity: 0.5795)
    monarch: (similarity: 0.5469)
    princess: (similarity: 0.4418)
    empress: (similarity: 0.4368)
    royal: (similarity: 0.4295)
    computer: (similarity: 0.2987)
    dog: (similarity: 0.2843)
    car: (similarity: 0.2641)
    dgist: (similarity: 0.1999)
    aprl: (similarity: 0.0625)
"""