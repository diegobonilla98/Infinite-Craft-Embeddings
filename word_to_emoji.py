from emoji_data_python import emoji_data
from sentence_transformers import SentenceTransformer, util
import torch

# Load a pre-trained sentence-transformer model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="D:\\hf")
print("...done.")

# Prepare data
emojis = []
emoji_texts = []
for e in emoji_data:
    # Compose searchable text for each emoji
    info = [e.name, *(e.short_names or []), *(getattr(e, "keywords", []) or [])]
    fulltext = " ".join(filter(None, info)).lower()
    emojis.append({"char": e.char, "name": e.name, "fulltext": fulltext})
    emoji_texts.append(fulltext)

# Pre-compute all emoji embeddings at once for performance
print("Pre-computing emoji embeddings...")
emoji_embeddings = model.encode(emoji_texts, convert_to_tensor=True)
for i, e in enumerate(emojis):
    e["embedding"] = emoji_embeddings[i]
print("...done.")

def best_emoji(query):
    q = query.strip().lower()

    # Only perform embedding similarity search
    query_embedding = model.encode(q, convert_to_tensor=True)
    
    # Calculate cosine similarities
    cos_scores = util.cos_sim(query_embedding, emoji_embeddings)[0]
    
    # Find the best-scoring emoji
    best_score_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_score_idx].item()
    best_emoji = emojis[best_score_idx]

    # Debug: Show top 5 matches for problematic cases
    if q in ["fire", "steam", "wet", "dry"] or best_score < 0.5:
        top_scores, top_indices = torch.topk(cos_scores, 5)
        print(f"DEBUG - Top 5 matches for '{q}':")
        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
            emoji_match = emojis[idx.item()]
            print(f"  {i+1}. {emoji_match['char']} ({score:.3f}) - {emoji_match['name']}")

    return best_emoji["char"], best_emoji["name"], f"semantic ({best_score:.2f})"


if __name__ == "__main__":
    for word in ["moon", "steam", "mother", "blue", "sword", "parent", "sad", "internet", "money", "danger", "rain", "fire"]:
        emoji, name, how = best_emoji(word)
        print(f"{word:10} → {emoji}  ({name})   [{how}]")
