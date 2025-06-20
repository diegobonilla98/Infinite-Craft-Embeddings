import numpy as np
import h5py
from scipy.spatial.distance import cosine
from scipy.special import softmax
from typing import List, Tuple, Optional


class InfiniteCraftArithmetic:
    def __init__(self, embeddings_file: str = "concept_embeddings_clean.h5"):
        self.embeddings_file = embeddings_file
        
        self.concepts = None
        self.embeddings = None
        self.concept_to_idx = {}
        
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings and concepts from HDF5 file."""
        try:
            with h5py.File(self.embeddings_file, "r") as f:
                self.embeddings = f["embeddings"][:]
                # Normalize embeddings to unit vectors for spherical arithmetic
                self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                self.concepts = [c.decode('utf-8') for c in f["concepts"][:]]
                print(f"Loaded and normalized {len(self.concepts)} concepts.")
        except Exception as e:
            raise FileNotFoundError(f"Could not load embeddings file: {e}")
        self.concept_to_idx = {c.lower(): i for i, c in enumerate(self.concepts)}
    
    def _get_embedding(self, concept: str) -> Optional[np.ndarray]:
        """Get embedding for a concept."""
        idx = self.concept_to_idx.get(concept.lower())
        return self.embeddings[idx] if idx is not None else None
    
    def _find_closest_concepts_raw_score(self, target_embedding: np.ndarray, k: int = 5, 
                                         exclude_concepts: List[str] = None) -> List[Tuple[str, float]]:
        """Find k closest concepts and return them with raw cosine similarity scores."""
        if exclude_concepts is None:
            exclude_concepts = []
        exclude_indices = {self.concept_to_idx.get(c.lower()) for c in exclude_concepts if self.concept_to_idx.get(c.lower()) is not None}

        # Cosine similarity is the dot product of normalized vectors
        similarities = self.embeddings @ target_embedding
        
        # Create pairs of (index, similarity)
        sim_pairs = list(enumerate(similarities))
        
        # Sort by similarity
        sim_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_k = []
        for i, sim in sim_pairs:
            if len(top_k) >= k:
                break
            if i not in exclude_indices:
                top_k.append((self.concepts[i], sim))
        
        return top_k

    def _slerp(self, v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two vectors."""
        dot = np.dot(v0, v1)
        # Clamp dot product to handle floating point inaccuracies
        dot = np.clip(dot, -1.0, 1.0)
        
        # If the inputs are too close, linearly interpolate and renormalize.
        if dot > 0.9995:
            result = v0 + t * (v1 - v0)
            return result / np.linalg.norm(result)
            
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        
        theta = t * theta_0
        s0 = np.sin(theta_0 * (1.0 - t)) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        
        return (s0 * v0) + (s1 * v1)

    def _semantic_coherence_score(self, emb1: np.ndarray, emb2: np.ndarray, result_emb: np.ndarray) -> float:
        """Calculate how semantically coherent a combination result is."""
        # Cosine distance
        dist1 = cosine(result_emb, emb1)
        dist2 = cosine(result_emb, emb2)
        input_dist = cosine(emb1, emb2)

        if input_dist < 1e-6: # Avoid division by zero if inputs are identical
            return 1.0 - min(dist1, dist2)

        # Coherence is higher when the result is "between" the inputs.
        # We check if the result lies within the hyperspherical triangle formed by the inputs.
        max_dist_from_input = max(dist1, dist2)
        
        # The score is high if the result is closer to the inputs than they are to each other.
        coherence = max(0, 1 - (max_dist_from_input / (input_dist + 1e-6)))
        return coherence

    def combine_enhanced(self, concept1: str, concept2: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Enhanced zero-shot combination using only advanced embedding similarities.
        """
        emb1 = self._get_embedding(concept1)
        emb2 = self._get_embedding(concept2)
        
        if emb1 is None:
            raise ValueError(f"Concept '{concept1}' not found.")
        if emb2 is None:
            raise ValueError(f"Concept '{concept2}' not found.")

        # --- 1. Candidate Generation ---
        # Generate multiple fusion embeddings using different strategies
        fusion_embeddings = {
            'slerp_0.5': self._slerp(emb1, emb2, 0.5),
            'slerp_0.3': self._slerp(emb1, emb2, 0.3), # Biased towards concept1
            'slerp_0.7': self._slerp(emb1, emb2, 0.7), # Biased towards concept2
            'addition': (emb1 + emb2) / np.linalg.norm(emb1 + emb2),
        }

        # Gather a diverse pool of candidates from all strategies
        candidate_pool = {}
        for f_emb in fusion_embeddings.values():
            # Get more candidates than needed initially
            candidates = self._find_closest_concepts_raw_score(f_emb, k=20, exclude_concepts=[concept1, concept2])
            for concept, sim_score in candidates:
                # Store the highest similarity score for each unique candidate
                candidate_pool[concept] = max(candidate_pool.get(concept, 0), sim_score)

        # --- 2. Reranking using only embedding similarities ---
        reranked_results = []
        for concept, initial_score in candidate_pool.items():
            cand_emb = self._get_embedding(concept)
            if cand_emb is None:
                continue

            sim_to_1 = np.dot(cand_emb, emb1)
            sim_to_2 = np.dot(cand_emb, emb2)

            # Remove exact matches (skip if candidate is nearly identical to either parent)
            if sim_to_1 > 0.999 or sim_to_2 > 0.999:
                continue

            # Only use advanced embedding similarities (no heuristics)
            # Score: mean similarity to both parents
            mean_sim = (sim_to_1 + sim_to_2) / 2.0

            reranked_results.append((concept, mean_sim))

        # --- 3. Final Selection ---
        if not reranked_results:
            return [("unknown", 1.0)]

        # Sort by mean similarity
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        final_results = reranked_results[:k]
        
        # Normalize scores to probabilities for a clean output
        scores = np.array([score for _, score in final_results])
        probabilities = softmax(scores * 10) # Sharpen distribution for confidence
        return [(concept, prob) for (concept, _), prob in zip(final_results, probabilities)]


if __name__ == "__main__":
    craft = InfiniteCraftArithmetic()

    pairs = [
        ("fire", "water"),
        ("earth", "wind"), 
        ("horse", "horn"),
        ("human", "magic"),
        ("tree", "fire"),
        ("metal", "fire"),
        ("stone", "magic"),
        # Test some unknown combinations
        ("cloud", "mountain"),
        ("robot", "dog"), 
        ("music", "light"),
        ("time", "space"),
        ("book", "magic"),
        ("sword", "game"),
        ("magic", "gun"),
        ("metal", "water"),
        ("fire", "gun")
    ]

    print("\n--- Testing Enhanced Zero-Shot Combinations ---")
    for a, b in pairs:
        print(f"\n--- Combining '{a}' + '{b}' ---")
        try:
            results = craft.combine_enhanced(a, b, k=3)
            if not results or results[0][0] == "unknown":
                print("  No suitable combination found.")
                continue
            for i, (concept, prob) in enumerate(results, 1):
                print(f"  {i}. {concept} (confidence: {prob:.3f})")
        except ValueError as e:
            print(f"  Error: {e}")
