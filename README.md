# Infinite Craft Embeddings

A recreation of Neal.fun's [Infinite Craft](https://neal.fun/infinite-craft/) game using semantic embeddings to enable truly infinite concept combinations. Instead of relying on predefined rules or limited databases, this project uses machine learning embeddings to understand semantic relationships between concepts and generate meaningful combinations.

## 🎮 What is Infinite Craft?

Infinite Craft is a creative sandbox game where players combine basic elements (fire, water, earth, air) to create new concepts. For example:
- Fire + Water = Steam
- Horse + Horn = Unicorn
- Human + Magic = Wizard

This project recreates that experience using **semantic embeddings** to understand the meaning and relationships between concepts, enabling potentially infinite combinations that make semantic sense.

## 🧠 How It Works

The system uses **sentence transformer embeddings** to understand the semantic meaning of concepts. When combining two concepts, it:

1. **Loads pre-computed embeddings** for hundreds of concepts
2. **Applies embedding arithmetic** to find semantically similar combinations
3. **Uses multiple strategies**:
   - Simple vector averaging
   - Semantic bridging (finding concepts related to both inputs)
   - Rule-based combinations for known patterns
4. **Ranks results** using cosine similarity and softmax probability distributions

### Key Features

- **Semantic Understanding**: Uses Qwen3-Embedding-0.6B model for high-quality concept representations
- **Multiple Combination Strategies**: Hybrid approach combining arithmetic and rule-based methods
- **Extensible Concept Database**: Easy to add new concepts and expand the vocabulary
- **Probabilistic Results**: Returns multiple possibilities with confidence scores
- **Known Combinations**: Includes curated combinations for common/expected results

### Run the app

```cmd
python app.py
```

A work in progress but... Enjoy!
