# Text-Similarity-Checker-using-NLP-
A simple yet effective Text Similarity Checker using Natural Language Processing (NLP) with Python and popular libraries like scikit-learn and NLTK. This version uses TF-IDF (Term Frequency–Inverse Document Frequency) and Cosine Similarity, which are widely used in academic and real-world NLP applications.
#code
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# STEP 1: Load Pre-trained Model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast BERT variant

# STEP 2: Preprocessing Function
def preprocess(text):
    """
    Clean and normalize text for better embedding performance.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# STEP 3: Define or Input Texts
# You can also take input dynamically using input() or from a CSV
text_samples = [
    "The food was absolutely wonderful, from preparation to presentation.",
    "Dinner was delicious, especially the main course and dessert.",
    "The restaurant ambiance was dull and the food was average.",
    "He plays football every Sunday with his friends.",
    "She goes running every morning as part of her routine.",
    "Paris is known for its iconic Eiffel Tower and amazing cuisine.",
    "New York is a bustling city with skyscrapers and nightlife.",
]
# Optional: Preprocess texts
print("Preprocessing texts...")
clean_texts = [preprocess(text) for text in text_samples]

# STEP 4: Encode with BERT
print("Generating embeddings...")
embeddings = model.encode(clean_texts, convert_to_tensor=True)

# STEP 5: Compute Similarities
print("Calculating similarity scores...")
similarity_matrix = util.cos_sim(embeddings, embeddings)

# Convert to NumPy array
similarity_matrix_np = similarity_matrix.cpu().numpy()

# STEP 6: Display Pairwise Scores
def print_similarities(texts, sim_matrix, threshold=0.7):
    print("\n--- Text Similarity Report ---")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = sim_matrix[i][j]
            verdict = "HIGH" if score > threshold else "LOW"
            print(f"\n[{verdict} SIMILARITY - Score: {score:.4f}]")
            print(f"Text A: {texts[i]}")
            print(f"Text B: {texts[j]}")

print_similarities(text_samples, similarity_matrix_np)

# STEP 7: Visualize (Optional)
def show_heatmap(matrix, labels):
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("Text Similarity Heatmap (Cosine Scores)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Uncomment to display similarity heatmap
# show_heatmap(similarity_matrix_np, [f"Text {i+1}" for i in range(len(text_samples))])
