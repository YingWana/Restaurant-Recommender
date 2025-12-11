#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
04_SBERT.PY - SBERT Embedding Generation with Hierarchical Chunking
================================================================================
This script generates SBERT embeddings for restaurant reviews.

Key Features:
    - Token-based chunking (handles SBERT's 512 token limit)
    - Hierarchical aggregation (review → restaurant)
    - GPU acceleration support

TWO-LEVEL HIERARCHY:
    Level 1 (Inner Loop - Per Review):
        - Short review (<450 tokens): Encode directly
        - Long review (>450 tokens): Chunk → Encode → Mean pool
        - Result: One vector per review

    Level 2 (Outer Loop - Per Restaurant):
        - Collect all review vectors
        - Mean pool into single restaurant vector
        - Result: One vector per restaurant

Input:
    - data/raw/review.parquet

Output:
    - data/processed/restaurant_embeddings.parquet

Usage:
    python src/04_sbert.py
================================================================================
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
import os
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CONFIG, PATHS

# SBERT imports
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch


class SBERTEncoder:
    """
    SBERT encoder with GPU support.

    This class wraps the SentenceTransformer model and its tokenizer,
    providing a clean interface for encoding text.
    """

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = CONFIG.sbert_model_name
            
        # Detect GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing SBERT on {self.device}...")

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)

        # Load tokenizer (for counting tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"✅ SBERT loaded: {model_name}")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts into vectors.

        Args:
            texts: List of strings to encode
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), 384)
        """
        return self.model.encode(texts, batch_size=128, show_progress_bar=show_progress)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string into a vector."""
        return self.model.encode([text])[0]


class LongTextHandler:
    """
    Handles long reviews by chunking based on TOKEN count (not characters).

    Why token-based?
    - SBERT has a 512 TOKEN limit, not character limit
    - A 1000-character review might be 200 tokens (OK) or 400 tokens (OK)
    - A 500-character review with many rare words might be 600 tokens (NOT OK!)
    - Token counting is the ONLY reliable way to check
    """

    def __init__(self, encoder: SBERTEncoder):
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer
        self.max_tokens = CONFIG.max_tokens_per_chunk
        self.min_tokens = CONFIG.min_chunk_tokens

    def chunk_by_tokens(self, text: str) -> List[str]:
        """
        Split text into chunks based on TOKEN count.

        Algorithm:
        1. Tokenize entire text → list of token IDs
        2. Split into chunks of max_tokens each
        3. Decode each chunk back to text

        Args:
            text: Long text to chunk

        Returns:
            List of text chunks, each ≤ max_tokens
        """
        # Tokenize (get token IDs)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # If short enough, return as-is
        if len(token_ids) <= self.max_tokens:
            return [text]

        # Split into chunks
        chunks = []
        for i in range(0, len(token_ids), self.max_tokens):
            chunk_ids = token_ids[i:i + self.max_tokens]

            # Skip tiny chunks (often just punctuation)
            if len(chunk_ids) < self.min_tokens:
                continue

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks if chunks else [text[:1000]]  # Fallback

    def encode(self, text: str, pooling: str = 'mean') -> np.ndarray:
        """
        Encode text with automatic chunking for long reviews.

        Args:
            text: Text to encode (any length)
            pooling: How to combine chunk vectors ('mean' or 'max')

        Returns:
            Single 384-dim vector representing the text
        """
        # Safety check
        if not text or not text.strip():
            return np.zeros(CONFIG.sbert_embedding_dim)

        # Count tokens
        token_count = self.encoder.count_tokens(text)

        # FAST PATH: Short review → encode directly
        if token_count <= self.max_tokens:
            return self.encoder.encode_single(text)

        # SLOW PATH: Long review → chunk, encode, pool
        chunks = self.chunk_by_tokens(text)

        if not chunks:
            return np.zeros(CONFIG.sbert_embedding_dim)

        # Encode all chunks in one batch (efficient!)
        chunk_vectors = self.encoder.encode(chunks, show_progress=False)

        # Pool chunk vectors into single vector
        if pooling == 'mean':
            return np.mean(chunk_vectors, axis=0)
        elif pooling == 'max':
            return np.max(chunk_vectors, axis=0)
        else:
            return np.mean(chunk_vectors, axis=0)


def encode_reviews_hierarchical(
    reviews: List[str],
    encoder: SBERTEncoder,
    handler: LongTextHandler,
    pooling: str = 'mean'
) -> np.ndarray:
    """
    Encode all reviews for ONE restaurant into a single vector.

    TWO-LEVEL HIERARCHY:

    Level 1 (Inner Loop - Per Review):
        - Short review (<450 tokens): Encode directly
        - Long review (>450 tokens): Chunk → Encode → Mean pool
        - Result: One vector per review

    Level 2 (Outer Loop - Per Restaurant):
        - Collect all review vectors
        - Mean pool into single restaurant vector
        - Result: One vector per restaurant

    Args:
        reviews: List of review texts for one restaurant
        encoder: SBERTEncoder instance
        handler: LongTextHandler instance (contains chunking logic)
        pooling: Pooling strategy ('mean' or 'max')

    Returns:
        Single 384-dim vector representing the restaurant
    """
    # ─────────────────────────────────────────────────────────────
    # SAFETY CHECKS
    # ─────────────────────────────────────────────────────────────
    if not reviews:
        return np.zeros(CONFIG.sbert_embedding_dim)

    # Filter empty/too-short reviews
    valid_reviews = [
        str(r).strip()
        for r in reviews
        if r and len(str(r).strip()) > 10
    ]

    if not valid_reviews:
        return np.zeros(CONFIG.sbert_embedding_dim)

    # ─────────────────────────────────────────────────────────────
    # LEVEL 1: INNER LOOP (Per Review) - Batch Processing
    # ─────────────────────────────────────────────────────────────
    all_chunks = []
    review_indices = []  # Maps chunk -> review index (e.g., [0, 0, 1, 2, 2, 2])

    for i, text in enumerate(valid_reviews):
        # Use the handler to just GET chunks (don't encode yet!)
        chunks = handler.chunk_by_tokens(text)
        all_chunks.extend(chunks)
        review_indices.extend([i] * len(chunks))

    if not all_chunks:
        return np.zeros(CONFIG.sbert_embedding_dim)

    # ─────────────────────────────────────────────────────────────
    # STEP 2: MASSIVE ENCODING (GPU Work - The Fast Part)
    # Encode everything in one giant call.
    # ─────────────────────────────────────────────────────────────
    # batch_size=128 is efficient for T4 GPU
    all_vectors = encoder.model.encode(all_chunks, batch_size=128, show_progress_bar=False)

    # ─────────────────────────────────────────────────────────────
    # STEP 3: RE-AGGREGATE
    # Group vectors back into reviews, then into restaurant
    # ─────────────────────────────────────────────────────────────

    # Use a simple dictionary to group vectors by review_index
    # Format: { review_index: [vector1, vector2...] }
    review_vector_map = {}

    for i, review_idx in enumerate(review_indices):
        if review_idx not in review_vector_map:
            review_vector_map[review_idx] = []
        review_vector_map[review_idx].append(all_vectors[i])

    # Now calculate the single vector for each review (Level 1 Aggregation)
    final_review_vectors = []

    for review_idx in review_vector_map:
        chunk_vecs = np.array(review_vector_map[review_idx])

        # Mean pool chunks into one review vector
        if pooling == 'max':
            review_vec = np.max(chunk_vecs, axis=0)
        else:
            review_vec = np.mean(chunk_vecs, axis=0)

        final_review_vectors.append(review_vec)

    # ─────────────────────────────────────────────────────────────
    # LEVEL 2: OUTER LOOP (Restaurant Aggregation)
    # ─────────────────────────────────────────────────────────────
    final_review_vectors = np.array(final_review_vectors)

    if pooling == 'max':
        return np.max(final_review_vectors, axis=0)
    else:
        return np.mean(final_review_vectors, axis=0)


def main():
    """Main SBERT encoding pipeline."""
    print("\n" + "="*60)
    print("SBERT EMBEDDING GENERATION")
    print("="*60)
    
    # Ensure directories exist
    PATHS.ensure_dirs()

    # ─────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1: Loading Review Data")
    print(f"{'='*60}")

    df_raw = pd.read_parquet(PATHS.RAW_REVIEW)
    print(f"  Loaded {len(df_raw):,} reviews")

    # Ensure text is string
    df_raw['text'] = df_raw['text'].fillna("").astype(str)

    # ─────────────────────────────────────────────────────────────
    # 2. GROUP BY RESTAURANT
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2: Grouping Reviews by Restaurant")
    print(f"{'='*60}")

    grouped = df_raw.groupby('business_id')['text'].apply(list).reset_index()
    business_ids = grouped['business_id'].tolist()
    all_reviews_lists = grouped['text'].tolist()

    print(f"  Found {len(business_ids):,} unique restaurants")

    # Quick stats
    review_counts = [len(r) for r in all_reviews_lists]
    print(f"  Reviews per restaurant: min={min(review_counts)}, "
          f"max={max(review_counts)}, mean={np.mean(review_counts):.1f}")

    # ─────────────────────────────────────────────────────────────
    # 3. INITIALIZE ENCODER
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: Initializing SBERT Encoder")
    print(f"{'='*60}")

    encoder = SBERTEncoder()
    handler = LongTextHandler(encoder)

    # ─────────────────────────────────────────────────────────────
    # 4. ANALYZE TOKEN DISTRIBUTION (Optional but informative)
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4: Analyzing Token Distribution (sampling 1000 reviews)")
    print(f"{'='*60}")

    # Sample reviews for analysis
    sample_reviews = df_raw['text'].sample(min(1000, len(df_raw)), random_state=42)
    token_counts = [encoder.count_tokens(str(r)) for r in sample_reviews]

    print(f"  Token counts: min={min(token_counts)}, max={max(token_counts)}, "
          f"mean={np.mean(token_counts):.1f}")

    long_reviews = sum(1 for t in token_counts if t > CONFIG.max_tokens_per_chunk)
    print(f"  Reviews > {CONFIG.max_tokens_per_chunk} tokens: "
          f"{long_reviews} ({100*long_reviews/len(token_counts):.1f}%)")

    # ─────────────────────────────────────────────────────────────
    # 5. ENCODE ALL RESTAURANTS
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 5: Encoding Restaurants (This will take time...)")
    print(f"{'='*60}")

    vectors = []

    for reviews in tqdm(all_reviews_lists, total=len(all_reviews_lists), 
                        desc="Encoding Restaurants", unit="biz"):
        vec = encode_reviews_hierarchical(
            reviews=reviews,
            encoder=encoder,
            handler=handler,
            pooling='mean'
        )
        vectors.append(vec)

    # Convert to numpy
    final_vectors = np.array(vectors)
    print(f"\n  Final vectors shape: {final_vectors.shape}")

    # Sanity check
    assert final_vectors.shape == (len(business_ids), CONFIG.sbert_embedding_dim), \
        "Vector shape mismatch!"

    # ─────────────────────────────────────────────────────────────
    # 6. SAVE RESULTS
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 6: Saving as DataFrame")
    print(f"{'='*60}")

    # Create a DataFrame to lock IDs and Vectors together
    df_output = pd.DataFrame({
        'business_id': business_ids,
        'embedding': list(final_vectors)  # Convert matrix to list of arrays for storage
    })

    # Save to parquet
    df_output.to_parquet(PATHS.RESTAURANT_EMBEDDINGS, engine='pyarrow')
    print(f"  ✅ Saved: {PATHS.RESTAURANT_EMBEDDINGS}")

    # Display first few rows to confirm
    print(f"\n  Preview:\n{df_output.head(3)}")

    print(f"\n{'='*60}")
    print("✅ SBERT ENCODING COMPLETE!")
    print(f"{'='*60}")

    return final_vectors, business_ids


if __name__ == "__main__":
    main()
