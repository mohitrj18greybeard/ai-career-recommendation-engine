"""
Build Embeddings Script — Precompute BERT embeddings for job descriptions and skills.
Saves as .npy files for fast loading at runtime.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    JOB_DESC_CSV, SKILLS_DB_JSON, MODELS_DIR,
    JOB_EMBEDDINGS_PATH, JOB_IDS_PATH,
    SKILL_EMBEDDINGS_PATH, SKILL_NAMES_PATH
)
from src.embedding_engine import EmbeddingEngine


def build_job_embeddings(engine: EmbeddingEngine):
    """Precompute embeddings for all job descriptions."""
    print("📋 Computing job description embeddings...")

    if not os.path.exists(JOB_DESC_CSV):
        print("   ⚠️ Job descriptions CSV not found. Run prepare_data.py first.")
        return

    df = pd.read_csv(JOB_DESC_CSV, encoding='utf-8')
    descriptions = df['description'].fillna('').tolist()

    print(f"   Encoding {len(descriptions)} job descriptions...")
    embeddings = engine.encode(descriptions, show_progress=True)

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(JOB_EMBEDDINGS_PATH, embeddings)
    np.save(JOB_IDS_PATH, df['job_id'].values)

    print(f"   ✅ Job embeddings saved: {embeddings.shape}")
    print(f"      Shape: {embeddings.shape[0]} jobs × {embeddings.shape[1]} dimensions")


def build_skill_embeddings(engine: EmbeddingEngine):
    """Precompute embeddings for all skills in the database."""
    print("🔧 Computing skill embeddings...")

    if not os.path.exists(SKILLS_DB_JSON):
        print("   ⚠️ Skills database not found.")
        return

    with open(SKILLS_DB_JSON, 'r', encoding='utf-8') as f:
        skills_db = json.load(f)

    # Collect all skill names
    skill_names = []
    for category, skills in skills_db.get("categories", {}).items():
        for skill in skills:
            skill_names.append(skill["name"])

    skill_names = list(set(skill_names))  # Deduplicate
    print(f"   Encoding {len(skill_names)} unique skills...")

    embeddings = engine.encode(skill_names, show_progress=True)

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(SKILL_EMBEDDINGS_PATH, embeddings)
    np.save(SKILL_NAMES_PATH, np.array(skill_names, dtype=object))

    print(f"   ✅ Skill embeddings saved: {embeddings.shape}")


def main():
    """Run complete embedding precomputation pipeline."""
    print("=" * 60)
    print("  AI Resume Analyzer — Embedding Builder")
    print("=" * 60)
    print()

    # Initialize embedding engine
    print("🚀 Loading Sentence Transformer model...")
    engine = EmbeddingEngine(use_transformers=True)

    # Test encoding
    test_emb = engine.encode("test sentence")
    print(f"   Model loaded. Embedding dimension: {test_emb.shape[1]}")
    print()

    # Build embeddings
    build_job_embeddings(engine)
    print()

    build_skill_embeddings(engine)
    print()

    print("=" * 60)
    print("  ✅ All embeddings precomputed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
