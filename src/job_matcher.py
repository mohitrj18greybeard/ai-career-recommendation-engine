"""
Job Matcher — Similarity-based job matching using cosine similarity on BERT embeddings.
No classification — pure semantic similarity matching with weighted scoring.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    JOB_DESC_CSV, JOB_EMBEDDINGS_PATH, JOB_IDS_PATH,
    MATCH_WEIGHTS, TOP_N_RECOMMENDATIONS, MIN_MATCH_SCORE
)


class JobMatcher:
    """
    Similarity-based job matching engine.
    Uses cosine similarity between resume and job description embeddings
    with weighted scoring (text similarity + skill overlap + experience).
    """

    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine
        self.job_data = None
        self.job_embeddings = None
        self._load_job_data()

    def _load_job_data(self):
        """Load job descriptions and precomputed embeddings."""
        # Load job descriptions
        if os.path.exists(JOB_DESC_CSV):
            self.job_data = pd.read_csv(JOB_DESC_CSV, encoding='utf-8')
        else:
            self.job_data = pd.DataFrame()

        # Load precomputed embeddings
        if os.path.exists(JOB_EMBEDDINGS_PATH):
            self.job_embeddings = np.load(JOB_EMBEDDINGS_PATH)
        elif not self.job_data.empty:
            # Compute on-the-fly if precomputed not available
            self._compute_job_embeddings()

    def _compute_job_embeddings(self):
        """Compute job description embeddings on-the-fly."""
        if self.job_data is not None and not self.job_data.empty:
            descriptions = self.job_data['description'].fillna('').tolist()
            self.job_embeddings = self.embedding_engine.encode(descriptions, show_progress=False)

    def compute_text_similarity(self, resume_text: str) -> np.ndarray:
        """Compute text-level cosine similarity between resume and all jobs."""
        if self.job_embeddings is None or len(self.job_embeddings) == 0:
            return np.array([])

        resume_embedding = self.embedding_engine.encode(resume_text)
        similarities = self.embedding_engine.compute_similarity_batch(
            resume_embedding[0], self.job_embeddings
        )
        return similarities

    def compute_skill_overlap(self, resume_skills: List[str], job_idx: int) -> float:
        """Compute skill overlap score between resume skills and job requirements."""
        if self.job_data is None or self.job_data.empty:
            return 0.0

        job_row = self.job_data.iloc[job_idx]
        required_skills_str = str(job_row.get('required_skills', ''))
        required_skills = {s.strip().lower() for s in required_skills_str.split(',') if s.strip()}

        if not required_skills:
            return 0.5  # Neutral score if no skills specified

        resume_skills_lower = {s.lower() for s in resume_skills}

        # Direct overlap
        overlap = resume_skills_lower.intersection(required_skills)
        overlap_ratio = len(overlap) / len(required_skills) if required_skills else 0

        # Semantic overlap (check if semantically similar skills exist)
        if self.embedding_engine and overlap_ratio < 0.5:
            unmatched_required = required_skills - overlap
            for req_skill in list(unmatched_required)[:10]:  # Limit for performance
                for res_skill in resume_skills_lower:
                    try:
                        sim = self.embedding_engine.compute_similarity(
                            self.embedding_engine.encode(req_skill)[0],
                            self.embedding_engine.encode(res_skill)[0]
                        )
                        if sim > 0.6:
                            overlap_ratio += (1 / len(required_skills)) * 0.7
                            break
                    except Exception:
                        pass

        return min(overlap_ratio, 1.0)

    def compute_experience_match(self, resume_years: Optional[int], job_idx: int) -> float:
        """Compute experience level match score."""
        if self.job_data is None or self.job_data.empty:
            return 0.5

        job_row = self.job_data.iloc[job_idx]
        job_level = str(job_row.get('experience_level', 'Mid Level')).lower()

        if resume_years is None:
            return 0.5  # Neutral

        level_ranges = {
            'entry level': (0, 2),
            'mid level': (2, 5),
            'senior': (5, 10),
            'lead': (8, 15),
            'director': (10, 25),
        }

        min_yr, max_yr = level_ranges.get(job_level, (2, 5))
        if min_yr <= resume_years <= max_yr:
            return 1.0
        elif resume_years < min_yr:
            return max(0.3, 1.0 - (min_yr - resume_years) * 0.15)
        else:
            return max(0.5, 1.0 - (resume_years - max_yr) * 0.05)

    def match_jobs(
        self,
        resume_text: str,
        resume_skills: List[str],
        experience_years: Optional[int] = None,
        top_n: int = TOP_N_RECOMMENDATIONS,
    ) -> List[Dict]:
        """
        Full job matching pipeline with weighted scoring.

        Returns top-N jobs with scores and match breakdown.
        """
        if self.job_data is None or self.job_data.empty:
            return []

        # 1. Text similarity scores
        text_scores = self.compute_text_similarity(resume_text)
        if len(text_scores) == 0:
            return []

        # 2. Compute weighted scores for each job
        results = []
        for idx in range(len(self.job_data)):
            text_sim = float(text_scores[idx])
            skill_overlap = self.compute_skill_overlap(resume_skills, idx)
            exp_match = self.compute_experience_match(experience_years, idx)

            # Weighted final score
            final_score = (
                MATCH_WEIGHTS["text_similarity"] * text_sim +
                MATCH_WEIGHTS["skill_overlap"] * skill_overlap +
                MATCH_WEIGHTS["experience_match"] * exp_match
            )

            if final_score >= MIN_MATCH_SCORE:
                job_row = self.job_data.iloc[idx]
                results.append({
                    "job_id": int(job_row.get('job_id', idx)),
                    "title": str(job_row.get('title', 'Unknown')),
                    "category": str(job_row.get('category', 'Unknown')),
                    "description": str(job_row.get('description', '')),
                    "required_skills": str(job_row.get('required_skills', '')),
                    "experience_level": str(job_row.get('experience_level', 'Mid Level')),
                    "final_score": round(final_score, 4),
                    "score_breakdown": {
                        "text_similarity": round(text_sim, 4),
                        "skill_overlap": round(skill_overlap, 4),
                        "experience_match": round(exp_match, 4),
                    },
                })

        # Sort by final score descending
        results.sort(key=lambda x: x["final_score"], reverse=True)

        # Return top N unique categories (avoid duplicate roles)
        seen_categories = set()
        unique_results = []
        for r in results:
            if r["category"] not in seen_categories:
                seen_categories.add(r["category"])
                unique_results.append(r)
            if len(unique_results) >= top_n:
                break

        # If not enough unique categories, fill with remaining
        if len(unique_results) < top_n:
            for r in results:
                if r not in unique_results:
                    unique_results.append(r)
                if len(unique_results) >= top_n:
                    break

        return unique_results[:top_n]

    def compare_resume_to_job(self, resume_text: str, job_description: str,
                                resume_skills: List[str]) -> Dict:
        """
        Detailed resume vs job description comparison.
        Returns granular match breakdown.
        """
        # Text similarity
        resume_emb = self.embedding_engine.encode(resume_text)
        job_emb = self.embedding_engine.encode(job_description)
        text_similarity = self.embedding_engine.compute_similarity(resume_emb[0], job_emb[0])

        # Keyword overlap
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        keyword_overlap = len(resume_words & job_words) / max(len(job_words), 1)

        # Skill comparison
        job_skills_raw = set()
        # Try to extract skills from job description too
        for skill in resume_skills:
            if skill.lower() in job_description.lower():
                job_skills_raw.add(skill)

        return {
            "text_similarity": round(float(text_similarity), 4),
            "keyword_overlap": round(keyword_overlap, 4),
            "matching_skills": list(job_skills_raw),
            "overall_match": round(
                0.6 * float(text_similarity) + 0.3 * keyword_overlap + 0.1 * (len(job_skills_raw) / max(len(resume_skills), 1)), 4
            ) if resume_skills else round(0.7 * float(text_similarity) + 0.3 * keyword_overlap, 4),
        }
