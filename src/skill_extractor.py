"""
Skill Extractor — Two-tier skill extraction: Dictionary matching + Semantic matching.
Extracts, categorizes, and ranks skills from resume text.
"""

import os
import re
import json
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SKILLS_DB_JSON, SKILL_EMBEDDINGS_PATH, SKILL_NAMES_PATH


class SkillExtractor:
    """
    Two-tier skill extraction engine:
    1. Dictionary-based exact + alias matching (fast, precise)
    2. Semantic matching via embeddings (catches variations)
    """

    def __init__(self, embedding_engine=None):
        self.embedding_engine = embedding_engine
        self.skills_db = self._load_skills_database()
        self.skill_lookup = self._build_lookup_table()
        self._skill_embeddings = None
        self._skill_names = None

    def _load_skills_database(self) -> Dict:
        """Load the comprehensive skills database."""
        if os.path.exists(SKILLS_DB_JSON):
            with open(SKILLS_DB_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"categories": {}}

    def _build_lookup_table(self) -> Dict[str, Dict]:
        """Build a fast lookup table of skill name/alias → skill info."""
        lookup = {}
        for category, skills in self.skills_db.get("categories", {}).items():
            for skill in skills:
                name = skill["name"].lower()
                lookup[name] = {
                    "canonical_name": skill["name"],
                    "category": category,
                    "related_roles": skill.get("related_roles", []),
                }
                # Add aliases
                for alias in skill.get("aliases", []):
                    lookup[alias.lower()] = {
                        "canonical_name": skill["name"],
                        "category": category,
                        "related_roles": skill.get("related_roles", []),
                    }
        return lookup

    def _load_skill_embeddings(self):
        """Load precomputed skill embeddings for semantic matching."""
        if self._skill_embeddings is None:
            if os.path.exists(SKILL_EMBEDDINGS_PATH) and os.path.exists(SKILL_NAMES_PATH):
                self._skill_embeddings = np.load(SKILL_EMBEDDINGS_PATH)
                self._skill_names = np.load(SKILL_NAMES_PATH, allow_pickle=True)

    def extract_skills_dictionary(self, text: str) -> List[Dict]:
        """
        Tier 1: Dictionary-based skill extraction.
        Matches against the comprehensive skills database including aliases.
        """
        text_lower = text.lower()
        found_skills = {}

        # Sort skills by length (longest first) to match compound skills first
        sorted_skills = sorted(self.skill_lookup.keys(), key=len, reverse=True)

        for skill_key in sorted_skills:
            # Use word boundary matching for precision
            pattern = r'(?<![a-zA-Z])' + re.escape(skill_key) + r'(?![a-zA-Z])'
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                info = self.skill_lookup[skill_key]
                canonical = info["canonical_name"]
                if canonical not in found_skills:
                    found_skills[canonical] = {
                        "name": canonical,
                        "category": info["category"],
                        "frequency": len(matches),
                        "related_roles": info["related_roles"],
                        "match_type": "dictionary",
                        "confidence": 0.95,
                    }
                else:
                    found_skills[canonical]["frequency"] += len(matches)

        return list(found_skills.values())

    def extract_skills_semantic(self, text: str, threshold: float = 0.55) -> List[Dict]:
        """
        Tier 2: Semantic skill extraction using embeddings.
        Catches skills phrased differently than in the dictionary.
        """
        if self.embedding_engine is None:
            return []

        self._load_skill_embeddings()
        if self._skill_embeddings is None or self._skill_names is None:
            return []

        # Extract candidate phrases from text (2-3 word chunks)
        words = text.split()
        candidates = set()
        for i in range(len(words)):
            candidates.add(words[i].lower())
            if i + 1 < len(words):
                candidates.add(f"{words[i]} {words[i+1]}".lower())
            if i + 2 < len(words):
                candidates.add(f"{words[i]} {words[i+1]} {words[i+2]}".lower())

        # Filter out very short or common words
        candidates = [c for c in candidates if len(c) > 2 and not c.isdigit()]

        if not candidates:
            return []

        # Encode candidates
        try:
            candidate_embeddings = self.embedding_engine.encode(list(candidates))
        except Exception:
            return []

        # Compare against skill embeddings
        found_semantic = []
        already_found = set()

        for idx, candidate in enumerate(candidates):
            similarities = self.embedding_engine.compute_similarity_batch(
                candidate_embeddings[idx], self._skill_embeddings
            )
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            if best_score >= threshold:
                skill_name = str(self._skill_names[best_idx])
                if skill_name not in already_found:
                    already_found.add(skill_name)
                    # Find category from database
                    category = "General"
                    for cat, skills in self.skills_db.get("categories", {}).items():
                        for s in skills:
                            if s["name"] == skill_name:
                                category = cat
                                break

                    found_semantic.append({
                        "name": skill_name,
                        "category": category,
                        "frequency": 1,
                        "related_roles": [],
                        "match_type": "semantic",
                        "confidence": round(float(best_score), 3),
                    })

        return found_semantic

    def extract_all_skills(self, text: str) -> List[Dict]:
        """
        Full two-tier skill extraction pipeline.
        Returns combined, deduplicated, ranked skills.
        """
        # Tier 1: Dictionary matching
        dict_skills = self.extract_skills_dictionary(text)
        dict_skill_names = {s["name"] for s in dict_skills}

        # Tier 2: Semantic matching (only for skills not already found)
        semantic_skills = self.extract_skills_semantic(text)
        semantic_skills = [s for s in semantic_skills if s["name"] not in dict_skill_names]

        # Combine and rank
        all_skills = dict_skills + semantic_skills
        all_skills.sort(key=lambda x: (x["confidence"], x["frequency"]), reverse=True)

        return all_skills

    def categorize_skills(self, skills: List[Dict]) -> Dict[str, List[Dict]]:
        """Group skills by category."""
        categories = {}
        for skill in skills:
            cat = skill.get("category", "General")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(skill)
        return categories

    def get_skill_summary(self, skills: List[Dict]) -> Dict:
        """Generate a summary of extracted skills."""
        categorized = self.categorize_skills(skills)
        return {
            "total_skills": len(skills),
            "categories": {cat: len(skills_list) for cat, skills_list in categorized.items()},
            "top_skills": [s["name"] for s in skills[:10]],
            "technical_ratio": sum(
                1 for s in skills
                if s.get("category") in [
                    "Programming Languages", "Machine Learning & AI",
                    "Cloud & DevOps", "Databases", "Web Development",
                    "Data Science", "Frameworks"
                ]
            ) / max(len(skills), 1),
            "avg_confidence": np.mean([s["confidence"] for s in skills]) if skills else 0,
        }
