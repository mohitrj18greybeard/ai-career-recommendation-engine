"""
Recommender — Explainable job recommendations with "Why this job?" analysis.
Wraps the job matcher with transparency and interpretability layer.
"""

import os
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Recommender:
    """
    Explainable job recommendation engine.
    Provides transparent "Why this job?" analysis for each recommendation.
    """

    def __init__(self, job_matcher, skill_extractor):
        self.job_matcher = job_matcher
        self.skill_extractor = skill_extractor

    def get_recommendations(
        self,
        resume_text: str,
        resume_skills: List[str],
        experience_years: Optional[int] = None,
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Get explainable job recommendations.

        Returns recommendations with full explainability:
        - Match score breakdown
        - Matching skills list
        - Missing skills list
        - Why this job explanation
        - Confidence level
        """
        # Get base matches from job matcher
        matches = self.job_matcher.match_jobs(
            resume_text=resume_text,
            resume_skills=resume_skills,
            experience_years=experience_years,
            top_n=top_n,
        )

        # Enrich with explainability
        recommendations = []
        for match in matches:
            recommendation = self._enrich_with_explanation(
                match=match,
                resume_skills=resume_skills,
                experience_years=experience_years,
            )
            recommendations.append(recommendation)

        return recommendations

    def _enrich_with_explanation(
        self, match: Dict, resume_skills: List[str], experience_years: Optional[int]
    ) -> Dict:
        """Add explainability layer to a job match."""
        # Parse required skills
        required_skills_str = match.get("required_skills", "")
        required_skills = [s.strip() for s in required_skills_str.split(',') if s.strip()]
        required_skills_lower = {s.lower() for s in required_skills}
        resume_skills_lower = {s.lower() for s in resume_skills}

        # Matching skills
        matching_skills = []
        for req in required_skills:
            if req.lower() in resume_skills_lower:
                matching_skills.append(req)

        # Missing skills
        matching_lower = {s.lower() for s in matching_skills}
        missing_skills = [s for s in required_skills if s.lower() not in matching_lower]

        # Skill match percentage
        skill_match_pct = (len(matching_skills) / len(required_skills) * 100) if required_skills else 0

        # Generate explanation
        explanation = self._generate_explanation(
            match=match,
            matching_skills=matching_skills,
            missing_skills=missing_skills,
            skill_match_pct=skill_match_pct,
            experience_years=experience_years,
        )

        # Confidence level
        score = match["final_score"]
        if score >= 0.7:
            confidence = "🟢 High"
            confidence_text = "Strong alignment with this role"
        elif score >= 0.45:
            confidence = "🟡 Medium"
            confidence_text = "Good potential match with some gaps"
        else:
            confidence = "🟠 Low"
            confidence_text = "Partial match — upskilling recommended"

        return {
            **match,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "skill_match_pct": round(skill_match_pct, 1),
            "explanation": explanation,
            "confidence": confidence,
            "confidence_text": confidence_text,
            "improvement_potential": self._assess_improvement_potential(
                matching_skills, missing_skills
            ),
        }

    def _generate_explanation(
        self,
        match: Dict,
        matching_skills: List[str],
        missing_skills: List[str],
        skill_match_pct: float,
        experience_years: Optional[int],
    ) -> List[str]:
        """Generate human-readable explanation for why this job is recommended."""
        reasons = []
        breakdown = match.get("score_breakdown", {})

        # Text similarity reason
        text_sim = breakdown.get("text_similarity", 0)
        if text_sim >= 0.6:
            reasons.append(
                f"📄 Your resume content strongly aligns with this role "
                f"(semantic similarity: {text_sim:.0%})"
            )
        elif text_sim >= 0.35:
            reasons.append(
                f"📄 Your resume has good relevance to this role "
                f"(semantic similarity: {text_sim:.0%})"
            )
        else:
            reasons.append(
                f"📄 Your resume shows some relevance to this role "
                f"(semantic similarity: {text_sim:.0%})"
            )

        # Skill match reason
        if matching_skills:
            top_matches = matching_skills[:5]
            reasons.append(
                f"🎯 You already have {len(matching_skills)} of {len(matching_skills) + len(missing_skills)} "
                f"required skills ({skill_match_pct:.0f}%): "
                f"{', '.join(top_matches)}"
            )

        # Missing skills (opportunity)
        if missing_skills and len(missing_skills) <= 5:
            reasons.append(
                f"📈 Only {len(missing_skills)} skill(s) to bridge: "
                f"{', '.join(missing_skills[:5])}"
            )
        elif missing_skills:
            reasons.append(
                f"📈 Key skills to develop: {', '.join(missing_skills[:5])} "
                f"(+{len(missing_skills) - 5} more)"
            )

        # Experience reason
        exp_match = breakdown.get("experience_match", 0)
        if experience_years and exp_match >= 0.8:
            reasons.append(
                f"⏱️ Your experience level ({experience_years} years) "
                f"matches this role well"
            )

        return reasons

    def _assess_improvement_potential(
        self, matching_skills: List[str], missing_skills: List[str]
    ) -> str:
        """Assess how easy it would be to bridge the gap."""
        total = len(matching_skills) + len(missing_skills)
        if total == 0:
            return "Unable to assess"

        match_ratio = len(matching_skills) / total

        if match_ratio >= 0.8:
            return "Excellent fit — minimal upskilling needed"
        elif match_ratio >= 0.6:
            return "Good fit — few skills to develop"
        elif match_ratio >= 0.4:
            return "Moderate fit — focused learning plan recommended"
        else:
            return "Growth opportunity — structured upskilling path advised"
