"""
Skill Gap Analyzer — Identifies missing skills, severity, and learning paths.
Compares extracted skills against target role requirements.
"""

import os
import json
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SKILLS_DB_JSON, GAP_SEVERITY


class SkillGapAnalyzer:
    """
    Analyzes skill gaps between a candidate's resume and target role requirements.
    Provides severity scoring and learning path recommendations.
    """

    # Learning resources per skill category
    LEARNING_RESOURCES = {
        "Programming Languages": {
            "platform": "Codecademy, LeetCode, HackerRank",
            "time": "2-4 weeks",
            "difficulty": "Medium",
        },
        "Machine Learning & AI": {
            "platform": "Coursera (Andrew Ng), Fast.ai, Kaggle",
            "time": "4-8 weeks",
            "difficulty": "Hard",
        },
        "Cloud & DevOps": {
            "platform": "AWS/Azure/GCP Free Tier, A Cloud Guru",
            "time": "3-6 weeks",
            "difficulty": "Medium",
        },
        "Databases": {
            "platform": "SQLZoo, MongoDB University",
            "time": "2-3 weeks",
            "difficulty": "Medium",
        },
        "Web Development": {
            "platform": "freeCodeCamp, The Odin Project",
            "time": "4-8 weeks",
            "difficulty": "Medium",
        },
        "Data Science": {
            "platform": "DataCamp, Kaggle Learn",
            "time": "3-6 weeks",
            "difficulty": "Medium",
        },
        "Frameworks": {
            "platform": "Official Docs, Udemy",
            "time": "2-4 weeks",
            "difficulty": "Medium",
        },
        "Soft Skills": {
            "platform": "LinkedIn Learning, Toastmasters",
            "time": "Ongoing",
            "difficulty": "Easy",
        },
        "Tools & Platforms": {
            "platform": "Official Documentation, YouTube",
            "time": "1-2 weeks",
            "difficulty": "Easy",
        },
        "Testing": {
            "platform": "Test Automation University, Udemy",
            "time": "2-4 weeks",
            "difficulty": "Medium",
        },
    }

    def __init__(self):
        self.skills_db = self._load_skills_database()

    def _load_skills_database(self) -> Dict:
        """Load the comprehensive skills database."""
        if os.path.exists(SKILLS_DB_JSON):
            with open(SKILLS_DB_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"categories": {}}

    def analyze_gap(
        self,
        resume_skills: List[str],
        target_role: str,
        job_required_skills: Optional[List[str]] = None,
    ) -> Dict:
        """
        Full skill gap analysis.

        Args:
            resume_skills: List of skills extracted from resume.
            target_role: Target job role/category.
            job_required_skills: Optional explicit list of required skills.

        Returns:
            Comprehensive gap analysis with severity and learning paths.
        """
        # Get required skills for target role
        if job_required_skills:
            required = job_required_skills
        else:
            required = self._get_role_requirements(target_role)

        resume_skills_lower = {s.lower() for s in resume_skills}
        required_lower = {s.lower(): s for s in required}

        # Categorize skills
        matched = []
        missing = []

        for req_lower, req_original in required_lower.items():
            if req_lower in resume_skills_lower:
                matched.append(req_original)
            else:
                missing.append(req_original)

        # Assess severity for each missing skill
        gap_details = []
        for skill in missing:
            severity = self._assess_severity(skill, target_role)
            category = self._get_skill_category(skill)
            learning = self.LEARNING_RESOURCES.get(category, {
                "platform": "Online courses, Official documentation",
                "time": "2-4 weeks",
                "difficulty": "Medium",
            })

            gap_details.append({
                "skill": skill,
                "severity": severity["label"],
                "severity_level": severity["level"],
                "severity_color": severity["color"],
                "category": category,
                "learning_path": {
                    "recommended_platform": learning["platform"],
                    "estimated_time": learning["time"],
                    "difficulty": learning["difficulty"],
                },
            })

        # Sort by severity (critical first)
        severity_order = {"critical": 0, "important": 1, "nice_to_have": 2}
        gap_details.sort(key=lambda x: severity_order.get(x["severity_level"], 3))

        # Calculate overall readiness
        total = len(matched) + len(missing)
        readiness_score = (len(matched) / total * 100) if total > 0 else 0

        return {
            "target_role": target_role,
            "total_required": total,
            "matched_count": len(matched),
            "missing_count": len(missing),
            "readiness_score": round(readiness_score, 1),
            "matched_skills": matched,
            "gap_details": gap_details,
            "readiness_level": self._get_readiness_level(readiness_score),
            "summary": self._generate_summary(target_role, readiness_score, len(matched), len(missing)),
        }

    def _get_role_requirements(self, target_role: str) -> List[str]:
        """Get typical required skills for a role from the database."""
        role_lower = target_role.lower()
        required = set()

        for category, skills in self.skills_db.get("categories", {}).items():
            for skill in skills:
                related_roles = [r.lower() for r in skill.get("related_roles", [])]
                if any(role_lower in r or r in role_lower for r in related_roles):
                    required.add(skill["name"])

        # If no matches found, return common skills
        if not required:
            required = {"Python", "SQL", "Communication", "Problem Solving", "Teamwork"}

        return list(required)

    def _assess_severity(self, skill: str, target_role: str) -> Dict:
        """Assess how critical a missing skill is for the target role."""
        # Core skills per role type (simplified mapping)
        critical_skills = {
            "data science": ["python", "machine learning", "statistics", "sql", "pandas"],
            "web": ["html", "css", "javascript", "react", "node.js"],
            "devops": ["docker", "kubernetes", "ci/cd", "aws", "linux"],
            "java developer": ["java", "spring", "sql", "rest api", "git"],
            "python developer": ["python", "django", "flask", "sql", "git"],
            "business analyst": ["sql", "excel", "tableau", "communication", "requirements"],
        }

        skill_lower = skill.lower()
        role_lower = target_role.lower()

        # Check if skill is critical for this role
        for role_key, core_skills in critical_skills.items():
            if role_key in role_lower:
                if any(skill_lower in cs or cs in skill_lower for cs in core_skills):
                    return {"label": "🔴 Critical", "level": "critical", "color": "#FF4444"}

        # Check category importance
        category = self._get_skill_category(skill)
        if category in ["Programming Languages", "Machine Learning & AI", "Databases"]:
            return {"label": "🟠 Important", "level": "important", "color": "#FF8C00"}

        return {"label": "🟢 Nice to Have", "level": "nice_to_have", "color": "#44BB44"}

    def _get_skill_category(self, skill_name: str) -> str:
        """Find the category of a skill."""
        for category, skills in self.skills_db.get("categories", {}).items():
            for s in skills:
                if s["name"].lower() == skill_name.lower():
                    return category
        return "General"

    def _get_readiness_level(self, score: float) -> Dict:
        """Get readiness level based on score."""
        if score >= 80:
            return {"level": "Ready", "emoji": "🟢", "message": "You're well-prepared for this role!"}
        elif score >= 60:
            return {"level": "Almost Ready", "emoji": "🟡", "message": "Minor skill gaps to address."}
        elif score >= 40:
            return {"level": "Developing", "emoji": "🟠", "message": "Focused learning plan recommended."}
        else:
            return {"level": "Building", "emoji": "🔴", "message": "Significant upskilling needed."}

    def _generate_summary(self, role: str, score: float, matched: int, missing: int) -> str:
        """Generate a human-readable gap analysis summary."""
        if score >= 80:
            return (
                f"Excellent! You have {matched} out of {matched + missing} skills required for "
                f"{role}. You're a strong candidate with minimal gaps to address."
            )
        elif score >= 60:
            return (
                f"Good progress! You have {matched} out of {matched + missing} skills for {role}. "
                f"Focus on {missing} missing skill(s) to strengthen your candidacy."
            )
        elif score >= 40:
            return (
                f"You're building towards {role} with {matched} of {matched + missing} skills. "
                f"A structured learning plan for the {missing} missing skills will boost your profile."
            )
        else:
            return (
                f"You have {matched} of {matched + missing} skills for {role}. "
                f"Consider a dedicated upskilling program targeting the {missing} key gaps."
            )
