"""
Resume Improver — AI-based resume improvement suggestions.
Analyzes structure, missing keywords, action verbs, and quantification.
"""

import re
from typing import Dict, List


class ResumeImprover:
    """
    AI-powered resume improvement engine.
    Provides actionable suggestions for structure, keywords, action verbs,
    and quantification to strengthen resume quality.
    """

    # Strong action verbs for resumes
    STRONG_VERBS = [
        "achieved", "accelerated", "architected", "automated", "built",
        "championed", "consolidated", "created", "delivered", "designed",
        "developed", "drove", "eliminated", "engineered", "established",
        "executed", "expanded", "generated", "implemented", "improved",
        "increased", "influenced", "initiated", "innovated", "integrated",
        "launched", "led", "managed", "mentored", "modernized",
        "negotiated", "optimized", "orchestrated", "overhauled", "pioneered",
        "reduced", "refactored", "resolved", "revamped", "scaled",
        "simplified", "spearheaded", "streamlined", "supervised",
        "transformed", "upgraded",
    ]

    # Weak verbs to replace
    WEAK_VERBS = {
        "did": "executed",
        "made": "developed",
        "helped": "facilitated",
        "worked on": "contributed to",
        "was responsible for": "managed",
        "used": "leveraged",
        "got": "achieved",
        "tried": "pursued",
        "handled": "managed",
        "assisted": "supported",
        "participated": "contributed",
    }

    # Expected resume sections
    EXPECTED_SECTIONS = [
        "summary", "experience", "education", "skills", "projects"
    ]

    def analyze_resume(
        self,
        resume_text: str,
        sections: Dict[str, str],
        extracted_skills: List[str],
        target_role: str = "",
        job_description: str = "",
    ) -> Dict:
        """
        Full resume improvement analysis.

        Returns categorized suggestions with severity.
        """
        suggestions = []

        # 1. Structure analysis
        suggestions.extend(self._analyze_structure(sections))

        # 2. Missing keywords for target role
        if job_description:
            suggestions.extend(self._analyze_keywords(resume_text, job_description))

        # 3. Action verbs analysis
        suggestions.extend(self._analyze_action_verbs(resume_text))

        # 4. Quantification analysis
        suggestions.extend(self._analyze_quantification(resume_text))

        # 5. Length and formatting
        suggestions.extend(self._analyze_formatting(resume_text))

        # 6. Technical depth
        suggestions.extend(self._analyze_technical_depth(extracted_skills, target_role))

        # Sort by impact
        impact_order = {"High": 0, "Medium": 1, "Low": 2}
        suggestions.sort(key=lambda x: impact_order.get(x.get("impact", "Low"), 3))

        # Calculate overall score
        total_issues = len(suggestions)
        high_issues = sum(1 for s in suggestions if s["impact"] == "High")
        medium_issues = sum(1 for s in suggestions if s["impact"] == "Medium")

        score = max(0, 100 - (high_issues * 15) - (medium_issues * 8) - ((total_issues - high_issues - medium_issues) * 3))

        return {
            "overall_score": min(score, 100),
            "total_suggestions": total_issues,
            "high_impact": high_issues,
            "medium_impact": medium_issues,
            "low_impact": total_issues - high_issues - medium_issues,
            "suggestions": suggestions,
            "grade": self._get_grade(score),
        }

    def _analyze_structure(self, sections: Dict[str, str]) -> List[Dict]:
        """Check for missing or weak resume sections."""
        suggestions = []
        present_sections = {s.lower() for s in sections.keys()}

        for expected in self.EXPECTED_SECTIONS:
            if expected not in present_sections:
                suggestions.append({
                    "category": "📋 Structure",
                    "issue": f"Missing '{expected.title()}' section",
                    "suggestion": f"Add a dedicated '{expected.title()}' section to your resume. "
                                  f"This is a standard section that recruiters look for.",
                    "impact": "High",
                    "icon": "🔴",
                })

        # Check section lengths
        for section, content in sections.items():
            if section.lower() == "experience" and len(content.split()) < 50:
                suggestions.append({
                    "category": "📋 Structure",
                    "issue": "Experience section is too brief",
                    "suggestion": "Expand your experience section with more details about your roles, "
                                  "responsibilities, and achievements. Use bullet points.",
                    "impact": "High",
                    "icon": "🔴",
                })
            elif section.lower() == "skills" and len(content.split()) < 10:
                suggestions.append({
                    "category": "📋 Structure",
                    "issue": "Skills section is too brief",
                    "suggestion": "List more relevant technical and soft skills. "
                                  "Group them by category (Programming, Tools, Frameworks).",
                    "impact": "Medium",
                    "icon": "🟡",
                })

        return suggestions

    def _analyze_keywords(self, resume_text: str, job_description: str) -> List[Dict]:
        """Identify important keywords from job description missing in resume."""
        suggestions = []

        # Extract important words from JD
        jd_words = set(job_description.lower().split())
        resume_words = set(resume_text.lower().split())

        # Common noise words to ignore
        noise = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "and", "or", "but", "if", "in", "on", "at", "to", "for",
                 "of", "with", "by", "from", "as", "into", "through",
                 "during", "before", "after", "above", "below", "between",
                 "out", "off", "over", "under", "again", "further", "then",
                 "once", "here", "there", "when", "where", "why", "how",
                 "all", "each", "every", "both", "few", "more", "most",
                 "other", "some", "such", "no", "not", "only", "own",
                 "same", "so", "than", "too", "very", "we", "you", "your",
                 "our", "their", "this", "that", "these", "those", "what",
                 "which", "who", "whom"}

        # Find missing keywords
        important_missing = []
        for word in jd_words:
            if (word not in resume_words and
                word not in noise and
                len(word) > 3 and
                word.isalpha()):
                important_missing.append(word)

        if important_missing:
            top_missing = important_missing[:10]
            suggestions.append({
                "category": "🔑 Keywords",
                "issue": f"{len(important_missing)} keywords from the job description are missing",
                "suggestion": f"Consider incorporating these keywords naturally into your resume: "
                              f"**{', '.join(top_missing)}**. ATS systems scan for keyword matches.",
                "impact": "High",
                "icon": "🔴",
            })

        return suggestions

    def _analyze_action_verbs(self, resume_text: str) -> List[Dict]:
        """Check for strong action verbs and flag weak ones."""
        suggestions = []
        text_lower = resume_text.lower()

        # Check for weak verbs
        weak_found = []
        for weak, strong in self.WEAK_VERBS.items():
            if weak in text_lower:
                weak_found.append((weak, strong))

        if weak_found:
            replacements = [f"'{w}' → '{s}'" for w, s in weak_found[:5]]
            suggestions.append({
                "category": "💪 Action Verbs",
                "issue": f"Found {len(weak_found)} weak verb(s) that could be strengthened",
                "suggestion": f"Replace weak verbs with stronger alternatives: "
                              f"{', '.join(replacements)}. Strong action verbs make your "
                              f"achievements more impactful.",
                "impact": "Medium",
                "icon": "🟡",
            })

        # Check if any strong verbs are used
        strong_used = sum(1 for v in self.STRONG_VERBS if v in text_lower)
        if strong_used < 3:
            suggestions.append({
                "category": "💪 Action Verbs",
                "issue": "Few strong action verbs detected",
                "suggestion": f"Use more impactful action verbs like: "
                              f"{', '.join(self.STRONG_VERBS[:8])}. "
                              f"These make your contributions stand out.",
                "impact": "Medium",
                "icon": "🟡",
            })

        return suggestions

    def _analyze_quantification(self, resume_text: str) -> List[Dict]:
        """Check for quantified achievements."""
        suggestions = []

        # Count numbers/percentages in text
        numbers = re.findall(r'\d+%|\d+\+|\$[\d,]+|\d{2,}', resume_text)

        if len(numbers) < 3:
            suggestions.append({
                "category": "📊 Quantification",
                "issue": "Limited use of metrics and numbers",
                "suggestion": "Add quantified achievements wherever possible. Examples: "
                              "'Improved performance by 40%', 'Managed team of 8', "
                              "'Reduced costs by $50K', 'Processed 10K+ transactions daily'.",
                "impact": "High",
                "icon": "🔴",
            })
        elif len(numbers) < 6:
            suggestions.append({
                "category": "📊 Quantification",
                "issue": "More quantification would strengthen your resume",
                "suggestion": "You have some metrics, but adding more quantified results "
                              "will make your resume more compelling. Aim for at least "
                              "1-2 metrics per role.",
                "impact": "Medium",
                "icon": "🟡",
            })

        return suggestions

    def _analyze_formatting(self, resume_text: str) -> List[Dict]:
        """Analyze resume length and formatting."""
        suggestions = []
        word_count = len(resume_text.split())

        if word_count < 150:
            suggestions.append({
                "category": "📝 Formatting",
                "issue": f"Resume is very short ({word_count} words)",
                "suggestion": "Your resume appears too brief. Aim for 400-800 words "
                              "for a standard 1-page resume. Add more details about "
                              "your experience, projects, and skills.",
                "impact": "High",
                "icon": "🔴",
            })
        elif word_count > 1200:
            suggestions.append({
                "category": "📝 Formatting",
                "issue": f"Resume may be too long ({word_count} words)",
                "suggestion": "Consider condensing to 1-2 pages. Focus on the most "
                              "relevant and recent experience. Remove dated or "
                              "irrelevant information.",
                "impact": "Low",
                "icon": "🟢",
            })

        return suggestions

    def _analyze_technical_depth(self, skills: List[str], target_role: str) -> List[Dict]:
        """Analyze technical skill depth for target role."""
        suggestions = []

        if len(skills) < 5:
            suggestions.append({
                "category": "🛠️ Technical Depth",
                "issue": "Limited technical skills detected",
                "suggestion": "Add more specific technical skills relevant to your "
                              "target role. Include tools, frameworks, and technologies "
                              "you've used.",
                "impact": "Medium",
                "icon": "🟡",
            })

        return suggestions

    def _get_grade(self, score: int) -> Dict:
        """Get letter grade based on score."""
        if score >= 90:
            return {"grade": "A+", "emoji": "🌟", "message": "Outstanding resume!"}
        elif score >= 80:
            return {"grade": "A", "emoji": "⭐", "message": "Excellent resume with minor tweaks needed."}
        elif score >= 70:
            return {"grade": "B+", "emoji": "👍", "message": "Good resume — follow suggestions to improve."}
        elif score >= 60:
            return {"grade": "B", "emoji": "📝", "message": "Solid foundation — address key suggestions."}
        elif score >= 50:
            return {"grade": "C", "emoji": "⚠️", "message": "Needs improvement — focus on high-impact items."}
        else:
            return {"grade": "D", "emoji": "🔧", "message": "Significant revision recommended."}
