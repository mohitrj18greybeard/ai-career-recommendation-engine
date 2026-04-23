"""
Report Generator — Professional PDF report export using FPDF2.
Generates downloadable analysis reports with charts and insights.
"""

import io
import os
from datetime import datetime
from typing import Dict, List, Optional
from fpdf import FPDF


class ReportGenerator:
    """Generates professional PDF reports for resume analysis results."""

    def __init__(self):
        self.pdf = None

    def generate_report(
        self,
        contact_info: Dict,
        skills: List[Dict],
        recommendations: List[Dict],
        gap_analysis: Dict,
        improvement: Dict,
        resume_score: float = 0,
    ) -> bytes:
        """
        Generate a comprehensive PDF analysis report.

        Returns PDF as bytes for download.
        """
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=20)

        # ── Cover Page ──────────────────────────────────────────────
        self._add_cover_page(contact_info)

        # ── Skills Analysis ─────────────────────────────────────────
        self._add_skills_section(skills)

        # ── Job Recommendations ─────────────────────────────────────
        self._add_recommendations_section(recommendations)

        # ── Skill Gap Analysis ──────────────────────────────────────
        if gap_analysis:
            self._add_gap_analysis_section(gap_analysis)

        # ── Improvement Suggestions ────────────────────────────────
        if improvement:
            self._add_improvement_section(improvement)

        # Output
        return self.pdf.output()

    def _add_cover_page(self, contact_info: Dict):
        """Add professional cover page."""
        self.pdf.add_page()

        # Title
        self.pdf.set_font('Helvetica', 'B', 28)
        self.pdf.set_text_color(108, 99, 255)  # Primary purple
        self.pdf.cell(0, 20, '', ln=True)
        self.pdf.cell(0, 15, 'AI Resume Analysis', ln=True, align='C')
        self.pdf.cell(0, 12, 'Report', ln=True, align='C')

        # Divider
        self.pdf.set_draw_color(108, 99, 255)
        self.pdf.set_line_width(0.8)
        self.pdf.line(60, self.pdf.get_y() + 5, 150, self.pdf.get_y() + 5)
        self.pdf.cell(0, 15, '', ln=True)

        # Contact info
        self.pdf.set_font('Helvetica', '', 12)
        self.pdf.set_text_color(60, 60, 60)

        name = contact_info.get("name", "Candidate")
        if name:
            self.pdf.set_font('Helvetica', 'B', 16)
            self.pdf.cell(0, 10, name, ln=True, align='C')
            self.pdf.set_font('Helvetica', '', 12)

        email = contact_info.get("email", "")
        if email:
            self.pdf.cell(0, 8, f'Email: {email}', ln=True, align='C')

        phone = contact_info.get("phone", "")
        if phone:
            self.pdf.cell(0, 8, f'Phone: {phone}', ln=True, align='C')

        self.pdf.cell(0, 15, '', ln=True)

        # Date
        self.pdf.set_font('Helvetica', 'I', 10)
        self.pdf.set_text_color(120, 120, 120)
        self.pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', ln=True, align='C')
        self.pdf.cell(0, 6, 'Powered by AI Resume Analyzer', ln=True, align='C')

    def _add_section_header(self, title: str):
        """Add a styled section header."""
        self.pdf.cell(0, 8, '', ln=True)
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.set_text_color(108, 99, 255)
        self.pdf.cell(0, 10, title, ln=True)
        self.pdf.set_draw_color(108, 99, 255)
        self.pdf.set_line_width(0.5)
        self.pdf.line(10, self.pdf.get_y(), 200, self.pdf.get_y())
        self.pdf.cell(0, 5, '', ln=True)
        self.pdf.set_text_color(40, 40, 40)

    def _add_skills_section(self, skills: List[Dict]):
        """Add skills analysis section."""
        self.pdf.add_page()
        self._add_section_header('Skills Analysis')

        self.pdf.set_font('Helvetica', '', 11)
        self.pdf.cell(0, 8, f'Total Skills Detected: {len(skills)}', ln=True)
        self.pdf.cell(0, 5, '', ln=True)

        # Skills table
        if skills:
            # Header
            self.pdf.set_font('Helvetica', 'B', 10)
            self.pdf.set_fill_color(108, 99, 255)
            self.pdf.set_text_color(255, 255, 255)
            self.pdf.cell(60, 8, 'Skill', border=1, fill=True)
            self.pdf.cell(45, 8, 'Category', border=1, fill=True)
            self.pdf.cell(30, 8, 'Match Type', border=1, fill=True)
            self.pdf.cell(30, 8, 'Confidence', border=1, fill=True)
            self.pdf.ln()

            # Data rows
            self.pdf.set_font('Helvetica', '', 9)
            self.pdf.set_text_color(40, 40, 40)
            for i, skill in enumerate(skills[:25]):
                if i % 2 == 0:
                    self.pdf.set_fill_color(245, 245, 250)
                else:
                    self.pdf.set_fill_color(255, 255, 255)

                name = str(skill.get('name', ''))[:25]
                cat = str(skill.get('category', ''))[:20]
                match_type = str(skill.get('match_type', ''))[:15]
                conf = f"{skill.get('confidence', 0):.0%}"

                self.pdf.cell(60, 7, name, border=1, fill=True)
                self.pdf.cell(45, 7, cat, border=1, fill=True)
                self.pdf.cell(30, 7, match_type, border=1, fill=True)
                self.pdf.cell(30, 7, conf, border=1, fill=True)
                self.pdf.ln()

    def _add_recommendations_section(self, recommendations: List[Dict]):
        """Add job recommendations section."""
        self.pdf.add_page()
        self._add_section_header('Job Recommendations')

        for i, rec in enumerate(recommendations[:5], 1):
            self.pdf.set_font('Helvetica', 'B', 12)
            self.pdf.set_text_color(108, 99, 255)
            title = str(rec.get('title', 'Unknown'))
            score = rec.get('final_score', 0)
            self.pdf.cell(0, 10, f'{i}. {title} (Match: {score:.0%})', ln=True)

            self.pdf.set_font('Helvetica', '', 10)
            self.pdf.set_text_color(40, 40, 40)

            # Explanation
            explanations = rec.get('explanation', [])
            for exp in explanations:
                # Remove emoji for PDF
                clean_exp = exp.encode('ascii', 'ignore').decode('ascii').strip()
                if clean_exp:
                    self.pdf.multi_cell(0, 6, f'  - {clean_exp}')

            # Matching skills
            matching = rec.get('matching_skills', [])
            if matching:
                self.pdf.set_font('Helvetica', 'I', 9)
                self.pdf.cell(0, 6, f'  Matching Skills: {", ".join(matching[:8])}', ln=True)

            self.pdf.cell(0, 4, '', ln=True)

    def _add_gap_analysis_section(self, gap_analysis: Dict):
        """Add skill gap analysis section."""
        self.pdf.add_page()
        self._add_section_header('Skill Gap Analysis')

        self.pdf.set_font('Helvetica', '', 11)
        self.pdf.cell(0, 8, f'Target Role: {gap_analysis.get("target_role", "N/A")}', ln=True)
        self.pdf.cell(0, 8, f'Readiness Score: {gap_analysis.get("readiness_score", 0):.1f}%', ln=True)
        self.pdf.cell(0, 8, f'Skills Matched: {gap_analysis.get("matched_count", 0)} / {gap_analysis.get("total_required", 0)}', ln=True)
        self.pdf.cell(0, 5, '', ln=True)

        # Gap details
        gaps = gap_analysis.get("gap_details", [])
        if gaps:
            self.pdf.set_font('Helvetica', 'B', 11)
            self.pdf.cell(0, 8, 'Missing Skills & Learning Paths:', ln=True)
            self.pdf.set_font('Helvetica', '', 10)

            for gap in gaps[:15]:
                skill = gap.get('skill', '')
                severity = gap.get('severity', '').encode('ascii', 'ignore').decode('ascii').strip()
                platform = gap.get('learning_path', {}).get('recommended_platform', '')
                time_est = gap.get('learning_path', {}).get('estimated_time', '')

                self.pdf.set_font('Helvetica', 'B', 10)
                self.pdf.cell(0, 7, f'  {skill} [{severity}]', ln=True)
                self.pdf.set_font('Helvetica', '', 9)
                self.pdf.cell(0, 6, f'    Platform: {platform}', ln=True)
                self.pdf.cell(0, 6, f'    Est. Time: {time_est}', ln=True)

    def _add_improvement_section(self, improvement: Dict):
        """Add improvement suggestions section."""
        self.pdf.add_page()
        self._add_section_header('Resume Improvement Suggestions')

        grade = improvement.get('grade', {})
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.cell(0, 10, f'Resume Grade: {grade.get("grade", "N/A")} ({improvement.get("overall_score", 0)}/100)', ln=True)

        self.pdf.set_font('Helvetica', '', 10)
        self.pdf.cell(0, 8, f'Total Suggestions: {improvement.get("total_suggestions", 0)}', ln=True)
        self.pdf.cell(0, 5, '', ln=True)

        for sugg in improvement.get('suggestions', [])[:15]:
            category = str(sugg.get('category', '')).encode('ascii', 'ignore').decode('ascii').strip()
            issue = str(sugg.get('issue', ''))
            suggestion = str(sugg.get('suggestion', ''))
            impact = str(sugg.get('impact', ''))

            self.pdf.set_font('Helvetica', 'B', 10)
            self.pdf.cell(0, 7, f'{category} [{impact} Impact]', ln=True)
            self.pdf.set_font('Helvetica', '', 9)
            self.pdf.set_text_color(180, 0, 0)
            self.pdf.cell(0, 6, f'  Issue: {issue}', ln=True)
            self.pdf.set_text_color(0, 120, 0)
            self.pdf.multi_cell(0, 6, f'  Fix: {suggestion}')
            self.pdf.set_text_color(40, 40, 40)
            self.pdf.cell(0, 3, '', ln=True)
