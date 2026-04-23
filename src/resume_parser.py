"""
Resume Parser — Extracts text from PDF and plain text files.
Detects resume sections, contact info, and structures content.
"""

import re
import io
from typing import Dict, List, Optional


class ResumeParser:
    """Production-grade resume parsing engine supporting PDF and text input."""

    # Section header patterns
    SECTION_PATTERNS = {
        "education": r"(?i)\b(education|academic|qualification|degree|university|college|school)\b",
        "experience": r"(?i)\b(experience|employment|work\s*history|professional\s*background|career)\b",
        "skills": r"(?i)\b(skills|technical\s*skills|core\s*competencies|proficiency|expertise|technologies)\b",
        "projects": r"(?i)\b(projects|portfolio|assignments|case\s*studies)\b",
        "certifications": r"(?i)\b(certif|licenses?|accreditation|credential)\b",
        "summary": r"(?i)\b(summary|profile|about\s*me|professional\s*summary|overview|objective)\b",
        "achievements": r"(?i)\b(achievement|accomplishment|award|honor|recognition)\b",
        "publications": r"(?i)\b(publication|research|paper|journal|conference)\b",
        "languages": r"(?i)\b(language|linguistic|fluent)\b",
        "interests": r"(?i)\b(interest|hobbi|activit|volunteer)\b",
    }

    # Contact info patterns
    EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    PHONE_PATTERN = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}'
    LINKEDIN_PATTERN = r'(?:linkedin\.com/in/|linkedin\.com/pub/)[\w\-]+'
    GITHUB_PATTERN = r'(?:github\.com/)[\w\-]+'

    def parse_pdf(self, uploaded_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            import pdfplumber
            text_parts = []
            pdf_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset for potential re-read

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return '\n'.join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")

    def parse_text(self, text: str) -> str:
        """Clean and normalize plain text input."""
        if not text:
            return ""
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information from resume text."""
        contact = {
            "name": self._extract_name(text),
            "email": None,
            "phone": None,
            "linkedin": None,
            "github": None,
        }

        # Email
        emails = re.findall(self.EMAIL_PATTERN, text)
        if emails:
            contact["email"] = emails[0]

        # Phone
        phones = re.findall(self.PHONE_PATTERN, text)
        if phones:
            contact["phone"] = phones[0].strip()

        # LinkedIn
        linkedin = re.findall(self.LINKEDIN_PATTERN, text, re.IGNORECASE)
        if linkedin:
            contact["linkedin"] = linkedin[0]

        # GitHub
        github = re.findall(self.GITHUB_PATTERN, text, re.IGNORECASE)
        if github:
            contact["github"] = github[0]

        return contact

    def _extract_name(self, text: str) -> Optional[str]:
        """Attempt to extract name from the first few lines of resume."""
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
            # Skip lines that look like addresses, phones, emails
            if re.search(self.EMAIL_PATTERN, line):
                continue
            if re.search(self.PHONE_PATTERN, line):
                continue
            if len(line) < 3 or len(line) > 50:
                continue
            # Name is usually the first non-empty, non-contact line
            words = line.split()
            if 1 < len(words) <= 4:
                if all(w[0].isupper() or w[0] == '.' for w in words if w):
                    return line
        return None

    def detect_sections(self, text: str) -> Dict[str, str]:
        """Detect and extract resume sections."""
        sections = {}
        lines = text.split('\n')
        current_section = "header"
        current_content = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_content.append("")
                continue

            # Check if this line is a section header
            detected_section = None
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.search(pattern, stripped):
                    # Section headers are typically short lines
                    if len(stripped.split()) <= 6:
                        detected_section = section_name
                        break

            if detected_section:
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections[current_section] = content
                current_section = detected_section
                current_content = []
            else:
                current_content.append(stripped)

        # Save last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections[current_section] = content

        return sections

    def extract_experience_years(self, text: str) -> Optional[int]:
        """Estimate years of experience from resume text."""
        # Pattern: "X years of experience"
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
            r'(?:experience|exp)\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        # Count date ranges (e.g., 2018-2022)
        date_ranges = re.findall(r'(20[0-2]\d)\s*[-–—to]+\s*(20[0-2]\d|present|current)', text, re.IGNORECASE)
        if date_ranges:
            total_years = 0
            for start, end in date_ranges:
                start_year = int(start)
                end_year = 2025 if end.lower() in ['present', 'current'] else int(end)
                total_years += max(0, end_year - start_year)
            return total_years if total_years > 0 else None

        return None

    def full_parse(self, text: str, source: str = "text") -> Dict:
        """Complete resume parsing pipeline."""
        cleaned_text = self.parse_text(text)
        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "contact_info": self.extract_contact_info(cleaned_text),
            "sections": self.detect_sections(cleaned_text),
            "experience_years": self.extract_experience_years(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "source": source,
        }
