"""
Configuration settings for AI Resume Analyzer System.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

RESUMES_CSV = os.path.join(DATA_DIR, "resumes.csv")
JOB_DESC_CSV = os.path.join(DATA_DIR, "job_descriptions.csv")
SKILLS_DB_JSON = os.path.join(DATA_DIR, "skills_database.json")
SAMPLE_RESUMES_DIR = os.path.join(DATA_DIR, "sample_resumes")

JOB_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "job_embeddings.npy")
JOB_IDS_PATH = os.path.join(MODELS_DIR, "job_ids.npy")
SKILL_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "skill_embeddings.npy")
SKILL_NAMES_PATH = os.path.join(MODELS_DIR, "skill_names.npy")

DB_PATH = os.path.join(BASE_DIR, "database", "resume_analyzer.db")

# ── Embedding Model ───────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Job Categories ────────────────────────────────────────────────────────
JOB_CATEGORIES = [
    "Data Science", "HR", "Advocate", "Arts", "Web Designing",
    "Mechanical Engineer", "Sales", "Health and Fitness",
    "Civil Engineer", "Java Developer", "Business Analyst",
    "SAP Developer", "Automation Testing", "Electrical Engineering",
    "Operations Manager", "Python Developer", "DevOps Engineer",
    "Network Security Engineer", "Database Administrator",
    "Hadoop Developer", "ETL Developer", "DotNet Developer",
    "Blockchain Developer", "Testing", "Full Stack Developer"
]

# ── Matching Weights ──────────────────────────────────────────────────────
MATCH_WEIGHTS = {
    "text_similarity": 0.55,
    "skill_overlap": 0.35,
    "experience_match": 0.10,
}

# ── Recommendation Settings ──────────────────────────────────────────────
TOP_N_RECOMMENDATIONS = 5
MIN_MATCH_SCORE = 0.15

# ── Skill Gap Severity ───────────────────────────────────────────────────
GAP_SEVERITY = {
    "critical": {"threshold": 0.8, "label": "🔴 Critical", "color": "#FF4444"},
    "important": {"threshold": 0.5, "label": "🟠 Important", "color": "#FF8C00"},
    "nice_to_have": {"threshold": 0.0, "label": "🟢 Nice to Have", "color": "#44BB44"},
}

# ── Experience Levels ────────────────────────────────────────────────────
EXPERIENCE_LEVELS = ["Entry Level", "Mid Level", "Senior", "Lead", "Director"]

# ── Resume Sections ──────────────────────────────────────────────────────
RESUME_SECTIONS = [
    "education", "experience", "skills", "projects",
    "certifications", "summary", "objective", "achievements",
    "publications", "languages", "interests"
]

# ── Dashboard Theme ──────────────────────────────────────────────────────
THEME = {
    "primary": "#6C63FF",
    "secondary": "#00D2FF",
    "accent": "#FF6B6B",
    "success": "#00E676",
    "warning": "#FFD740",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1E2E",
    "bg_card_hover": "#242842",
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0B0B0",
    "gradient_1": "linear-gradient(135deg, #6C63FF 0%, #00D2FF 100%)",
    "gradient_2": "linear-gradient(135deg, #FF6B6B 0%, #FFD740 100%)",
    "gradient_3": "linear-gradient(135deg, #00E676 0%, #00D2FF 100%)",
}
