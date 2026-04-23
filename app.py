"""
AI Resume Analyzer & Job Recommendation System
═══════════════════════════════════════════════
Premium Streamlit Dashboard — NLP-Powered Resume Intelligence

Features:
- Resume parsing (PDF/Text)
- BERT-powered skill extraction (Dictionary + Semantic)
- Similarity-based job recommendations with explainability
- Resume vs Job Description comparison
- Skill gap analysis with learning paths
- AI-powered improvement suggestions
- PDF report export
"""

import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ── Path Setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    THEME, JOB_CATEGORIES, SAMPLE_RESUMES_DIR, DATA_DIR,
    JOB_DESC_CSV, SKILLS_DB_JSON, MODELS_DIR
)
from src.text_processor import TextProcessor
from src.resume_parser import ResumeParser
from src.embedding_engine import EmbeddingEngine
from src.skill_extractor import SkillExtractor
from src.job_matcher import JobMatcher
from src.recommender import Recommender
from src.skill_gap_analyzer import SkillGapAnalyzer
from src.resume_improver import ResumeImprover
from src.report_generator import ReportGenerator
from src.visualizer import Visualizer
from database.db_manager import DatabaseManager

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Resume Analyzer | NLP-Powered Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Theme with Glassmorphism
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Import Fonts ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global Styles ────────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide Streamlit Defaults ──────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Glassmorphism Card ────────────────────────────────────────── */
    .glass-card {
        background: rgba(26, 30, 46, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(108, 99, 255, 0.4);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
        transform: translateY(-2px);
    }

    /* ── Hero Section ──────────────────────────────────────────────── */
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6C63FF 0%, #00D2FF 50%, #00E676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #B0B0B0;
        text-align: center;
        margin-top: 8px;
        font-weight: 400;
    }

    /* ── Metric Cards ──────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.1) 0%, rgba(0,210,255,0.05) 100%);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(108,99,255,0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #B0B0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Skill Badge ───────────────────────────────────────────────── */
    .skill-badge {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        border: 1px solid rgba(108, 99, 255, 0.3);
        color: #B8B3FF;
        padding: 5px 14px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.82rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .skill-badge:hover {
        background: rgba(108, 99, 255, 0.3);
        transform: scale(1.05);
    }
    .skill-badge-semantic {
        background: rgba(0, 210, 255, 0.15);
        border-color: rgba(0, 210, 255, 0.3);
        color: #80E8FF;
    }

    /* ── Match Score Bar ───────────────────────────────────────────── */
    .match-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        overflow: hidden;
        margin: 8px 0;
    }
    .match-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease;
    }

    /* ── Recommendation Card ───────────────────────────────────────── */
    .rec-card {
        background: rgba(26, 30, 46, 0.9);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid #6C63FF;
        transition: all 0.3s ease;
    }
    .rec-card:hover {
        border-left-color: #00D2FF;
        box-shadow: 0 5px 20px rgba(108, 99, 255, 0.15);
    }

    /* ── Gap Severity Badges ───────────────────────────────────────── */
    .severity-critical {
        color: #FF4444;
        font-weight: 600;
    }
    .severity-important {
        color: #FF8C00;
        font-weight: 600;
    }
    .severity-nice {
        color: #44BB44;
        font-weight: 600;
    }

    /* ── Section Header ────────────────────────────────────────────── */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(108, 99, 255, 0.3);
    }

    /* ── Sidebar Styling ───────────────────────────────────────────── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1E2E 100%);
    }

    /* ── Upload Area ───────────────────────────────────────────────── */
    .stFileUploader > div {
        border: 2px dashed rgba(108, 99, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }

    /* ── Expander Styling ──────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
    }

    /* ── Progress Animation ────────────────────────────────────────── */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(108, 99, 255, 0.3); }
        50% { box-shadow: 0 0 20px rgba(108, 99, 255, 0.6); }
    }
    .pulse-glow {
        animation: pulse-glow 2s infinite;
    }

    /* ── Tab Styling ───────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* ── Footer ────────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        padding: 20px 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES — Load Once
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_embedding_engine():
    """Load the Sentence Transformer model (cached — loads only once)."""
    engine = EmbeddingEngine(use_transformers=True)
    # Warm up
    engine.encode("warm up sentence")
    return engine

@st.cache_resource(show_spinner=False)
def load_components():
    """Load all ML components."""
    engine = load_embedding_engine()
    text_processor = TextProcessor()
    parser = ResumeParser()
    skill_extractor = SkillExtractor(embedding_engine=engine)
    job_matcher = JobMatcher(embedding_engine=engine)
    recommender = Recommender(job_matcher=job_matcher, skill_extractor=skill_extractor)
    gap_analyzer = SkillGapAnalyzer()
    improver = ResumeImprover()
    report_gen = ReportGenerator()
    visualizer = Visualizer()
    db = DatabaseManager()
    return {
        "engine": engine,
        "text_processor": text_processor,
        "parser": parser,
        "skill_extractor": skill_extractor,
        "job_matcher": job_matcher,
        "recommender": recommender,
        "gap_analyzer": gap_analyzer,
        "improver": improver,
        "report_gen": report_gen,
        "visualizer": visualizer,
        "db": db,
    }


def load_sample_resume(filename: str) -> str:
    """Load a sample resume from the data directory."""
    path = os.path.join(SAMPLE_RESUMES_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the premium sidebar navigation."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 15px 0;">
            <div style="font-size: 2.5rem;">🧠</div>
            <div style="font-size: 1.2rem; font-weight: 700; background: linear-gradient(135deg, #6C63FF, #00D2FF);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                AI Resume Analyzer
            </div>
            <div style="font-size: 0.75rem; color: #888; margin-top: 2px;">NLP-Powered Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "**Navigation**",
            [
                "🏠 Home",
                "📄 Resume Analyzer",
                "💼 Job Recommender",
                "🔍 Resume vs Job",
                "📊 Skill Gap & Improve",
                "🌍 Market Insights",
                "📥 Export & History",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Quick stats
        try:
            components = load_components()
            analytics = components["db"].get_analytics()
            st.markdown(f"""
            <div class="metric-card" style="margin: 5px 0;">
                <div class="metric-value" style="font-size: 1.5rem;">{analytics['total_analyses']}</div>
                <div class="metric-label">Analyses Done</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.7rem; color: #666;">
            Powered by Sentence Transformers<br>
            BERT • NLTK • Scikit-learn<br>
            v2.0 © 2025
        </div>
        """, unsafe_allow_html=True)

        return page


# ══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════

def render_home():
    """Render the premium home page."""
    st.markdown("""
    <div class="hero-title">AI Resume Analyzer</div>
    <div class="hero-subtitle">
        NLP-Powered Resume Intelligence • BERT Embeddings • Explainable Recommendations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">📄</div>
            <div style="font-size: 1.1rem; font-weight: 700; margin: 8px 0; color: #FFF;">Smart Resume Parsing</div>
            <div style="color: #B0B0B0; font-size: 0.85rem;">
                Upload PDF or paste text. Our NLP engine extracts skills, experience,
                and contact info using dictionary + semantic matching.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">🎯</div>
            <div style="font-size: 1.1rem; font-weight: 700; margin: 8px 0; color: #FFF;">Explainable Matching</div>
            <div style="color: #B0B0B0; font-size: 0.85rem;">
                BERT-powered cosine similarity matching with transparent
                "Why this job?" explanations and score breakdowns.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">📊</div>
            <div style="font-size: 1.1rem; font-weight: 700; margin: 8px 0; color: #FFF;">Gap Analysis & Improve</div>
            <div style="color: #B0B0B0; font-size: 0.85rem;">
                Identify missing skills, get severity-graded learning paths,
                and AI-powered resume improvement suggestions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.1rem; font-weight: 700; color: #FFF; margin-bottom: 12px;">
                🏗️ System Architecture
            </div>
            <div style="color: #B0B0B0; font-size: 0.85rem; line-height: 1.8;">
                <code style="color: #6C63FF;">1.</code> Resume Upload (PDF / Text)<br>
                <code style="color: #6C63FF;">2.</code> Text Preprocessing (NLTK)<br>
                <code style="color: #6C63FF;">3.</code> BERT Encoding (all-MiniLM-L6-v2)<br>
                <code style="color: #6C63FF;">4.</code> Skill Extraction (Dictionary + Semantic)<br>
                <code style="color: #6C63FF;">5.</code> Cosine Similarity Matching<br>
                <code style="color: #6C63FF;">6.</code> Explainable Recommendations<br>
                <code style="color: #6C63FF;">7.</code> Gap Analysis & Improvements<br>
                <code style="color: #6C63FF;">8.</code> PDF Report Export
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.1rem; font-weight: 700; color: #FFF; margin-bottom: 12px;">
                🛠️ Tech Stack
            </div>
            <div style="color: #B0B0B0; font-size: 0.85rem; line-height: 1.8;">
                <code style="color: #00D2FF;">NLP</code> Sentence Transformers, NLTK<br>
                <code style="color: #00D2FF;">ML</code> Scikit-learn, Cosine Similarity<br>
                <code style="color: #00D2FF;">Embeddings</code> all-MiniLM-L6-v2 (BERT)<br>
                <code style="color: #00D2FF;">Data</code> Pandas, NumPy<br>
                <code style="color: #00D2FF;">Viz</code> Plotly, Matplotlib, Seaborn<br>
                <code style="color: #00D2FF;">Dashboard</code> Streamlit<br>
                <code style="color: #00D2FF;">Database</code> SQLite (PostgreSQL-ready)<br>
                <code style="color: #00D2FF;">Export</code> FPDF2 (PDF Reports)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quick start
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="text-align: center; border-color: rgba(0,230,118,0.3);">
        <div style="font-size: 1.2rem; font-weight: 700; color: #00E676;">🚀 Quick Start</div>
        <div style="color: #B0B0B0; margin-top: 8px; font-size: 0.9rem;">
            Navigate to <strong>📄 Resume Analyzer</strong> in the sidebar to upload your resume
            and get instant AI-powered analysis, job recommendations, and improvement suggestions.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: RESUME ANALYZER
# ══════════════════════════════════════════════════════════════════════════

def render_resume_analyzer():
    """Render the resume analysis page."""
    st.markdown('<div class="section-header">📄 Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Upload your resume or paste text to get comprehensive NLP-powered analysis.")

    components = load_components()

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        input_method = st.radio(
            "Input Method",
            ["📎 Upload PDF", "📝 Paste Text", "📂 Sample Resume"],
            horizontal=True,
        )

    resume_text = ""

    if input_method == "📎 Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF)",
            type=["pdf"],
            help="Upload a PDF resume for analysis",
        )
        if uploaded_file:
            with st.spinner("Parsing PDF..."):
                resume_text = components["parser"].parse_pdf(uploaded_file)

    elif input_method == "📝 Paste Text":
        resume_text = st.text_area(
            "Paste your resume text",
            height=300,
            placeholder="Paste your resume content here...",
        )

    elif input_method == "📂 Sample Resume":
        sample_options = {
            "Data Scientist": "data_scientist.txt",
            "Web Developer": "web_developer.txt",
            "DevOps Engineer": "devops_engineer.txt",
            "Business Analyst": "business_analyst.txt",
            "Java Developer": "java_developer.txt",
        }
        selected_sample = st.selectbox("Choose a sample resume", list(sample_options.keys()))
        resume_text = load_sample_resume(sample_options[selected_sample])
        if resume_text:
            with st.expander("📖 Preview Resume Text", expanded=False):
                st.text(resume_text[:2000])

    # Analyze button
    if resume_text and st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        _run_analysis(resume_text, components)


def _run_analysis(resume_text: str, components: dict):
    """Run the full analysis pipeline."""
    progress = st.progress(0, text="Initializing analysis...")

    # Step 1: Parse resume
    progress.progress(10, text="📄 Parsing resume...")
    parsed = components["parser"].full_parse(resume_text)
    time.sleep(0.3)

    # Step 2: Preprocess text
    progress.progress(20, text="🔤 Preprocessing text...")
    processed_text = components["text_processor"].process(resume_text)
    time.sleep(0.3)

    # Step 3: Extract skills
    progress.progress(40, text="🎯 Extracting skills (Dictionary + Semantic)...")
    skills = components["skill_extractor"].extract_all_skills(resume_text)
    skill_names = [s["name"] for s in skills]
    time.sleep(0.3)

    # Step 4: Get recommendations
    progress.progress(60, text="💼 Computing job recommendations (BERT matching)...")
    recommendations = components["recommender"].get_recommendations(
        resume_text=processed_text,
        resume_skills=skill_names,
        experience_years=parsed.get("experience_years"),
        top_n=5,
    )
    time.sleep(0.3)

    # Step 5: Gap analysis (for top role)
    progress.progress(80, text="📊 Analyzing skill gaps...")
    top_role = recommendations[0]["category"] if recommendations else "Data Science"
    top_job_skills_str = recommendations[0].get("required_skills", "") if recommendations else ""
    top_job_skills = [s.strip() for s in top_job_skills_str.split(',') if s.strip()]

    gap_analysis = components["gap_analyzer"].analyze_gap(
        resume_skills=skill_names,
        target_role=top_role,
        job_required_skills=top_job_skills if top_job_skills else None,
    )

    # Step 6: Improvement suggestions
    progress.progress(90, text="💡 Generating improvement suggestions...")
    improvement = components["improver"].analyze_resume(
        resume_text=resume_text,
        sections=parsed.get("sections", {}),
        extracted_skills=skill_names,
        target_role=top_role,
    )

    # Step 7: Save to database
    progress.progress(95, text="💾 Saving analysis...")
    contact = parsed.get("contact_info", {})
    try:
        components["db"].save_analysis(
            candidate_name=contact.get("name", "Unknown"),
            candidate_email=contact.get("email", ""),
            resume_text=resume_text,
            skills=skills,
            recommendations=recommendations,
            gap_analysis=gap_analysis,
            improvement=improvement,
            overall_score=recommendations[0]["final_score"] * 100 if recommendations else 0,
            top_role=top_role,
        )
    except Exception:
        pass

    progress.progress(100, text="✅ Analysis complete!")
    time.sleep(0.5)
    progress.empty()

    # Store results in session state
    st.session_state["analysis_results"] = {
        "parsed": parsed,
        "skills": skills,
        "recommendations": recommendations,
        "gap_analysis": gap_analysis,
        "improvement": improvement,
        "resume_text": resume_text,
    }

    # Display results
    _display_analysis_results(
        parsed, skills, recommendations, gap_analysis, improvement, components
    )


def _display_analysis_results(parsed, skills, recommendations, gap_analysis, improvement, components):
    """Display comprehensive analysis results."""
    viz = components["visualizer"]

    # ── Contact Info ──────────────────────────────────────────────────────
    contact = parsed.get("contact_info", {})
    st.markdown(f"""
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 1.4rem; font-weight: 700; color: #FFF;">
                    {contact.get('name', 'Candidate')}
                </span>
                <br>
                <span style="color: #B0B0B0; font-size: 0.85rem;">
                    {contact.get('email', '')} {'| ' + contact.get('phone', '') if contact.get('phone') else ''}
                </span>
            </div>
            <div style="text-align: right;">
                <span style="color: #888;">Words: {parsed.get('word_count', 0)}</span>
                {'<br><span style="color: #888;">Experience: ~' + str(parsed.get("experience_years", "N/A")) + ' years</span>' if parsed.get("experience_years") else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Row ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(skills)}</div>
            <div class="metric-label">Skills Found</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        top_score = recommendations[0]["final_score"] * 100 if recommendations else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top_score:.0f}%</div>
            <div class="metric-label">Best Match</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        readiness = gap_analysis.get("readiness_score", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{readiness:.0f}%</div>
            <div class="metric-label">Role Readiness</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        grade = improvement.get("grade", {}).get("grade", "N/A")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{grade}</div>
            <div class="metric-label">Resume Grade</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs for detailed results ─────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Skills", "💼 Recommendations", "📊 Skill Gap", "💡 Improvements"
    ])

    with tab1:
        _render_skills_tab(skills, viz)

    with tab2:
        _render_recommendations_tab(recommendations, viz)

    with tab3:
        _render_gap_tab(gap_analysis, viz)

    with tab4:
        _render_improvement_tab(improvement, viz)


def _render_skills_tab(skills, viz):
    """Render skills analysis tab."""
    if not skills:
        st.info("No skills detected. Try providing more detailed resume text.")
        return

    # Skill badges
    dict_skills = [s for s in skills if s.get("match_type") == "dictionary"]
    semantic_skills = [s for s in skills if s.get("match_type") == "semantic"]

    st.markdown("##### 📚 Dictionary-Matched Skills")
    if dict_skills:
        badges_html = ""
        for s in dict_skills:
            badges_html += f'<span class="skill-badge">{s["name"]}</span>'
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.caption("None found via dictionary matching.")

    if semantic_skills:
        st.markdown("##### 🧠 Semantically-Matched Skills")
        badges_html = ""
        for s in semantic_skills:
            badges_html += f'<span class="skill-badge skill-badge-semantic">{s["name"]} ({s["confidence"]:.0%})</span>'
        st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart
    categorized = {}
    for s in skills:
        cat = s.get("category", "General")
        categorized[cat] = categorized.get(cat, 0) + 1

    if categorized:
        st.markdown("##### 📊 Skill Distribution by Category")
        fig = viz.create_skill_radar(categorized)
        st.plotly_chart(fig, use_container_width=True)

    # Skills table
    with st.expander("📋 Detailed Skills Table"):
        df = pd.DataFrame(skills)
        if not df.empty:
            display_cols = ["name", "category", "match_type", "confidence", "frequency"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[available_cols].rename(columns={
                    "name": "Skill", "category": "Category",
                    "match_type": "Match Type", "confidence": "Confidence",
                    "frequency": "Frequency"
                }),
                use_container_width=True,
                hide_index=True,
            )


def _render_recommendations_tab(recommendations, viz):
    """Render job recommendations tab with explainability."""
    if not recommendations:
        st.info("No recommendations available. Ensure job descriptions data is loaded.")
        return

    # Score chart
    st.markdown("##### 🏆 Top Job Matches")
    fig = viz.create_recommendation_scores(recommendations)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed recommendation cards
    for i, rec in enumerate(recommendations):
        score = rec["final_score"]
        color = "#00E676" if score >= 0.7 else "#FFD740" if score >= 0.45 else "#FF6B6B"

        st.markdown(f"""
        <div class="rec-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #FFF;">
                        {i+1}. {rec['title']}
                    </span>
                    <br>
                    <span style="color: #888; font-size: 0.85rem;">{rec['category']} • {rec['experience_level']}</span>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 1.6rem; font-weight: 800; color: {color};">{score:.0%}</span>
                    <br>
                    <span style="color: #888; font-size: 0.8rem;">{rec['confidence']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"🔍 Why this job? — {rec['title']}"):
            # Explanations
            for exp in rec.get("explanation", []):
                st.markdown(f"  {exp}")

            # Score breakdown
            st.markdown("**Score Breakdown:**")
            fig = viz.create_score_breakdown_bar(rec.get("score_breakdown", {}))
            st.plotly_chart(fig, use_container_width=True, key=f"breakdown_{i}")

            # Matching skills
            matching = rec.get("matching_skills", [])
            missing = rec.get("missing_skills", [])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**✅ Matching Skills ({len(matching)})**")
                for s in matching:
                    st.markdown(f"  • {s}")
                if not matching:
                    st.caption("No direct matches found")
            with col2:
                st.markdown(f"**❌ Missing Skills ({len(missing)})**")
                for s in missing[:8]:
                    st.markdown(f"  • {s}")
                if not missing:
                    st.caption("All required skills present!")

            st.info(f"📈 {rec.get('improvement_potential', '')}")


def _render_gap_tab(gap_analysis, viz):
    """Render skill gap analysis tab."""
    if not gap_analysis:
        st.info("Run resume analysis first to see skill gap results.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        # Readiness gauge
        readiness = gap_analysis.get("readiness_score", 0) / 100
        fig = viz.create_match_gauge(readiness, f"Readiness for {gap_analysis.get('target_role', 'Role')}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Gap donut
        fig = viz.create_skill_gap_chart(
            gap_analysis.get("matched_count", 0),
            gap_analysis.get("missing_count", 0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Readiness level
    readiness_info = gap_analysis.get("readiness_level", {})
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <span style="font-size: 2rem;">{readiness_info.get('emoji', '📊')}</span>
        <span style="font-size: 1.2rem; font-weight: 700; color: #FFF;"> {readiness_info.get('level', 'N/A')}</span>
        <br>
        <span style="color: #B0B0B0; font-size: 0.9rem;">{readiness_info.get('message', '')}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"_{gap_analysis.get('summary', '')}_")

    # Missing skills with learning paths
    gaps = gap_analysis.get("gap_details", [])
    if gaps:
        st.markdown("##### 📚 Missing Skills & Learning Paths")
        for gap in gaps:
            severity_class = {
                "critical": "severity-critical",
                "important": "severity-important",
                "nice_to_have": "severity-nice",
            }.get(gap.get("severity_level", ""), "")

            learning = gap.get("learning_path", {})
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px 20px;">
                <span style="font-weight: 700; color: #FFF; font-size: 0.95rem;">{gap['skill']}</span>
                <span class="{severity_class}" style="float: right; font-size: 0.85rem;">{gap['severity']}</span>
                <br>
                <span style="color: #888; font-size: 0.8rem;">
                    📖 {learning.get('recommended_platform', 'N/A')} •
                    ⏱️ {learning.get('estimated_time', 'N/A')} •
                    📊 {learning.get('difficulty', 'N/A')}
                </span>
            </div>
            """, unsafe_allow_html=True)


def _render_improvement_tab(improvement, viz):
    """Render AI improvement suggestions tab."""
    if not improvement:
        st.info("Run resume analysis first to see improvement suggestions.")
        return

    grade = improvement.get("grade", {})

    # Grade card
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <span style="font-size: 3rem;">{grade.get('emoji', '📝')}</span>
        <br>
        <span style="font-size: 2.5rem; font-weight: 900; color: #FFF;">
            {grade.get('grade', 'N/A')}
        </span>
        <br>
        <span style="font-size: 1rem; color: #B0B0B0;">
            Score: {improvement.get('overall_score', 0)}/100 • {grade.get('message', '')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 High Impact", improvement.get("high_impact", 0))
    with col2:
        st.metric("🟡 Medium Impact", improvement.get("medium_impact", 0))
    with col3:
        st.metric("🟢 Low Impact", improvement.get("low_impact", 0))

    # Impact distribution chart
    fig = viz.create_improvement_chart(improvement.get("suggestions", []))
    st.plotly_chart(fig, use_container_width=True)

    # Suggestions list
    st.markdown("##### 💡 Actionable Suggestions")
    for sugg in improvement.get("suggestions", []):
        impact = sugg.get("impact", "Low")
        icon = sugg.get("icon", "💡")
        color = "#FF4444" if impact == "High" else "#FFD740" if impact == "Medium" else "#44BB44"

        st.markdown(f"""
        <div class="glass-card" style="padding: 14px 20px; border-left: 3px solid {color};">
            <div style="font-weight: 600; color: #FFF; font-size: 0.9rem;">
                {icon} {sugg.get('category', '')} — {sugg.get('issue', '')}
            </div>
            <div style="color: #B0B0B0; font-size: 0.85rem; margin-top: 6px;">
                {sugg.get('suggestion', '')}
            </div>
            <div style="color: {color}; font-size: 0.75rem; margin-top: 4px; font-weight: 600;">
                Impact: {impact}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: JOB RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════

def render_job_recommender():
    """Render standalone job recommender page."""
    st.markdown('<div class="section-header">💼 Job Recommender</div>', unsafe_allow_html=True)

    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]
        recommendations = results.get("recommendations", [])
        skills = results.get("skills", [])
        viz = load_components()["visualizer"]

        if recommendations:
            st.success(f"Showing recommendations based on your last analysis ({len([s for s in skills])} skills detected)")
            _render_recommendations_tab(recommendations, viz)
        else:
            st.warning("No recommendations found. Try uploading a more detailed resume.")
    else:
        st.info("👈 Please go to **Resume Analyzer** first to upload and analyze your resume. "
                "Recommendations will appear here after analysis.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: RESUME VS JOB COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def render_resume_vs_job():
    """Render resume vs job description comparison page."""
    st.markdown('<div class="section-header">🔍 Resume vs Job Description</div>', unsafe_allow_html=True)
    st.markdown("Compare your resume directly against a specific job description.")

    components = load_components()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📄 Your Resume")
        resume_input_method = st.radio(
            "Resume input",
            ["Use last analysis", "Paste new text", "Sample resume"],
            key="compare_resume_input",
        )

        if resume_input_method == "Use last analysis" and "analysis_results" in st.session_state:
            resume_text = st.session_state["analysis_results"]["resume_text"]
            st.text_area("Resume Preview", resume_text[:1000] + "...", height=200, disabled=True)
        elif resume_input_method == "Sample resume":
            sample_map = {
                "Data Scientist": "data_scientist.txt",
                "Web Developer": "web_developer.txt",
                "DevOps Engineer": "devops_engineer.txt",
            }
            selected = st.selectbox("Select sample", list(sample_map.keys()), key="cmp_sample")
            resume_text = load_sample_resume(sample_map[selected])
        else:
            resume_text = st.text_area("Paste resume text", height=200, key="compare_resume")

    with col2:
        st.markdown("##### 💼 Job Description")
        jd_text = st.text_area(
            "Paste job description",
            height=200,
            placeholder="Paste the target job description here...",
            key="compare_jd",
        )

    if resume_text and jd_text and st.button("🔍 Compare", type="primary", use_container_width=True):
        with st.spinner("Analyzing match..."):
            # Extract skills from resume
            skills = components["skill_extractor"].extract_all_skills(resume_text)
            skill_names = [s["name"] for s in skills]

            # Extract skills from JD
            jd_skills = components["skill_extractor"].extract_all_skills(jd_text)
            jd_skill_names = [s["name"] for s in jd_skills]

            # Compute comparison
            processed_resume = components["text_processor"].process(resume_text)
            processed_jd = components["text_processor"].process(jd_text)

            comparison = components["job_matcher"].compare_resume_to_job(
                resume_text=processed_resume,
                job_description=processed_jd,
                resume_skills=skill_names,
            )

        # Display results
        st.markdown("<br>", unsafe_allow_html=True)

        # Overall match gauge
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = components["visualizer"].create_match_gauge(
                comparison["text_similarity"], "Semantic Similarity"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = components["visualizer"].create_match_gauge(
                comparison["overall_match"], "Overall Match"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = components["visualizer"].create_match_gauge(
                comparison["keyword_overlap"], "Keyword Overlap"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Skill comparison
        st.markdown("##### 🎯 Skill Comparison")
        fig = components["visualizer"].create_comparison_bar(skill_names, jd_skill_names)
        st.plotly_chart(fig, use_container_width=True)

        # Matched skills
        common_skills = set(s.lower() for s in skill_names) & set(s.lower() for s in jd_skill_names)
        resume_only = set(s.lower() for s in skill_names) - common_skills
        jd_only = set(s.lower() for s in jd_skill_names) - common_skills

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**✅ Common Skills**")
            for s in common_skills:
                st.markdown(f"• {s.title()}")
            if not common_skills:
                st.caption("No overlap")
        with col2:
            st.markdown("**📄 Only in Resume**")
            for s in list(resume_only)[:10]:
                st.markdown(f"• {s.title()}")
        with col3:
            st.markdown("**💼 Only in Job**")
            for s in list(jd_only)[:10]:
                st.markdown(f"• {s.title()}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: SKILL GAP & IMPROVE
# ══════════════════════════════════════════════════════════════════════════

def render_skill_gap_improve():
    """Render skill gap analysis and improvement page."""
    st.markdown('<div class="section-header">📊 Skill Gap & Improvement</div>', unsafe_allow_html=True)

    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]
        gap_analysis = results.get("gap_analysis", {})
        improvement = results.get("improvement", {})
        viz = load_components()["visualizer"]

        tab1, tab2 = st.tabs(["📊 Skill Gap Analysis", "💡 Resume Improvement"])

        with tab1:
            # Let user select a different target role
            skill_names = [s["name"] for s in results.get("skills", [])]
            target_role = st.selectbox(
                "Select target role for gap analysis",
                JOB_CATEGORIES,
                index=JOB_CATEGORIES.index(gap_analysis.get("target_role", "Data Science")) if gap_analysis.get("target_role") in JOB_CATEGORIES else 0,
            )

            if st.button("🔄 Recalculate Gap", type="secondary"):
                components = load_components()
                gap_analysis = components["gap_analyzer"].analyze_gap(
                    resume_skills=skill_names,
                    target_role=target_role,
                )
                st.session_state["analysis_results"]["gap_analysis"] = gap_analysis

            _render_gap_tab(gap_analysis, viz)

        with tab2:
            _render_improvement_tab(improvement, viz)
    else:
        st.info("👈 Please go to **Resume Analyzer** first to upload and analyze your resume.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: EXPORT & HISTORY
# ══════════════════════════════════════════════════════════════════════════

def render_export_history():
    """Render export and history page."""
    st.markdown('<div class="section-header">📥 Export & History</div>', unsafe_allow_html=True)

    components = load_components()

    tab1, tab2 = st.tabs(["📥 PDF Export", "📜 Analysis History"])

    with tab1:
        if "analysis_results" in st.session_state:
            results = st.session_state["analysis_results"]
            st.success("Analysis results available for export!")

            if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating professional PDF report..."):
                    parsed = results.get("parsed", {})
                    pdf_bytes = components["report_gen"].generate_report(
                        contact_info=parsed.get("contact_info", {}),
                        skills=results.get("skills", []),
                        recommendations=results.get("recommendations", []),
                        gap_analysis=results.get("gap_analysis", {}),
                        improvement=results.get("improvement", {}),
                    )

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
                st.success("PDF generated successfully!")
        else:
            st.info("Run a resume analysis first to generate a PDF report.")

    with tab2:
        history = components["db"].get_recent_analyses(limit=20)

        if history:
            st.markdown(f"**{len(history)} recent analyses**")
            df = pd.DataFrame(history)
            df = df.rename(columns={
                "candidate_name": "Name",
                "candidate_email": "Email",
                "overall_score": "Match Score",
                "top_role": "Top Role",
                "created_at": "Date",
            })
            df["Match Score"] = df["Match Score"].round(1).astype(str) + "%"

            st.dataframe(
                df[["Name", "Email", "Match Score", "Top Role", "Date"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No analysis history yet. Analyze a resume to get started!")

        # Analytics
        analytics = components["db"].get_analytics()
        if analytics.get("total_analyses", 0) > 0:
            st.markdown("##### 📊 Platform Analytics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Analyses", analytics["total_analyses"])
            with col2:
                st.metric("Average Score", f"{analytics['avg_score']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    # Load components on startup
    with st.spinner("🧠 Loading AI models..."):
        load_components()

    # Sidebar navigation
    page = render_sidebar()

    # Page routing
    if page == "🏠 Home":
        render_home()
    elif page == "📄 Resume Analyzer":
        render_resume_analyzer()
    elif page == "💼 Job Recommender":
        render_job_recommender()
    elif page == "🔍 Resume vs Job":
        render_resume_vs_job()
    elif page == "📊 Skill Gap & Improve":
        render_skill_gap_improve()
    elif page == "🌍 Market Insights":
        render_market_insights()

    elif page == "📥 Export & History":
        render_export_history()

    # Footer
    st.markdown("""
    <div class="app-footer">
        Built with ❤️ using Python, Sentence Transformers (BERT), NLTK, Scikit-learn & Streamlit<br>
        AI Resume Analyzer v2.0 • NLP-Powered Intelligence
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------
# PAGE: MARKET INSIGHTS
# --------------------------------------------------------------------------

def render_market_insights():
    """Render the global platform analytics dashboard."""
    st.markdown('<div class="section-header">🌍 Global Market Insights</div>', unsafe_allow_html=True)
    st.markdown("Explore aggregate trends across the entire resume database to identify high-demand skills and popular career paths.")

    db = load_components()["db"]
    analytics = db.get_analytics()
    top_skills = db.get_top_skills(15)

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analytics['total_analyses']}</div>
            <div class="metric-label">Global Talent Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analytics['avg_score']}%</div>
            <div class="metric-label">Average Match Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(top_skills)}</div>
            <div class="metric-label">Unique Expertise Areas</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📈 Top Trending Skills")
        if top_skills:
            df_skills = pd.DataFrame(top_skills)
            import plotly.express as px
            fig = px.bar(
                df_skills,
                x="frequency",
                y="skill_name",
                orientation="h",
                color="frequency",
                labels={"frequency": "Usage Count", "skill_name": "Skill Name"},
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No skill data available yet. Start analyzing resumes!")

    with col2:
        st.markdown("##### 🏆 Popular Career Paths")
        top_roles = analytics.get("top_roles", [])
        if top_roles:
            df_roles = pd.DataFrame(top_roles)
            import plotly.express as px
            fig = px.pie(
                df_roles,
                values="count",
                names="top_role",
                hole=0.4,
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Tealgrn_r
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                height=450,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No role data available yet.")

    # Explanation
    st.markdown("""
    <div class="glass-card" style="border-color: rgba(108, 99, 255, 0.2); border-left: 4px solid #6c63ff;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #FFF; margin-bottom: 8px;">ℹ️ About Market Insights</div>
        <div style="font-size: 0.85rem; color: #B0B0B0;">
            This module provides live, real-time analytics by aggregating all processed data in the system's 
            intelligence database. It helps HR managers and candidates understand the <strong>Supply of Skills</strong> 
            and the most common <strong>Target Roles</strong> within the organization or talent pool.
        </div>
    </div>
    """, unsafe_allow_html=True)
