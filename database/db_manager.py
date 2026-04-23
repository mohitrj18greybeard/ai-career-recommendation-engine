"""
Database Manager — SQLite operations for storing analysis history.
PostgreSQL-compatible schema for easy migration.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


class DatabaseManager:
    """SQLite database manager for resume analysis history."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()

    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_name TEXT,
                candidate_email TEXT,
                resume_text TEXT,
                skills_json TEXT,
                recommendations_json TEXT,
                gap_analysis_json TEXT,
                improvement_json TEXT,
                overall_score REAL,
                top_role TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS skills_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_name TEXT UNIQUE,
                category TEXT,
                frequency INTEGER DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_analyses_created
                ON analyses(created_at);
            CREATE INDEX IF NOT EXISTS idx_analyses_role
                ON analyses(top_role);
        """)

        conn.commit()
        conn.close()

    def save_analysis(
        self,
        candidate_name: str,
        candidate_email: str,
        resume_text: str,
        skills: List[Dict],
        recommendations: List[Dict],
        gap_analysis: Dict,
        improvement: Dict,
        overall_score: float,
        top_role: str,
    ) -> int:
        """Save a complete analysis to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO analyses
                (candidate_name, candidate_email, resume_text, skills_json,
                 recommendations_json, gap_analysis_json, improvement_json,
                 overall_score, top_role)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate_name,
            candidate_email,
            resume_text[:5000],  # Truncate for storage
            json.dumps(skills, default=str),
            json.dumps(recommendations, default=str),
            json.dumps(gap_analysis, default=str),
            json.dumps(improvement, default=str),
            overall_score,
            top_role,
        ))

        analysis_id = cursor.lastrowid

        # Update skills catalog
        for skill in skills:
            cursor.execute("""
                INSERT INTO skills_catalog (skill_name, category, frequency)
                VALUES (?, ?, 1)
                ON CONFLICT(skill_name) DO UPDATE SET
                    frequency = frequency + 1,
                    updated_at = CURRENT_TIMESTAMP
            """, (skill.get("name", ""), skill.get("category", "")))

        conn.commit()
        conn.close()
        return analysis_id

    def get_recent_analyses(self, limit: int = 10) -> List[Dict]:
        """Get recent analysis history."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, candidate_name, candidate_email, overall_score,
                   top_role, created_at
            FROM analyses
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "candidate_name": row["candidate_name"],
                "candidate_email": row["candidate_email"],
                "overall_score": row["overall_score"],
                "top_role": row["top_role"],
                "created_at": row["created_at"],
            })

        conn.close()
        return results

    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict]:
        """Get a specific analysis by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()

        if row:
            result = dict(row)
            result["skills"] = json.loads(result.pop("skills_json", "[]"))
            result["recommendations"] = json.loads(result.pop("recommendations_json", "[]"))
            result["gap_analysis"] = json.loads(result.pop("gap_analysis_json", "{}"))
            result["improvement"] = json.loads(result.pop("improvement_json", "{}"))
            conn.close()
            return result

        conn.close()
        return None

    def get_top_skills(self, limit: int = 20) -> List[Dict]:
        """Get most frequently seen skills across all analyses."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT skill_name, category, frequency
            FROM skills_catalog
            ORDER BY frequency DESC
            LIMIT ?
        """, (limit,))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_analytics(self) -> Dict:
        """Get overall analytics for the insights dashboard."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total analyses
        cursor.execute("SELECT COUNT(*) as count FROM analyses")
        total = cursor.fetchone()["count"]

        # Average score
        cursor.execute("SELECT AVG(overall_score) as avg_score FROM analyses")
        avg_row = cursor.fetchone()
        avg_score = avg_row["avg_score"] if avg_row["avg_score"] else 0

        # Top roles
        cursor.execute("""
            SELECT top_role, COUNT(*) as count
            FROM analyses
            WHERE top_role IS NOT NULL
            GROUP BY top_role
            ORDER BY count DESC
            LIMIT 5
        """)
        top_roles = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "total_analyses": total,
            "avg_score": round(avg_score, 1),
            "top_roles": top_roles,
        }
