"""
Visualizer — Plotly-based interactive chart utilities for the dashboard.
Creates premium charts: radar, gauge, heatmap, bar, word cloud.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import numpy as np


class Visualizer:
    """Premium chart generator for the Streamlit dashboard."""

    # Color palette
    COLORS = {
        "primary": "#6C63FF",
        "secondary": "#00D2FF",
        "accent": "#FF6B6B",
        "success": "#00E676",
        "warning": "#FFD740",
        "bg": "#0E1117",
        "card": "#1A1E2E",
        "text": "#E0E0E0",
    }

    GRADIENT_COLORS = [
        "#6C63FF", "#7B73FF", "#8A83FF", "#00D2FF", "#00E5FF",
        "#00E676", "#69F0AE", "#FFD740", "#FFC107", "#FF6B6B",
    ]

    def create_skill_radar(self, skills_by_category: Dict[str, int]) -> go.Figure:
        """Create a radar chart showing skill distribution by category."""
        categories = list(skills_by_category.keys())
        values = list(skills_by_category.values())

        if not categories:
            return self._empty_chart("No skills to display")

        # Close the radar
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(108, 99, 255, 0.2)',
            line=dict(color=self.COLORS["primary"], width=2),
            marker=dict(size=8, color=self.COLORS["primary"]),
            name='Skills',
        ))

        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    linecolor='rgba(255,255,255,0.1)',
                    tickfont=dict(color=self.COLORS["text"], size=10),
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    linecolor='rgba(255,255,255,0.1)',
                    tickfont=dict(color=self.COLORS["text"], size=11),
                ),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.COLORS["text"]),
            showlegend=False,
            margin=dict(l=60, r=60, t=30, b=30),
            height=400,
        )

        return fig

    def create_match_gauge(self, score: float, title: str = "Match Score") -> go.Figure:
        """Create a gauge/donut chart for match scores."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 36, "color": self.COLORS["text"]}},
            title={"text": title, "font": {"size": 16, "color": self.COLORS["text"]}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": self.COLORS["text"],
                    "tickfont": {"color": self.COLORS["text"]},
                },
                "bar": {"color": self.COLORS["primary"], "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(255, 68, 68, 0.3)"},
                    {"range": [30, 60], "color": "rgba(255, 215, 64, 0.3)"},
                    {"range": [60, 100], "color": "rgba(0, 230, 118, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": self.COLORS["accent"], "width": 3},
                    "thickness": 0.8,
                    "value": score * 100,
                },
            },
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.COLORS["text"]),
            height=280,
            margin=dict(l=30, r=30, t=50, b=20),
        )

        return fig

    def create_score_breakdown_bar(self, breakdown: Dict[str, float]) -> go.Figure:
        """Create horizontal bar chart for score breakdown."""
        labels = {
            "text_similarity": "Text Similarity",
            "skill_overlap": "Skill Overlap",
            "experience_match": "Experience Match",
        }

        names = [labels.get(k, k) for k in breakdown.keys()]
        values = [v * 100 for v in breakdown.values()]
        colors = [self.COLORS["primary"], self.COLORS["secondary"], self.COLORS["success"]]

        fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker=dict(
                color=colors[:len(values)],
                line=dict(width=0),
                cornerradius=5,
            ),
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
            textfont=dict(color='white', size=12),
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.COLORS["text"]),
            xaxis=dict(
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
                title="Score (%)",
            ),
            yaxis=dict(showgrid=False),
            height=200,
            margin=dict(l=120, r=20, t=10, b=40),
        )

        return fig

    def create_skill_gap_chart(self, matched: int, missing: int) -> go.Figure:
        """Create a donut chart for skill gap visualization."""
        labels = ['Matched Skills', 'Missing Skills']
        values = [matched, missing]
        colors = [self.COLORS["success"], self.COLORS["accent"]]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color=self.COLORS["bg"], width=3)),
            textinfo='label+value',
            textfont=dict(size=12, color=self.COLORS["text"]),
            hovertemplate='%{label}: %{value} skills<extra></extra>',
        )])

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.COLORS["text"]),
            showlegend=True,
            legend=dict(
                font=dict(color=self.COLORS["text"]),
                bgcolor='rgba(0,0,0,0)',
            ),
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            annotations=[dict(
                text=f'{matched}/{matched+missing}',
                x=0.5, y=0.5,
                font_size=24,
                font_color=self.COLORS["text"],
                showarrow=False,
            )],
        )

        return fig

    def create_comparison_bar(self, resume_skills: List[str], job_skills: List[str]) -> go.Figure:
        """Create side-by-side comparison chart for resume vs job skills."""
        # Find all unique skills
        all_skills = list(set(resume_skills + job_skills))[:15]

        resume_has = [1 if s in resume_skills else 0 for s in all_skills]
        job_has = [1 if s in job_skills else 0 for s in all_skills]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=all_skills,
            x=resume_has,
            name='Your Resume',
            orientation='h',
            marker=dict(color=self.COLORS["primary"], cornerradius=3),
        ))

        fig.add_trace(go.Bar(
            y=all_skills,
            x=[-v for v in job_has],
            name='Job Requires',
            orientation='h',
            marker=dict(color=self.COLORS["secondary"], cornerradius=3),
        ))

        fig.update_layout(
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.COLORS["text"]),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.2)',
            ),
            yaxis=dict(showgrid=False),
            legend=dict(
                font=dict(color=self.COLORS["text"]),
                bgcolor='rgba(0,0,0,0)',
                orientation='h',
                yanchor='bottom',
                y=1.02,
            ),
            height=max(300, len(all_skills) * 28),
            margin=dict(l=140, r=20, t=40, b=20),
        )

        return fig

    def create_recommendation_scores(self, recommendations: List[Dict]) -> go.Figure:
        """Create bar chart for recommendation match scores."""
        if not recommendations:
            return self._empty_chart("No recommendations")

        titles = [f"{r['title'][:25]}..." if len(r['title']) > 25 else r['title'] for r in recommendations]
        scores = [r['final_score'] * 100 for r in recommendations]

        # Color gradient based on score
        colors = []
        for s in scores:
            if s >= 70:
                colors.append(self.COLORS["success"])
            elif s >= 45:
                colors.append(self.COLORS["warning"])
            else:
                colors.append(self.COLORS["accent"])

        fig = go.Figure(go.Bar(
            x=scores,
            y=titles,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=5,
            ),
            text=[f'{s:.1f}%' for s in scores],
            textposition='auto',
            textfont=dict(color='white', size=12, family='Arial Black'),
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.COLORS["text"]),
            xaxis=dict(
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.05)',
                title="Match Score (%)",
            ),
            yaxis=dict(showgrid=False, autorange='reversed'),
            height=max(250, len(titles) * 50),
            margin=dict(l=180, r=20, t=10, b=40),
        )

        return fig

    def create_improvement_chart(self, suggestions: List[Dict]) -> go.Figure:
        """Create chart showing improvement suggestion distribution."""
        impact_counts = {"High": 0, "Medium": 0, "Low": 0}
        for s in suggestions:
            impact = s.get("impact", "Low")
            if impact in impact_counts:
                impact_counts[impact] += 1

        fig = go.Figure(data=[go.Pie(
            labels=list(impact_counts.keys()),
            values=list(impact_counts.values()),
            hole=0.5,
            marker=dict(
                colors=[self.COLORS["accent"], self.COLORS["warning"], self.COLORS["success"]],
                line=dict(color=self.COLORS["bg"], width=3),
            ),
            textinfo='label+value',
            textfont=dict(size=12, color=self.COLORS["text"]),
        )])

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.COLORS["text"]),
            showlegend=False,
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        return fig

    def _empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.COLORS["text"]),
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=200,
        )
        return fig
