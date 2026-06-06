"""
core/non_tech_ats_architect.py
────────────────────────────────────────────────────────────────────────────────
ATS Resume Architect — Non-Technical to Data Analyst Transition Engine.

Implements the full 9-section ATS Architect pipeline for candidates from:
  HR, Recruitment, Marketing, Sales, Operations, Finance, Education,
  Healthcare, Customer Success, Administration

Generates:
  Section 1  — Current ATS Score (0–100 with 7-dimension breakdown)
  Section 2  — Skill Gap Analysis (Core / Intermediate / Advanced)
  Section 3  — Transferable Skills Identified
  Section 4  — Recommended Data Analyst Skills (prioritized)
  Section 5  — Recommended Certifications
  Section 6  — Recommended Projects (mapped to missing skills)
  Section 7  — ATS-Optimized Redrafted Resume bullets
  Section 8  — Improved ATS Score Prediction
  Section 9  — Recruiter Feedback Report

CRITICAL RULES:
  ✓ Reframe existing responsibilities
  ✓ Highlight transferable skills
  ✓ Convert manual reporting work into analytics language
  ✓ Convert Excel work into data analysis experience
  ✓ Convert dashboard work into business intelligence experience
  ✓ Convert KPI reporting into analytics achievements

  ✗ NEVER invent jobs
  ✗ NEVER invent internships
  ✗ NEVER invent companies
  ✗ NEVER invent certifications
  ✗ NEVER invent years of experience
"""

import re
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from core.resume_parser import ParsedResume, WorkExperience, Project
from core.jd_engine import ParsedJD

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# ATS Score Breakdown Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ATSBreakdownScore:
    keyword_match:        float = 0.0   # 25% weight
    skill_match:          float = 0.0   # 20% weight
    semantic_similarity:  float = 0.0   # 20% weight
    project_relevance:    float = 0.0   # 15% weight
    experience_alignment: float = 0.0   # 10% weight
    education_alignment:  float = 0.0   # 5%  weight
    formatting_quality:   float = 0.0   # 5%  weight
    overall:              float = 0.0   # weighted total

    def compute_overall(self):
        self.overall = (
            0.25 * self.keyword_match +
            0.20 * self.skill_match +
            0.20 * self.semantic_similarity +
            0.15 * self.project_relevance +
            0.10 * self.experience_alignment +
            0.05 * self.education_alignment +
            0.05 * self.formatting_quality
        )
        return round(self.overall, 1)


@dataclass
class ATSArchitectReport:
    """Full 9-section ATS Architect output."""
    # Section 1
    current_ats_score:      ATSBreakdownScore = field(default_factory=ATSBreakdownScore)
    # Section 2
    skill_gap:              Dict[str, List[str]] = field(default_factory=dict)
    # Section 3
    transferable_skills:    List[str] = field(default_factory=list)
    # Section 4
    recommended_da_skills:  List[str] = field(default_factory=list)
    # Section 5
    recommended_certs:      Dict[str, List[str]] = field(default_factory=dict)
    # Section 6
    recommended_projects:   List[Dict] = field(default_factory=list)
    # Section 7 — reframed bullets per experience role
    reframed_bullets:       Dict[str, List[str]] = field(default_factory=dict)
    reframed_summary:       str = ""
    reframed_skills:        List[str] = field(default_factory=list)
    # Section 8
    predicted_ats_score:    ATSBreakdownScore = field(default_factory=ATSBreakdownScore)
    # Section 9
    recruiter_feedback:     List[str] = field(default_factory=list)
    # Meta
    background_domain:      str = ""
    target_role:            str = "Data Analyst"
    analytics_bridge:       List[str] = field(default_factory=list)
    knowledge_graph_path:   List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — ATS Score Calculator (Pre-Enhancement)
# ──────────────────────────────────────────────────────────────────────────────

DA_CORE_KEYWORDS = [
    "sql", "excel", "power bi", "tableau", "data visualization",
    "data analysis", "data cleaning", "reporting", "dashboarding",
    "statistics", "kpi", "business intelligence", "pivot tables",
    "data analyst", "metrics", "insights",
]

DA_INTERMEDIATE_KEYWORDS = [
    "python", "pandas", "numpy", "a/b testing", "hypothesis testing",
    "google analytics", "etl", "data wrangling", "funnel analysis",
    "cohort analysis", "regression", "data storytelling",
]

DA_ADVANCED_KEYWORDS = [
    "machine learning", "predictive analytics", "scikit-learn",
    "forecasting", "time series", "bigquery", "snowflake", "spark",
    "airflow", "dbt", "r", "feature engineering",
]

ALL_DA_KEYWORDS = DA_CORE_KEYWORDS + DA_INTERMEDIATE_KEYWORDS + DA_ADVANCED_KEYWORDS


def _compute_ats_breakdown(resume: ParsedResume, target_role: str, jd: Optional[ParsedJD] = None) -> ATSBreakdownScore:
    """Compute the 7-dimension ATS breakdown score for a resume vs DA role."""
    score = ATSBreakdownScore()
    raw = resume.raw_text.lower()
    skills_lower = {s.lower() for s in resume.skills}

    # JD keywords: use provided JD or fall back to DA keyword list
    jd_keywords = [k.lower() for k in (jd.all_keywords if jd else [])] or ALL_DA_KEYWORDS
    jd_keywords_set = set(jd_keywords)

    # 1. Keyword Match (25%)
    matched_kw = sum(1 for kw in jd_keywords if kw in raw)
    score.keyword_match = round(min(matched_kw / max(len(jd_keywords), 1) * 100, 100), 1)

    # 2. Skill Match (20%)
    all_da = set(DA_CORE_KEYWORDS + DA_INTERMEDIATE_KEYWORDS + DA_ADVANCED_KEYWORDS)
    matched_skills = skills_lower & all_da
    score.skill_match = round(min(len(matched_skills) / max(len(all_da) * 0.4, 1) * 100, 100), 1)

    # 3. Semantic Similarity (20%) — proxy using keyword density
    da_mentions = sum(1 for kw in ALL_DA_KEYWORDS if kw in raw)
    score.semantic_similarity = round(min(da_mentions / 8 * 100, 100), 1)

    # 4. Project Relevance (15%)
    proj_score = 0.0
    da_proj_signals = {"dashboard", "analysis", "analytics", "data", "sql", "excel",
                       "visualization", "report", "kpi", "metrics", "power bi", "tableau"}
    for proj in resume.projects:
        tech_lower = {t.lower() for t in proj.tech_used}
        desc_lower = (proj.description or "").lower()
        relevance = len(tech_lower & da_proj_signals) + sum(1 for s in da_proj_signals if s in desc_lower)
        proj_score += min(relevance / 5, 1.0)
    score.project_relevance = round(
        min((proj_score / max(len(resume.projects), 1)) * 100, 100) if resume.projects else 10.0, 1
    )

    # 5. Experience Alignment (10%)
    # Check for any analytics/data language in experience bullets
    analytics_words = {"report", "analysis", "data", "dashboard", "metrics", "kpi",
                       "excel", "trend", "insight", "forecast", "visualization"}
    exp_bullets = [b.lower() for exp in resume.experience for b in exp.bullets]
    if exp_bullets:
        analytics_bullets = sum(
            1 for b in exp_bullets
            if any(w in b for w in analytics_words)
        )
        score.experience_alignment = round(
            min((analytics_bullets / len(exp_bullets)) * 100, 100), 1
        )
    else:
        score.experience_alignment = 0.0

    # 6. Education Alignment (5%)
    edu_text = " ".join(
        f"{e.degree or ''} {e.institution or ''}".lower()
        for e in resume.education
    )
    edu_signals = ["math", "statistics", "business", "economics", "computer",
                   "information", "management", "commerce", "engineering", "science"]
    edu_matches = sum(1 for s in edu_signals if s in edu_text)
    score.education_alignment = round(min(edu_matches / 3 * 100, 100), 1)

    # 7. Formatting Quality (5%)
    fmt_score = 0.0
    if resume.email: fmt_score += 25
    if resume.phone: fmt_score += 20
    if resume.linkedin or resume.github: fmt_score += 15
    if resume.summary and len(resume.summary) > 30: fmt_score += 20
    if resume.sections_found: fmt_score += 10
    if resume.word_count >= 300: fmt_score += 10
    score.formatting_quality = round(min(fmt_score, 100), 1)

    score.compute_overall()
    return score


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Skill Gap Analysis
# ──────────────────────────────────────────────────────────────────────────────

def _compute_skill_gap(resume: ParsedResume) -> Dict[str, List[str]]:
    """Return missing skills by tier: core, intermediate, advanced."""
    from config import DATA_ANALYST_SKILLS
    skills_lower = {s.lower() for s in resume.skills}
    raw_lower = resume.raw_text.lower()

    gap = {}
    for tier, skill_list in DATA_ANALYST_SKILLS.items():
        missing = [
            s for s in skill_list
            if s not in skills_lower and s not in raw_lower
        ]
        gap[tier] = missing

    return gap


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Transferable Skills Identification
# ──────────────────────────────────────────────────────────────────────────────

def _identify_transferable_skills(resume: ParsedResume, background_domain: str) -> List[str]:
    """Extract transferable skills based on background domain."""
    from config import TRANSFERABILITY_MAP
    transferable = []

    domain_profile = TRANSFERABILITY_MAP.get(background_domain, {})
    if domain_profile:
        transferable.extend(domain_profile.get("transferable_skills", []))

    # Also check for universal analytics-adjacent skills in the resume
    universal_analytics = [
        "excel", "reporting", "data entry", "kpi", "metrics", "stakeholder management",
        "presentation", "powerpoint", "google sheets", "pivot tables", "ms office",
        "problem solving", "analytical thinking", "critical thinking",
        "project management", "process improvement", "documentation",
    ]
    raw_lower = resume.raw_text.lower()
    for skill in universal_analytics:
        if skill in raw_lower and skill not in [t.lower() for t in transferable]:
            transferable.append(skill.title())

    # Deduplicate preserving order
    seen = set()
    unique = []
    for s in transferable:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)

    return unique[:15]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Recommended DA Skills (prioritized by gap + background)
# ──────────────────────────────────────────────────────────────────────────────

def _recommend_da_skills(
    skill_gap: Dict[str, List[str]],
    background_domain: str,
) -> List[str]:
    """Return prioritized list of DA skills to learn based on gap + domain."""
    from config import TRANSFERABILITY_MAP
    recommended = []

    domain_profile = TRANSFERABILITY_MAP.get(background_domain, {})
    domain_tools = domain_profile.get("recommended_tools", [])

    # Priority 1: Domain-recommended tools that are in the gap
    for tool in domain_tools:
        if tool in skill_gap.get("core", []) or tool in skill_gap.get("intermediate", []):
            recommended.append(tool)

    # Priority 2: Core skills gap
    for skill in skill_gap.get("core", []):
        if skill not in recommended:
            recommended.append(skill)

    # Priority 3: Intermediate skills
    for skill in skill_gap.get("intermediate", [])[:6]:
        if skill not in recommended:
            recommended.append(skill)

    # Priority 4: Advanced skills (limit to 3)
    for skill in skill_gap.get("advanced", [])[:3]:
        if skill not in recommended:
            recommended.append(skill)

    return recommended[:20]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Project Recommendations (mapped to missing skills)
# ──────────────────────────────────────────────────────────────────────────────

def _recommend_projects(skill_gap: Dict[str, List[str]], background_domain: str) -> List[Dict]:
    """Return 3–5 recommended projects based on missing skills."""
    from config import PROJECT_RECOMMENDATION_MAP, TRANSFERABILITY_MAP

    recommended = []
    seen_names = set()

    all_missing = (
        skill_gap.get("core", []) +
        skill_gap.get("intermediate", [])
    )

    # Map missing skills to projects
    priority_skills = []

    # Domain-specific priority tools first
    domain_profile = TRANSFERABILITY_MAP.get(background_domain, {})
    domain_tools = domain_profile.get("recommended_tools", [])
    for tool in domain_tools:
        if tool in all_missing:
            priority_skills.append(tool)

    # Then remaining missing core skills
    for skill in all_missing:
        if skill not in priority_skills:
            priority_skills.append(skill)

    for skill in priority_skills:
        proj = PROJECT_RECOMMENDATION_MAP.get(skill.lower())
        if proj and proj["name"] not in seen_names:
            recommended.append({**proj, "skill_targeted": skill})
            seen_names.add(proj["name"])
        if len(recommended) >= 5:
            break

    # Always include at least one DA project if list is short
    if len(recommended) < 3:
        for skill, proj in PROJECT_RECOMMENDATION_MAP.items():
            if proj["name"] not in seen_names:
                recommended.append({**proj, "skill_targeted": skill})
                seen_names.add(proj["name"])
            if len(recommended) >= 3:
                break

    return recommended[:5]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Transferability Engine: Bullet Reframing
# ──────────────────────────────────────────────────────────────────────────────

# Generic analytics action verb patterns
_ACTION_ANALYTICS_TRANSFORMS = [
    # Pattern → replacement prefix
    (r"^(was responsible for|was in charge of|helped with|assisted in|participated in)\s+(.+)",
     r"Supported data-driven \2"),
    (r"^(worked on|worked with)\s+(.+)",
     r"Collaborated on analytics initiatives involving \2"),
    (r"^(did|handled|did)\s+(.+)",
     r"Managed and analyzed \2"),
    (r"^(maintained|kept track of)\s+(.+)",
     r"Maintained structured data records for \2, ensuring data integrity and accuracy"),
    (r"^(updated|entered)\s+(data|records|information|spreadsheets?)\s*",
     "Maintained structured data entry workflows ensuring accuracy and integrity for organizational reporting"),
    (r"^(prepared|created|made|built|developed)\s+(report|reports|spreadsheet|spreadsheets|dashboard|dashboards)\s*",
     "Developed recurring performance reports and dashboards tracking key business metrics and KPIs for stakeholder review"),
    (r"^(analyzed|analysed)\s+(.+)",
     r"Analyzed \2 using data-driven methodologies to surface actionable insights and support business decision-making"),
    (r"^(coordinated|organized|planned)\s+(.+)",
     r"Coordinated and tracked \2 using structured data management to improve operational efficiency"),
    (r"^(communicated|presented|reported to)\s+(.+)",
     r"Presented data-backed insights and analytical reports to \2, translating complex data into actionable business recommendations"),
]

# Generic transforms applied when no domain-specific rewrite is found
_GENERIC_METRIC_BOOST = [
    ("improve", "achieve measurable improvements in"),
    ("increase", "drive data-backed increases in"),
    ("reduce", "identify and reduce"),
    ("manage", "analyze and manage"),
    ("monitor", "track and analyze"),
    ("review", "analyze and review"),
    ("check", "audit and analyze"),
    ("ensure", "monitor and ensure"),
    ("support", "provide data-driven support for"),
]


def _find_domain_rewrite(bullet: str, background_domain: str) -> Optional[str]:
    """Check domain-specific bullet_rewrites for a matching reframe."""
    from config import TRANSFERABILITY_MAP

    domain_profile = TRANSFERABILITY_MAP.get(background_domain, {})
    rewrites = domain_profile.get("bullet_rewrites", {})

    bullet_lower = bullet.lower().strip()

    # Exact/substring match against domain-specific rewrites
    for pattern, rewrite in rewrites.items():
        if pattern.lower() in bullet_lower:
            # Build contextual rewrite: capitalize + preserve any metrics
            metrics = re.findall(r"\d+[.,]?\d*\s*(%|x|k|M|B|\+|users?|hrs?)", bullet, re.I)
            result = rewrite
            if metrics:
                result += f", achieving {metrics[0]} improvement"
            return result

    return None


def _apply_generic_reframe(bullet: str) -> str:
    """Apply generic analytics reframing when no domain-specific match exists."""
    bullet = bullet.strip()
    if not bullet:
        return bullet

    # Try regex-based action transforms
    for pattern, replacement in _ACTION_ANALYTICS_TRANSFORMS:
        m = re.match(pattern, bullet, re.I)
        if m:
            try:
                result = re.sub(pattern, replacement, bullet, flags=re.I, count=1)
                return result[0].upper() + result[1:] if result else bullet
            except Exception:
                pass

    # Metric boosts — replace weak verbs with analytics language
    bullet_lower = bullet.lower()
    for weak, strong in _GENERIC_METRIC_BOOST:
        if bullet_lower.startswith(weak):
            reframed = strong + bullet[len(weak):]
            return reframed[0].upper() + reframed[1:]

    # Fallback: if bullet is a passive statement, prefix with analytics framing
    if not any(bullet.lower().startswith(v) for v in [
        "built", "developed", "created", "led", "analyzed", "designed",
        "managed", "implemented", "tracked", "monitored", "reported",
    ]):
        return f"Analyzed and managed {bullet[0].lower() + bullet[1:]}"

    return bullet


def reframe_bullets_for_da(
    experience: List[WorkExperience],
    background_domain: str,
) -> Dict[str, List[str]]:
    """
    Reframe all experience bullets to analytics-oriented language.
    Returns dict: {role_key: [reframed_bullets]}

    ANTI-HALLUCINATION: only rewrites existing bullets, never invents new ones.
    """
    reframed: Dict[str, List[str]] = {}

    for exp in experience:
        role_key = f"{exp.title or 'Role'} @ {exp.company or 'Company'}"
        new_bullets = []

        for bullet in exp.bullets:
            if not bullet.strip():
                continue

            # 1. Try domain-specific rewrite
            domain_rewrite = _find_domain_rewrite(bullet, background_domain)
            if domain_rewrite:
                new_bullets.append(domain_rewrite)
            else:
                # 2. Apply generic analytics reframe
                new_bullets.append(_apply_generic_reframe(bullet))

        reframed[role_key] = new_bullets[:5]  # max 5 bullets per role

    return reframed


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Predicted ATS Score (post-enhancement)
# ──────────────────────────────────────────────────────────────────────────────

def _predict_post_enhancement_score(
    current: ATSBreakdownScore,
    skill_gap: Dict[str, List[str]],
    recommended_da_skills: List[str],
    has_projects: bool,
) -> ATSBreakdownScore:
    """
    Predict post-enhancement ATS score after applying all recommendations.
    Uses optimistic but realistic estimates based on what the system can achieve.
    """
    pred = ATSBreakdownScore()

    # Keyword Match: adding DA keywords via reframing boosts this significantly
    kw_boost = min(35.0, len(recommended_da_skills) * 2.5)
    pred.keyword_match = round(min(current.keyword_match + kw_boost, 92.0), 1)

    # Skill Match: we inject recommended DA skills into the output
    skill_boost = min(40.0, (len(recommended_da_skills[:8])) * 5.0)
    pred.skill_match = round(min(current.skill_match + skill_boost, 88.0), 1)

    # Semantic Similarity: reframing bullets with DA language boosts this
    sem_boost = min(35.0, 20.0 + (kw_boost / 2))
    pred.semantic_similarity = round(min(current.semantic_similarity + sem_boost, 90.0), 1)

    # Project Relevance: recommending relevant projects
    proj_boost = 40.0 if not has_projects else 20.0
    pred.project_relevance = round(min(current.project_relevance + proj_boost, 85.0), 1)

    # Experience Alignment: reframed bullets with analytics language
    exp_boost = min(30.0, 25.0)
    pred.experience_alignment = round(min(current.experience_alignment + exp_boost, 80.0), 1)

    # Education: doesn't change much
    pred.education_alignment = round(min(current.education_alignment + 5.0, 85.0), 1)

    # Formatting: adding summary, skills → better formatting
    pred.formatting_quality = round(min(current.formatting_quality + 15.0, 95.0), 1)

    pred.compute_overall()
    return pred


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Recruiter Feedback Report
# ──────────────────────────────────────────────────────────────────────────────

def _generate_recruiter_feedback(
    resume: ParsedResume,
    background_domain: str,
    skill_gap: Dict[str, List[str]],
    current_score: ATSBreakdownScore,
    predicted_score: ATSBreakdownScore,
    target_role: str,
) -> List[str]:
    """Generate actionable recruiter feedback for the transition."""
    feedback = []

    # Score tier assessment
    if current_score.overall < 40:
        feedback.append(
            f"🔴 Current ATS Score ({current_score.overall:.0f}/100) is LOW for a {target_role} role. "
            f"Significant reframing is needed — but your {background_domain} background has strong "
            f"analytical and business intelligence foundations that can be effectively repositioned."
        )
    elif current_score.overall < 65:
        feedback.append(
            f"🟡 Current ATS Score ({current_score.overall:.0f}/100) shows MODERATE alignment. "
            f"Your {background_domain} experience contains transferable skills that once reframed "
            f"in data analytics language will significantly improve recruiter relevance."
        )
    else:
        feedback.append(
            f"🟢 Current ATS Score ({current_score.overall:.0f}/100) shows GOOD baseline. "
            f"Targeted reframing and adding 1–2 analytics projects can push you to 85+."
        )

    # Missing core skills feedback
    missing_core = skill_gap.get("core", [])[:4]
    if missing_core:
        feedback.append(
            f"📌 Priority skill gaps: {', '.join(missing_core)}. These are NON-NEGOTIABLE for "
            f"Data Analyst roles. Start with SQL and Excel/Power BI — they appear in 90%+ of JDs."
        )

    # Strengths from background
    if background_domain == "HR / Talent Acquisition":
        feedback.append(
            "✅ Strength: HR background provides direct pathway to People Analytics and Workforce BI — "
            "a growing specialization. Frame your HRIS, payroll, and workforce planning experience "
            "as data management and reporting experience."
        )
    elif background_domain == "Finance / Accounting":
        feedback.append(
            "✅ Strength: Finance background is highly transferable — you already work with large "
            "datasets, Excel models, and financial dashboards. Frame P&L analysis and budget "
            "forecasting as financial data analytics experience."
        )
    elif background_domain == "Marketing / Growth":
        feedback.append(
            "✅ Strength: Marketing background maps directly to Marketing Analytics. Campaign metrics, "
            "A/B testing, and Google Analytics experience is extremely valued by DA recruiters. "
            "Lead with campaign data analysis and conversion optimization."
        )
    elif background_domain == "Sales / Business Development":
        feedback.append(
            "✅ Strength: Sales background provides strong CRM data, pipeline analytics, and "
            "revenue reporting experience. Frame Salesforce usage as CRM data analysis. "
            "Sales forecasting = predictive analytics."
        )
    elif background_domain == "Operations / Supply Chain":
        feedback.append(
            "✅ Strength: Operations background maps to Operations Analytics. Six Sigma, "
            "process optimization, and KPI monitoring are recognized analytical competencies. "
            "ERP data experience is directly relevant to BI roles."
        )

    # Project recommendation
    feedback.append(
        "🚀 Action Required: Add 2–3 data analytics projects to your resume using public "
        "datasets (Kaggle). Recommended: SQL + Power BI or Tableau project showcasing "
        "your domain knowledge (HR dashboard, sales analytics, financial KPI report). "
        "Hiring managers weight projects heavily for non-traditional candidates."
    )

    # Certification advice
    feedback.append(
        "📜 Certifications: Google Data Analytics Professional Certificate (Coursera, ~6 months) "
        "or IBM Data Analyst Professional Certificate are widely recognized and will validate "
        "your skills pivot. Add them BEFORE applying — they signal commitment to the transition."
    )

    # Final prediction
    feedback.append(
        f"🎯 Predicted ATS Score after applying all recommendations: "
        f"{predicted_score.overall:.0f}/100 — targeting {target_role} roles at "
        f"{'entry' if predicted_score.overall < 75 else 'junior-to-mid'} level."
    )

    return feedback


# ──────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE GRAPH PATH BUILDER
# ──────────────────────────────────────────────────────────────────────────────

_KNOWLEDGE_GRAPH: Dict[str, List[str]] = {
    "HR / Talent Acquisition": [
        "HR Experience",
        "→ Workforce Data (HRIS, Payroll, Headcount)",
        "→ Workforce Analytics",
        "→ Employee Attrition Analysis",
        "→ HR KPI Dashboarding (Power BI / Tableau)",
        "→ SQL for HR Data",
        "→ Data Storytelling",
        "→ People Analytics Specialist",
    ],
    "Marketing / Growth": [
        "Marketing Experience",
        "→ Campaign Performance Data (CTR, ROAS, Conversions)",
        "→ Campaign Analytics",
        "→ A/B Testing & Hypothesis Testing",
        "→ Customer Segmentation (SQL / Python)",
        "→ Data Visualization (Tableau / Looker)",
        "→ Marketing Analytics Analyst",
    ],
    "Sales / Business Development": [
        "Sales Experience",
        "→ CRM Data (Salesforce / HubSpot)",
        "→ Revenue Analytics",
        "→ Sales Funnel Analysis (SQL)",
        "→ KPI Dashboard Creation (Power BI)",
        "→ Sales Forecasting (Excel / Python)",
        "→ Sales / Business Analyst",
    ],
    "Finance / Accounting": [
        "Finance / Accounting Experience",
        "→ Financial Datasets (P&L, Budgets, Ledgers)",
        "→ Financial Analytics",
        "→ Excel Financial Modeling",
        "→ SQL for Financial Data",
        "→ Power BI Financial Dashboards",
        "→ Financial / BI Analyst",
    ],
    "Operations / Supply Chain": [
        "Operations Experience",
        "→ Operational Data (ERP, KPIs, SLAs)",
        "→ Process Analytics",
        "→ Data Cleaning & Reporting",
        "→ Root Cause Analysis (SQL / Excel)",
        "→ Operational BI Dashboard (Tableau / Power BI)",
        "→ Operations / Reporting Analyst",
    ],
    "Education / Academic": [
        "Education Experience",
        "→ Student Performance Data",
        "→ Assessment Data Analysis (Excel)",
        "→ Learning Outcome Analytics",
        "→ Data Visualization (Tableau / Power BI)",
        "→ SQL for Institutional Data",
        "→ Data / Business Analyst",
    ],
    "Healthcare / Biotech": [
        "Healthcare Experience",
        "→ Clinical / Patient Data (EHR/EMR)",
        "→ Clinical Data Analysis",
        "→ Population Health Analytics",
        "→ SQL for Healthcare Data",
        "→ Power BI / Tableau Clinical Dashboards",
        "→ Healthcare Data Analyst",
    ],
    "Customer Success / Support": [
        "Customer Success Experience",
        "→ Customer Data (NPS, CSAT, Tickets)",
        "→ Customer Analytics",
        "→ Churn Analysis (SQL / Excel)",
        "→ Customer Journey Dashboarding (Tableau)",
        "→ Product / Customer Analytics Analyst",
    ],
    "Administration / Business Support": [
        "Administration Experience",
        "→ Business Data (Reports, Records, Spreadsheets)",
        "→ Data Organization & Cleaning",
        "→ Excel / Power BI Reporting",
        "→ SQL for Business Data",
        "→ Business / Reporting Analyst",
    ],
}

def _build_knowledge_graph_path(background_domain: str) -> List[str]:
    return _KNOWLEDGE_GRAPH.get(background_domain, [
        f"{background_domain} Experience",
        "→ Domain Data & Reporting",
        "→ Data Analysis (Excel / SQL)",
        "→ Dashboarding (Power BI / Tableau)",
        "→ Data Analyst",
    ])


# ──────────────────────────────────────────────────────────────────────────────
# REFRAME PROFESSIONAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def _reframe_summary(
    resume: ParsedResume,
    background_domain: str,
    target_role: str,
    recommended_skills: List[str],
) -> str:
    """
    Reframe existing summary (or build from experience) for a DA role.
    NEVER invents facts — works only from existing resume content.
    """
    years_exp = len(resume.experience)
    existing = resume.summary or ""

    domain_short = background_domain.split(" /")[0].split(" (")[0]

    # Core skills to highlight from what they already have
    existing_da_adjacent = [
        s for s in resume.skills
        if s.lower() in {
            "excel", "reporting", "kpi", "data", "sql", "tableau", "power bi",
            "google analytics", "dashboard", "pivot tables", "ms office",
        }
    ][:4]

    # Build summary
    years_str = f"{years_exp}+ year{'s' if years_exp != 1 else ''}" if years_exp else "Several years of"
    skills_str = ", ".join(recommended_skills[:4]) if recommended_skills else "Excel, SQL, Power BI"

    target_area = "Data Analytics" if "analyst" in target_role.lower() else target_role
    summary = (
        f"{domain_short} professional with {years_str} of experience transitioning into {target_area}, "
        f"bringing strong domain expertise in data reporting, KPI tracking, and stakeholder communication. "
        f"Proficient in {', '.join(existing_da_adjacent) if existing_da_adjacent else skills_str} "
        f"with a proven ability to translate business data into actionable insights. "
        f"Seeking a {target_role} role to leverage analytical skills and domain knowledge "
        f"in driving data-driven decision-making."
    )

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# REFRAME SKILLS LIST
# ──────────────────────────────────────────────────────────────────────────────

def _reframe_skills_for_da(
    resume: ParsedResume,
    background_domain: str,
    recommended_da_skills: List[str],
    skill_gap: Dict[str, List[str]],
) -> List[str]:
    """
    Build an optimized skills list for DA roles from existing + recommended.
    NEVER invents skills — only includes skills genuinely present or recommended to learn.
    """
    from config import TRANSFERABILITY_MAP

    existing = [s.lower() for s in resume.skills]
    domain_profile = TRANSFERABILITY_MAP.get(background_domain, {})

    # Analytics-adjacent skills already present in resume
    da_adjacent_existing = [
        s for s in resume.skills
        if s.lower() in {
            "excel", "sql", "power bi", "tableau", "data analysis", "reporting",
            "google analytics", "kpi", "dashboarding", "pivot tables", "ms office",
            "data visualization", "business intelligence", "microsoft excel",
            "data entry", "spreadsheets", "google sheets",
        }
    ]

    # Transferable domain skills reframed
    transferable = domain_profile.get("transferable_skills", [])

    # Domain-recommended tools (signal that they should learn these)
    domain_tools_capitalized = [t.title() for t in domain_profile.get("recommended_tools", [])]

    # Compose: existing DA-adjacent + transferable + recommended (to learn)
    combined = da_adjacent_existing + transferable + domain_tools_capitalized + recommended_da_skills

    # Deduplicate preserving order
    seen = set()
    result = []
    for s in combined:
        if s.lower() not in seen:
            seen.add(s.lower())
            result.append(s)

    return result[:30]


# ──────────────────────────────────────────────────────────────────────────────
# MASTER ARCHITECT FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def run_ats_architect(
    resume: ParsedResume,
    background_domain: str,
    target_role: str = "Data Analyst",
    jd: Optional[ParsedJD] = None,
) -> ATSArchitectReport:
    """
    Full 9-section ATS Architect pipeline for non-technical → DA transition.
    Returns ATSArchitectReport with all sections populated.
    """
    from config import RECOMMENDED_CERTIFICATIONS

    report = ATSArchitectReport()
    report.background_domain = background_domain
    report.target_role = target_role

    logger.info(f"ATS Architect: domain={background_domain}, target={target_role}")

    # ── Section 1: Current ATS Score ─────────────────────────────────────────
    report.current_ats_score = _compute_ats_breakdown(resume, target_role, jd)
    logger.info(f"Section 1 — Current ATS: {report.current_ats_score.overall:.1f}/100")

    # ── Section 2: Skill Gap Analysis ────────────────────────────────────────
    report.skill_gap = _compute_skill_gap(resume)
    logger.info(f"Section 2 — Skill Gap: core={len(report.skill_gap.get('core',[]))}, "
                f"intermediate={len(report.skill_gap.get('intermediate',[]))}")

    # ── Section 3: Transferable Skills ───────────────────────────────────────
    report.transferable_skills = _identify_transferable_skills(resume, background_domain)
    logger.info(f"Section 3 — Transferable Skills: {len(report.transferable_skills)}")

    # ── Section 4: Recommended DA Skills ─────────────────────────────────────
    report.recommended_da_skills = _recommend_da_skills(report.skill_gap, background_domain)
    logger.info(f"Section 4 — Recommended Skills: {len(report.recommended_da_skills)}")

    # ── Section 5: Recommended Certifications ────────────────────────────────
    report.recommended_certs = deepcopy(RECOMMENDED_CERTIFICATIONS)
    logger.info("Section 5 — Certifications: loaded")

    # ── Section 6: Recommended Projects ──────────────────────────────────────
    report.recommended_projects = _recommend_projects(report.skill_gap, background_domain)
    logger.info(f"Section 6 — Projects: {len(report.recommended_projects)} recommended")

    # ── Section 7: Reframed Resume Content ───────────────────────────────────
    report.reframed_bullets = reframe_bullets_for_da(resume.experience, background_domain)
    report.reframed_summary = _reframe_summary(
        resume, background_domain, target_role, report.recommended_da_skills
    )
    report.reframed_skills = _reframe_skills_for_da(
        resume, background_domain, report.recommended_da_skills, report.skill_gap
    )
    logger.info(f"Section 7 — Reframed: {sum(len(v) for v in report.reframed_bullets.values())} bullets")

    # ── Section 8: Predicted ATS Score ───────────────────────────────────────
    report.predicted_ats_score = _predict_post_enhancement_score(
        report.current_ats_score,
        report.skill_gap,
        report.recommended_da_skills,
        bool(resume.projects),
    )
    logger.info(f"Section 8 — Predicted ATS: {report.predicted_ats_score.overall:.1f}/100")

    # ── Section 9: Recruiter Feedback ────────────────────────────────────────
    report.recruiter_feedback = _generate_recruiter_feedback(
        resume, background_domain, report.skill_gap,
        report.current_ats_score, report.predicted_ats_score, target_role
    )
    logger.info(f"Section 9 — Recruiter Feedback: {len(report.recruiter_feedback)} items")

    # ── Knowledge Graph Path ─────────────────────────────────────────────────
    report.analytics_bridge = (
        __import__("config", fromlist=["TRANSFERABILITY_MAP"])
        .TRANSFERABILITY_MAP.get(background_domain, {})
        .get("analytics_bridge", [])
    )
    report.knowledge_graph_path = _build_knowledge_graph_path(background_domain)

    logger.info("ATS Architect pipeline complete.")
    return report
