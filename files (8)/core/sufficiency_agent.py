"""
core/sufficiency_agent.py
────────────────────────────────────────────────────────
Resume Sufficiency Agent.
Evaluates whether a candidate's resume contains enough information
(metrics, action verbs, scope, technology) to build a strong ATS-optimized resume.
Generates custom clarification questions for the interactive questionnaire.

Upgrade: Non-technical background awareness.
When a non-tech CV is detected, generates domain-specific DA transition questions
instead of generic tech-focused ones.
"""

import re
import logging
from typing import List, Dict
from core.resume_parser import ParsedResume
from core.scoring_engine import STRONG_VERBS, METRIC_RE

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# NON-TECH DA TRANSITION CLARIFICATION QUESTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _generate_non_tech_da_questions(background_domain: str) -> List[Dict]:
    """
    Generate data analytics transition clarification questions
    tailored to a candidate's non-technical background domain.
    """
    base_questions = [
        {
            "id": "nt_reporting_tools",
            "section": "skills",
            "item_index": -1,
            "field": "skills",
            "label": "What tools have you used for reporting or data tracking? (Excel, Power BI, Tableau, Google Sheets, SQL, etc.)",
            "placeholder": "e.g., Excel (pivot tables, VLOOKUP), Google Sheets, Power BI dashboards, SQL queries..."
        },
        {
            "id": "nt_kpis_tracked",
            "section": "experience",
            "item_index": -1,
            "field": "metrics",
            "label": "What KPIs, metrics, or data points did you regularly track or report in your role?",
            "placeholder": "e.g., Monthly sales revenue, employee attrition rate, campaign CTR, customer satisfaction (CSAT)..."
        },
        {
            "id": "nt_reports_created",
            "section": "experience",
            "item_index": -1,
            "field": "bullets",
            "label": "Did you create any dashboards, reports, or data summaries? Describe what they tracked and who used them.",
            "placeholder": "e.g., Built weekly HR headcount report for leadership. Created Excel dashboard tracking hiring pipeline..."
        },
    ]

    domain_specific = {
        "HR / Talent Acquisition": {
            "id": "nt_hr_analytics",
            "section": "experience",
            "item_index": -1,
            "field": "bullets",
            "label": "Did you analyze any HR data? (e.g., attrition trends, time-to-hire, offer acceptance rates, headcount forecasting)",
            "placeholder": "e.g., Analyzed attrition data across departments, reduced time-to-hire by 2 weeks by tracking sourcing channel performance..."
        },
        "Finance / Accounting": {
            "id": "nt_finance_analytics",
            "section": "experience",
            "item_index": -1,
            "field": "bullets",
            "label": "Did you build any financial models, budget trackers, or variance reports in Excel or other tools?",
            "placeholder": "e.g., Built monthly budget vs actual variance report in Excel, forecasted quarterly P&L for 3 business units..."
        },
        "Marketing / Growth": {
            "id": "nt_marketing_analytics",
            "section": "experience",
            "item_index": -1,
            "field": "bullets",
            "label": "Did you analyze campaign performance, track digital metrics, or run A/B tests? What data did you use?",
            "placeholder": "e.g., Tracked email campaign open rates, analyzed Google Analytics traffic data, ran A/B tests on landing pages..."
        },
        "Sales / Business Development": {
            "id": "nt_sales_analytics",
            "section": "experience",
            "item_index": -1,
            "field": "bullets",
            "label": "Did you prepare sales reports, analyze pipeline data, or forecast revenue using CRM or Excel?",
            "placeholder": "e.g., Maintained Salesforce pipeline data, prepared monthly revenue reports for VP of Sales, forecasted Q4 targets..."
        },
        "Operations / Supply Chain": {
            "id": "nt_ops_analytics",
            "section": "experience",
            "item_index": -1,
            "field": "bullets",
            "label": "Did you track operational KPIs, analyze supply chain data, or create process performance reports?",
            "placeholder": "e.g., Tracked on-time delivery rates, analyzed inventory turnover using ERP data, created process efficiency reports..."
        },
    }

    questions = list(base_questions)
    if background_domain in domain_specific:
        questions.append(domain_specific[background_domain])

    return questions[:5]

def evaluate_sufficiency(resume: ParsedResume, background_domain: str = "") -> Dict:
    """
    Evaluates projects, work experiences, and skills for detailed content.
    Generates a list of questions to fill the gaps if details are insufficient.

    For non-technical candidates (background_domain set), generates DA-transition
    specific questions about reporting tools, KPIs, and data work.

    Returns a dict:
      - is_sufficient (bool)
      - insufficient_sections (list of str)
      - questions (list of dict)
      - report (list of str comments)
      - is_non_tech_mode (bool)
    """
    questions = []
    insufficient_sections = []
    report = []
    
    # ── 1. Experience Sufficiency ─────────────────────────────────────────────
    if not resume.experience:
        insufficient_sections.append("experience")
        report.append("No professional experience found in resume.")
        questions.append({
            "id": "exp_general_0",
            "section": "experience",
            "item_index": -1,
            "field": "general",
            "label": "You haven't listed any work experience. Can you describe your most recent job role and responsibilities?",
            "placeholder": "e.g., Software Engineer at Tech Corp, built and maintained web apps..."
        })
    else:
        for i, exp in enumerate(resume.experience):
            company = exp.company or f"Company {i+1}"
            title = exp.title or "Professional Role"
            
            # Check bullets count
            if not exp.bullets or len(exp.bullets) < 2:
                insufficient_sections.append("experience")
                report.append(f"Experience at {company} has sparse bullet points.")
                questions.append({
                    "id": f"exp_bullets_{i}",
                    "section": "experience",
                    "item_index": i,
                    "field": "bullets",
                    "label": f"Could you list 2-3 specific accomplishments or daily responsibilities for your role as {title} at {company}?",
                    "placeholder": "e.g., Led the migration of legacy APIs, mentored 2 junior developers, automated test suites..."
                })
                continue
                
            # Check for action verbs and metrics in bullets
            has_metrics = False
            has_strong_verbs = False
            
            for bullet in exp.bullets:
                # Metric check
                if METRIC_RE.search(bullet):
                    has_metrics = True
                # Action verb check
                first_word = bullet.split()[0].lower().rstrip("ed") if bullet.split() else ""
                if any(verb.startswith(first_word) for verb in STRONG_VERBS):
                    has_strong_verbs = True
                    
            if not has_metrics:
                report.append(f"No quantified metrics found in experience at {company}.")
                questions.append({
                    "id": f"exp_metrics_{i}",
                    "section": "experience",
                    "item_index": i,
                    "field": "metrics",
                    "label": f"Can you provide any metrics or numbers (e.g. scale, users, % improvement) for your work as {title} at {company}?",
                    "placeholder": "e.g., Reduced query times by 40%, served 100K+ daily active users, saved 5 hours/week..."
                })
                
            if not has_strong_verbs:
                report.append(f"Experience bullets at {company} start with weak or passive verbs.")
                # We don't block sufficiency just for verbs since the rewriter can fix it, 
                # but we note it in the report.

    # ── 2. Projects Sufficiency ───────────────────────────────────────────────
    if not resume.projects:
        insufficient_sections.append("projects")
        report.append("No projects found in resume.")
        questions.append({
            "id": "proj_general_0",
            "section": "projects",
            "item_index": -1,
            "field": "general",
            "label": "You don't have any major projects listed. Can you describe a key project you worked on (either personal, academic, or professional)?",
            "placeholder": "Project Name: E-Commerce App. Built a full-stack platform using React, Node.js..."
        })
    else:
        for i, proj in enumerate(resume.projects[:3]): # evaluate top 3 projects
            name = proj.name or f"Project {i+1}"
            
            # Check description depth
            if not proj.description or len(proj.description.split()) < 8:
                insufficient_sections.append("projects")
                report.append(f"Project '{name}' has a very brief description.")
                questions.append({
                    "id": f"proj_desc_{i}",
                    "section": "projects",
                    "item_index": i,
                    "field": "description",
                    "label": f"What was the main goal, system architecture, or problem you solved in the project '{name}'?",
                    "placeholder": "e.g., Built a real-time analytics pipeline to capture user actions and display graphs in a dashboard..."
                })
                
            # Check technology stack
            if not proj.tech_used or len(proj.tech_used) < 2:
                report.append(f"Project '{name}' does not specify a technology stack.")
                questions.append({
                    "id": f"proj_tech_{i}",
                    "section": "projects",
                    "item_index": i,
                    "field": "tech",
                    "label": f"Which specific technologies, languages, or databases did you use in '{name}'?",
                    "placeholder": "e.g., React, Node.js, Express, PostgreSQL, AWS S3"
                })
                
            # Check quantified outcome
            if not proj.outcome or not METRIC_RE.search(proj.outcome):
                report.append(f"Project '{name}' lacks a quantified outcome/metric.")
                questions.append({
                    "id": f"proj_outcome_{i}",
                    "section": "projects",
                    "item_index": i,
                    "field": "outcome",
                    "label": f"What was the outcome, metric, or final result of '{name}' (e.g. latency, accuracy, user count)?",
                    "placeholder": "e.g., Achieved 95% model accuracy, decreased server response time by 150ms..."
                })

    # ── 3. Skills Sufficiency ─────────────────────────────────────────────────
    if not resume.skills or len(resume.skills) < 5:
        insufficient_sections.append("skills")
        report.append("Skills section is empty or has very few items (<5).")
        questions.append({
            "id": "skills_general_0",
            "section": "skills",
            "item_index": -1,
            "field": "skills",
            "label": "Please list your core technical skills, programming languages, tools, or domain competencies.",
            "placeholder": "e.g., Python, SQL, Git, Excel, Communication, Project Management..."
        })

    # ── 4. Summary Sufficiency ────────────────────────────────────────────────
    if not resume.summary or len(resume.summary.split()) < 10:
        report.append("Professional summary is missing or too short.")
        # Summary can be generated fully by Agent 4a, so we don't necessarily prompt for it 
        # unless everything else is sparse too.

    # ── 5. Certifications Sufficiency ──────────────────────────────────────────
    if not resume.certifications:
        insufficient_sections.append("certifications")
        report.append("No certifications or online courses found in resume.")
        questions.append({
            "id": "certs_general_0",
            "section": "certifications",
            "item_index": -1,
            "field": "certifications",
            "label": "Industry-recognized certifications & online courses relevant to the role",
            "placeholder": "e.g., AWS Certified Cloud Practitioner, Coursera Machine Learning Specialization..."
        })

    # ── 6. Accomplishments Sufficiency ─────────────────────────────────────────
    if not resume.accomplishments:
        insufficient_sections.append("accomplishments")
        report.append("No achievements or awards found in resume.")
        questions.append({
            "id": "accs_general_0",
            "section": "accomplishments",
            "item_index": -1,
            "field": "accomplishments",
            "label": "Achievements / Awards (Competitions, Scholarships, Academic or professional recognitions)",
            "placeholder": "e.g., Dean's List (3 semesters), 1st place in University Hackathon, National Talent Search Scholarship..."
        })

    # ── Final Determination ───────────────────────────────────────────────────
    is_sufficient = len(insufficient_sections) == 0

    # Separate certifications/accomplishments questions to guarantee their inclusion
    macro_questions = [q for q in questions if q["id"] in ("certs_general_0", "accs_general_0")]
    other_questions = [q for q in questions if q["id"] not in ("certs_general_0", "accs_general_0")]

    # ── Non-Technical Override: inject DA transition questions ────────────────
    is_non_tech_mode = bool(background_domain)
    if is_non_tech_mode:
        nt_questions = _generate_non_tech_da_questions(background_domain)
        # Merge: non-tech questions take priority, macro questions next, others fill remaining
        merged = list(nt_questions)
        for q in macro_questions:
            if q["id"] not in {nq["id"] for nq in merged}:
                merged.append(q)
        for q in other_questions:
            if len(merged) >= 6:
                break
            if q["id"] not in {nq["id"] for nq in merged}:
                merged.append(q)
        questions = merged
        # Non-tech CVs are always considered insufficient for DA roles
        is_sufficient = False
        if "skills" not in insufficient_sections:
            insufficient_sections.append("skills")
        report.insert(0,
            f"Non-technical background detected ({background_domain}). "
            f"Activating Data Analyst Transition Mode — please answer the questions below "
            f"to help reframe your experience in analytics language."
        )
    else:
        # Tech mode: prioritize macro-questions (certifications/accomplishments)
        questions = macro_questions + other_questions
        questions = questions[:5]

    return {
        "is_sufficient": is_sufficient,
        "insufficient_sections": list(set(insufficient_sections)),
        "questions": questions,
        "report": report,
        "is_non_tech_mode": is_non_tech_mode,
    }
