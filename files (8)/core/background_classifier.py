"""
core/background_classifier.py
────────────────────────────────────────────────────────────────────────────────
Background Classification Engine.

Detects whether a candidate's resume is:
  - Technical (routes to existing enhancement pipeline)
  - Non-Technical (routes to ATS Architect pipeline for DA transition)

Also identifies the candidate's primary background domain from:
  HR, Marketing, Sales, Finance, Operations, Education, Healthcare,
  Customer Success, Administration/Business Support

Uses signal-based keyword matching against NON_TECH_BACKGROUND_SIGNALS in config.
No LLM — fast, deterministic, zero cost.
"""

import re
import logging
from typing import Dict, Tuple

from core.resume_parser import ParsedResume

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Technical role signal words — if resume strongly matches these, it's technical
# ──────────────────────────────────────────────────────────────────────────────

TECHNICAL_ROLE_SIGNALS = [
    # Core tech roles
    "software engineer", "software developer", "backend engineer", "frontend engineer",
    "full stack", "ml engineer", "machine learning engineer", "ai engineer",
    "data engineer", "data scientist", "devops engineer", "cloud engineer",
    "embedded engineer", "firmware engineer", "vlsi engineer", "hardware engineer",
    "cybersecurity analyst", "security engineer", "network engineer",
    # Tech titles in experience
    "sde", "sde-1", "sde-2", "sre", "mlops", "platform engineer",
    # Hard technical skills (strong signals)
    "pytorch", "tensorflow", "docker", "kubernetes", "microservices",
    "neural network", "llm", "fine-tuning", "verilog", "vhdl", "fpga",
    "penetration testing", "kali linux", "burp suite",
    # Programming depth signals
    "object oriented programming", "system design", "distributed systems",
    "computer science", "b.tech computer", "m.tech", "computer engineering",
]

TECHNICAL_SKILL_THRESHOLD = 3   # if ≥ N tech skills found → technical
TECHNICAL_SIGNAL_THRESHOLD = 1  # if ≥ N strong role signals → technical

# Skills that are BOTH technical and non-technical (don't count either way)
AMBIGUOUS_SKILLS = {
    "excel", "powerpoint", "sql", "google analytics", "tableau",
    "power bi", "data analysis", "reporting", "microsoft office",
    "ms office", "project management", "agile", "scrum",
}


# ──────────────────────────────────────────────────────────────────────────────
# Core tech skills from the DA / Software taxonomy (exclusive to tech)
# ──────────────────────────────────────────────────────────────────────────────

EXCLUSIVE_TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "scala", "r", "matlab", "bash", "linux",
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "pandas", "numpy", "matplotlib", "seaborn", "flask", "django", "fastapi",
    "react", "next.js", "node.js", "vue", "angular",
    "docker", "kubernetes", "aws", "gcp", "azure", "terraform", "github actions",
    "postgresql", "mongodb", "redis", "elasticsearch", "spark", "airflow", "kafka",
    "langchain", "huggingface", "transformers", "onnx", "mlflow",
    "verilog", "vhdl", "fpga", "rtos", "embedded linux", "arm", "stm32",
    "metasploit", "wireshark", "nmap", "burp suite", "kali linux",
    "junit", "pytest", "selenium", "cypress", "jest",
}


def classify_background(resume: ParsedResume) -> Dict:
    """
    Classify whether a resume is technical or non-technical.
    Returns a structured report used for pipeline routing.

    Returns:
        dict with keys:
          - is_non_technical (bool)
          - background_domain (str) — primary non-tech domain or ""
          - confidence (float 0–1)
          - tech_signal_count (int)
          - domain_scores (dict)
          - explanation (str)
    """
    from config import NON_TECH_BACKGROUND_SIGNALS

    raw_lower = resume.raw_text.lower()
    skills_lower = {s.lower() for s in resume.skills}
    titles_lower = " ".join(
        (exp.title or "").lower() for exp in resume.experience
    )

    # ── 1. Check for exclusive technical skills ───────────────────────────────
    exclusive_tech_found = EXCLUSIVE_TECH_SKILLS & skills_lower
    tech_depth = len(exclusive_tech_found)

    # ── 2. Check for technical role signal words in titles + raw text ─────────
    tech_signal_matches = [
        sig for sig in TECHNICAL_ROLE_SIGNALS
        if re.search(r"\b" + re.escape(sig) + r"\b", raw_lower)
    ]
    tech_signal_count = len(tech_signal_matches)

    # ── 3. Score each non-technical background domain ─────────────────────────
    domain_scores: Dict[str, float] = {}
    for domain, signals in NON_TECH_BACKGROUND_SIGNALS.items():
        score = sum(
            1 for sig in signals
            if re.search(r"\b" + re.escape(sig.lower()) + r"\b", raw_lower)
        )
        domain_scores[domain] = score

    best_domain = max(domain_scores, key=domain_scores.get)
    best_domain_score = domain_scores[best_domain]

    # ── 4. Decision Logic ──────────────────────────────────────────────────────
    # Technical if: strong exclusive tech skills OR strong role title signals
    is_technical = (
        tech_depth >= TECHNICAL_SKILL_THRESHOLD
        or tech_signal_count >= TECHNICAL_SIGNAL_THRESHOLD
    )

    # Override: even if few tech skills, if non-tech domain has strong signal → non-tech
    # e.g. a person who knows Excel + SQL but has HR/Marketing job titles
    if is_technical and best_domain_score >= 3 and tech_depth < 5:
        is_technical = False

    is_non_technical = not is_technical

    # ── 5. Confidence score ───────────────────────────────────────────────────
    if is_non_technical:
        # High confidence if strong non-tech domain signal + low tech depth
        confidence = min(
            0.5 + (best_domain_score / 10) + max(0, (5 - tech_depth) / 10),
            1.0
        )
        domain = best_domain if best_domain_score > 0 else "General / Business"
    else:
        confidence = min(
            0.5 + (tech_depth / 20) + (tech_signal_count / 10),
            1.0
        )
        domain = ""

    # ── 6. Explanation ────────────────────────────────────────────────────────
    if is_non_technical:
        explanation = (
            f"Classified as NON-TECHNICAL. "
            f"Primary domain: '{domain}' (signal score: {best_domain_score}). "
            f"Exclusive tech skill count: {tech_depth} (below threshold of {TECHNICAL_SKILL_THRESHOLD}). "
            f"Tech role signals: {tech_signal_count}."
        )
    else:
        explanation = (
            f"Classified as TECHNICAL. "
            f"Exclusive tech skills found: {tech_depth} ({', '.join(list(exclusive_tech_found)[:5])}). "
            f"Tech role signals: {tech_signal_count}."
        )

    logger.info(f"Background classifier: {explanation}")

    return {
        "is_non_technical": is_non_technical,
        "background_domain": domain,
        "confidence": round(confidence, 2),
        "tech_signal_count": tech_signal_count,
        "tech_depth": tech_depth,
        "exclusive_tech_skills": sorted(list(exclusive_tech_found)),
        "domain_scores": domain_scores,
        "explanation": explanation,
    }
