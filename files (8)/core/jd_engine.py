"""
core/jd_engine.py
────────────────────────────────────────────────────────
Module 2 — Job Description Intelligence Engine

Semantically analyses JDs to extract:
  - Explicit required/preferred skills
  - Role domain & seniority level
  - Implicit/inferred skill requirements
  - Recruiter expectation signals
  - Tech stack fingerprint
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from config import TECH_TAXONOMY, IMPLICIT_SKILL_MAP, ROLE_DOMAINS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedJD:
    raw_text           : str = ""
    role_title         : str = ""
    company_name       : str = ""
    domain             : str = ""
    seniority          : str = ""          # junior / mid / senior / lead / staff
    required_skills    : List[str] = field(default_factory=list)
    preferred_skills   : List[str] = field(default_factory=list)
    implicit_skills    : List[str] = field(default_factory=list)
    must_have          : List[str] = field(default_factory=list)
    nice_to_have       : List[str] = field(default_factory=list)
    responsibilities   : List[str] = field(default_factory=list)
    tech_stack         : List[str] = field(default_factory=list)
    years_experience   : Optional[int] = None
    education_req      : str = ""
    recruiter_signals  : List[str] = field(default_factory=list)
    all_keywords       : List[str] = field(default_factory=list)
    word_count         : int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Seniority detection
# ──────────────────────────────────────────────────────────────────────────────

SENIORITY_MAP = {
    "junior"   : ["junior","entry level","entry-level","fresher","graduate","0-2 years","0-1 year"],
    "mid"      : ["mid-level","mid level","2-4 years","2-5 years","3+ years","associate"],
    "senior"   : ["senior","sr.","5+ years","5-8 years","6+ years","experienced"],
    "lead"     : ["lead","tech lead","team lead","principal","architect","8+ years","10+ years"],
    "staff"    : ["staff","distinguished","vp of","head of","director of engineering"],
}

def detect_seniority(text: str) -> str:
    text_lower = text.lower()
    for level, keywords in SENIORITY_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return level
    return "mid"  # default assumption


# ──────────────────────────────────────────────────────────────────────────────
# Years of experience extraction
# ──────────────────────────────────────────────────────────────────────────────

YOE_RE = re.compile(
    r"(\d+)\+?\s*(?:-\s*\d+)?\s*years?\s*(?:of\s*)?(?:experience|exp\.?)",
    re.I,
)

def extract_years_experience(text: str) -> Optional[int]:
    matches = YOE_RE.findall(text)
    if matches:
        return int(matches[0])
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Domain classification
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_SIGNALS: dict[str, List[str]] = {
    "Artificial Intelligence / ML" : ["machine learning","deep learning","llm","nlp","computer vision","ml engineer","ai engineer","data scientist","model training","neural network","transformers","pytorch","tensorflow"],
    "Data Science / Analytics"     : ["data analyst","business intelligence","sql","tableau","power bi","data visualization","statistics","r language","data pipeline"],
    "Software Engineering (Backend)": ["backend","server-side","rest api","microservices","database","postgresql","node.js","django","fastapi","spring boot","java backend"],
    "Software Engineering (Frontend)": ["frontend","react","vue","angular","ui/ux","css","javascript","typescript frontend","next.js","svelte"],
    "Full Stack Development"        : ["full stack","fullstack","full-stack","mean stack","mern stack","both frontend and backend"],
    "DevOps / Cloud Engineering"    : ["devops","cloud","aws","gcp","azure","kubernetes","terraform","ci/cd","infrastructure","sre","platform engineering"],
    "Embedded Systems / IoT"        : ["embedded","firmware","rtos","arm","stm32","uart","can bus","iot","hardware","microcontroller"],
    "VLSI / Hardware Engineering"   : ["vlsi","verilog","vhdl","fpga","asic","rtl","synthesis","timing analysis","cadence","synopsys"],
    "Cybersecurity"                 : ["security","penetration testing","soc","vulnerability","compliance","siem","firewall","cryptography"],
    "Product Management"            : ["product manager","roadmap","stakeholder","agile","scrum","go-to-market","product strategy"],
    "HR / Talent Acquisition"       : ["hr", "human resources", "recruiting", "talent acquisition", "onboarding", "recruiter", "benefits", "compensation", "employee relations", "hris", "workday", "payroll", "sourcing"],
    "Finance / Accounting"          : ["finance", "accounting", "ledger", "bookkeeping", "cpa", "cfa", "financial", "budgeting", "forecasting", "taxation", "auditing", "p&l", "valuation", "sap finance"],
    "Marketing / Growth"            : ["marketing", "seo", "sem", "growth hacking", "brand", "digital marketing", "social media", "copywriting", "campaign", "hubspot", "advertising", "ppc"],
    "Sales / Business Development"  : ["sales", "b2b", "lead generation", "salesforce", "cold calling", "negotiation", "account manager", "pipeline", "quota", "closing deals"],
    "Healthcare / Biotech"          : ["healthcare", "clinical", "patient care", "nursing", "medical", "biotech", "diagnostics", "emr", "ehr", "hipaa", "clinic", "hospital"],
    "Education / Academic"          : ["education", "academic", "teacher", "teaching", "curriculum", "pedagogy", "lms", "classroom", "student", "instructional design", "e-learning"],
    "Operations / Supply Chain"     : ["operations", "supply chain", "logistics", "procurement", "inventory", "six sigma", "lean", "warehouse", "vendor", "ops"],
}

def classify_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for domain, signals in DOMAIN_SIGNALS.items():
        scores[domain] = sum(1 for s in signals if s in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Software Engineering (Backend)"


# ──────────────────────────────────────────────────────────────────────────────
# Required vs preferred skill separation
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_SIGNALS   = ["required","must have","mandatory","essential","minimum","you must","we require","need","necessary"]
PREFERRED_SIGNALS  = ["preferred","nice to have","plus","bonus","good to have","ideally","desirable","advantageous","familiarity"]

def split_required_preferred(text: str) -> tuple[str, str]:
    """Split JD text into required and preferred sections."""
    lines = text.split("\n")
    req_lines  = []
    pref_lines = []
    mode = "req"

    for line in lines:
        ll = line.lower()
        if any(sig in ll for sig in PREFERRED_SIGNALS):
            mode = "pref"
        elif any(sig in ll for sig in REQUIRED_SIGNALS):
            mode = "req"
        if mode == "req":
            req_lines.append(line)
        else:
            pref_lines.append(line)

    return "\n".join(req_lines), "\n".join(pref_lines)


# ──────────────────────────────────────────────────────────────────────────────
# Tech keyword extraction from text
# ──────────────────────────────────────────────────────────────────────────────

def extract_tech_keywords(text: str) -> List[str]:
    all_tech = []
    for terms in TECH_TAXONOMY.values():
        all_tech.extend(terms)

    text_lower = text.lower()
    found = []
    for tech in all_tech:
        if re.search(r"\b" + re.escape(tech) + r"\b", text_lower):
            found.append(tech)
    return list(dict.fromkeys(found))


# ──────────────────────────────────────────────────────────────────────────────
# Implicit skill inference
# ──────────────────────────────────────────────────────────────────────────────

def infer_implicit_skills(explicit_skills: List[str], jd_text: str) -> List[str]:
    """
    Given what's explicitly mentioned, infer what's implicitly expected.
    E.g. if "mlops" appears → Docker, Kubernetes, MLflow are implied.
    """
    text_lower = jd_text.lower()
    implicit = set()

    for trigger, implied_skills in IMPLICIT_SKILL_MAP.items():
        if trigger in text_lower:
            for skill in implied_skills:
                if skill not in explicit_skills:
                    implicit.add(skill)

    return sorted(implicit)


# ──────────────────────────────────────────────────────────────────────────────
# Recruiter signal extraction
# ──────────────────────────────────────────────────────────────────────────────

RECRUITER_SIGNALS = [
    (r"fast[\s-]?paced",        "Comfortable in fast-paced environments"),
    (r"cross[\s-]?functional",  "Cross-functional collaboration expected"),
    (r"ownership",              "Strong ownership & autonomy expected"),
    (r"startup",                "Startup mindset — wear multiple hats"),
    (r"agile|scrum",            "Agile/Scrum workflow"),
    (r"communicate|communication","Strong communication skills valued"),
    (r"mentor",                 "Mentorship or leadership opportunities"),
    (r"research",               "Research orientation / publications valued"),
    (r"open[\s-]?source",       "Open-source contributions valued"),
    (r"product",                "Product thinking expected alongside tech"),
]

def extract_recruiter_signals(text: str) -> List[str]:
    signals = []
    for pattern, signal in RECRUITER_SIGNALS:
        if re.search(pattern, text, re.I):
            signals.append(signal)
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Responsibility extraction
# ──────────────────────────────────────────────────────────────────────────────

BULLET_RE = re.compile(r"^[\s]*[•\-\*\u2022>]\s+(.+)$", re.M)

def extract_responsibilities(text: str) -> List[str]:
    # Find section
    lines = text.split("\n")
    resp_section = []
    in_section = False

    for line in lines:
        ll = line.lower().strip()
        if any(kw in ll for kw in ["responsibilit","you will","what you'll","role","duties","your role"]):
            in_section = True
        elif any(kw in ll for kw in ["requirement","qualification","you have","skills","about you"]):
            in_section = False
        if in_section and line.strip():
            resp_section.append(line)

    resp_text = "\n".join(resp_section)
    bullets = BULLET_RE.findall(resp_text)
    return [b.strip() for b in bullets if b.strip()][:12]


# ──────────────────────────────────────────────────────────────────────────────
# Education requirement
# ──────────────────────────────────────────────────────────────────────────────

def extract_education_req(text: str) -> str:
    patterns = [
        r"(b\.?tech|b\.?e\.?|bachelor['s]?\s+(?:degree\s+)?in\s+\w+(?:\s+\w+)?)",
        r"(m\.?tech|m\.?e\.?|master['s]?\s+degree)",
        r"(phd|doctorate)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m.group().strip()
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Role title extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_role_title(text: str) -> str:
    lines = text.strip().split("\n")
    for line in lines[:5]:
        clean = line.strip()
        if 2 < len(clean) < 80 and not EMAIL_RE_SIMPLE.search(clean):
            return clean
    return lines[0].strip() if lines else ""

EMAIL_RE_SIMPLE = re.compile(r"@")


# ──────────────────────────────────────────────────────────────────────────────
# Master JD parse function
# ──────────────────────────────────────────────────────────────────────────────

def parse_jd(jd_text: str, company_name: str = "") -> ParsedJD:
    """
    Full pipeline: raw JD text → ParsedJD dataclass.
    """
    jd = ParsedJD()
    jd.raw_text    = jd_text
    jd.word_count  = len(jd_text.split())
    jd.company_name = company_name

    # 1. Role metadata
    jd.role_title       = extract_role_title(jd_text)
    jd.seniority        = detect_seniority(jd_text)
    jd.years_experience = extract_years_experience(jd_text)
    jd.domain           = classify_domain(jd_text)
    jd.education_req    = extract_education_req(jd_text)

    # 2. Required vs preferred split
    req_text, pref_text = split_required_preferred(jd_text)

    # 3. Tech extraction
    jd.required_skills  = extract_tech_keywords(req_text)
    jd.preferred_skills = extract_tech_keywords(pref_text)
    jd.tech_stack       = list(dict.fromkeys(jd.required_skills + jd.preferred_skills))

    # 4. Must-have vs nice-to-have (human readable)
    jd.must_have    = jd.required_skills[:8]
    jd.nice_to_have = jd.preferred_skills[:6]

    # 5. Implicit skills
    jd.implicit_skills = infer_implicit_skills(jd.tech_stack, jd_text)

    # 6. Responsibilities
    jd.responsibilities = extract_responsibilities(jd_text)

    # 7. Recruiter signals
    jd.recruiter_signals = extract_recruiter_signals(jd_text)

    # 8. All keywords (for matching)
    jd.all_keywords = list(dict.fromkeys(jd.tech_stack + jd.implicit_skills))

    logger.info(f"Parsed JD: '{jd.role_title}' | domain={jd.domain} | "
                f"seniority={jd.seniority} | {len(jd.all_keywords)} keywords")

    return jd

if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    sample_jd = """
    We are looking for a Senior Software Engineer (Backend).
    Must have 5+ years of experience in building scalable backend systems.
    Required: Python, Django, REST API, PostgreSQL.
    Preferred: Docker, Kubernetes, AWS.
    You must be comfortable in a fast-paced environment and have a startup mindset.
    """
    print("Running JD Engine Test...")
    parsed = parse_jd(sample_jd, company_name="Tech Startup Inc.")
    print("\n--- Parsed JD Result ---")
    pprint.pprint(parsed.__dict__)
