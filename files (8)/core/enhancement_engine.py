"""
core/enhancement_engine.py — v4
────────────────────────────────────────────────────────────────────────────────
Multi-agent CV enhancement pipeline.
Uses core.llm_backend — supports Ollama, HuggingFace, and fine-tuned models.
Zero external API calls. Zero tokenization cost.

Agents:
  Agent 1 (Intake)        — Structured onboarding form (no LLM)
  Agent 2 (Validator)     — Merge form + parsed data (no LLM)
  Agent 3 (Gap Analyst)   — Identify missing JD keywords (no LLM)
  Agent 4a (Summariser)   — Rewrite professional summary (LLM)
  Agent 4b (Bullet Writer)— Rewrite experience bullets (LLM)
  Agent 4c (Proj Enhancer)— Improve project descriptions (LLM)
  Agent 5 (Skill Booster) — Inject JD keywords → ATS 90+ (LLM)
  Agent 6 (QA Guard)      — Strip contact bleed, validate output (no LLM)

Contact info is COPIED VERBATIM — no LLM ever sees or modifies it.
"""

import re
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from core.llm_backend import call_llm           # ← unified backend
from config import OLLAMA_MODEL                 # only used in error messages
from core.resume_parser import ParsedResume, WorkExperience, Project, Education
from core.jd_engine import ParsedJD
from core.scoring_engine import ScoreResult

logger = logging.getLogger(__name__)

_CONTACT_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}"
    r"|(\+?\d[\d\s\-().]{7,15}\d)"
    r"|linkedin\.com/in/[\w\-]+"
    r"|github\.com/[\w\-]+",
    re.I,
)


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OnboardingData:
    num_internships       : int = 0
    internship_details    : List[Dict] = field(default_factory=list)
    num_fulltime_jobs     : int = 0
    fulltime_details      : List[Dict] = field(default_factory=list)
    num_major_projects    : int = 3
    major_project_details : List[Dict] = field(default_factory=list)
    num_minor_projects    : int = 0
    minor_project_details : List[Dict] = field(default_factory=list)
    ug_degree     : str = ""
    ug_college    : str = ""
    ug_cgpa       : str = ""
    pg_degree     : str = ""
    pg_college    : str = ""
    pg_cgpa       : str = ""
    dual_degree   : str = ""
    sec_board     : str = ""
    sec_score     : str = ""
    pri_board     : str = ""
    pri_score     : str = ""
    clubs_societies   : List[str] = field(default_factory=list)
    volunteering      : List[str] = field(default_factory=list)
    linkedin_pdf_path : str = ""


@dataclass
class EnhancedResume:
    name            : str = ""
    email           : str = ""
    phone           : str = ""
    linkedin        : str = ""
    github          : str = ""
    portfolio       : str = ""
    target_role     : str = ""
    domain          : str = ""
    enhanced_summary    : str = ""
    enhanced_skills     : List[str] = field(default_factory=list)
    enhanced_experience : List[WorkExperience] = field(default_factory=list)
    enhanced_projects   : List[Project] = field(default_factory=list)
    education           : List[Education] = field(default_factory=list)
    certifications      : List[str] = field(default_factory=list)
    accomplishments     : List[str] = field(default_factory=list)
    publications        : List[str] = field(default_factory=list)
    clubs_volunteering  : List[str] = field(default_factory=list)
    improvement_notes   : List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _llm(prompt: str, max_tokens: int = 600) -> str:
    """Thin wrapper — delegates to the configured backend."""
    return call_llm(prompt, max_tokens=max_tokens)


def _parse_bullets(text: str, max_n: int = 5) -> List[str]:
    bullets = []
    for line in text.split("\n"):
        clean = re.sub(r"^[\s\-•*\d.)]+", "", line).strip()
        if _CONTACT_RE.search(clean):
            continue
        if 15 < len(clean) < 200:
            bullets.append(clean)
        if len(bullets) >= max_n:
            break
    return bullets


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Validator
# ══════════════════════════════════════════════════════════════════════════════

def agent_validate_and_merge(parsed: ParsedResume, onboarding: OnboardingData) -> ParsedResume:
    out = deepcopy(parsed)

    for detail in onboarding.internship_details:
        we = WorkExperience(
            company=detail.get("company", ""), title=detail.get("title", ""),
            duration=detail.get("duration", ""),
            bullets=[b for b in detail.get("bullets", []) if b.strip()],
        )
        already = any(
            we.company.lower() in e.company.lower() or e.company.lower() in we.company.lower()
            for e in out.experience
        )
        if not already and (we.company or we.title):
            out.experience.append(we)

    for detail in onboarding.fulltime_details:
        we = WorkExperience(
            company=detail.get("company", ""), title=detail.get("title", ""),
            duration=detail.get("duration", ""),
            bullets=[b for b in detail.get("bullets", []) if b.strip()],
        )
        already = any(
            we.company.lower() in e.company.lower()
            for e in out.experience if e.company
        )
        if not already and (we.company or we.title):
            out.experience.insert(0, we)

    for detail in onboarding.major_project_details:
        proj = Project(
            name=detail.get("name", ""),
            tech_used=[t.strip() for t in detail.get("tech", "").split(",") if t.strip()],
            description=detail.get("description", ""),
            outcome=detail.get("outcome", ""),
            link=detail.get("link", ""),
        )
        already = any(
            proj.name.lower() in p.name.lower() or p.name.lower() in proj.name.lower()
            for p in out.projects if p.name
        )
        if not already and proj.name:
            out.projects.insert(0, proj)

    if onboarding.ug_degree or onboarding.ug_college:
        from core.resume_parser import Education as Edu
        ug = Edu(
            degree=onboarding.ug_degree, institution=onboarding.ug_college,
            gpa=onboarding.ug_cgpa,
        )
        college_norm = re.sub(r"[^a-z0-9]", "", (onboarding.ug_college or "").lower())
        found = False
        if college_norm:
            for e in out.education:
                inst_norm = re.sub(r"[^a-z0-9]", "", (e.institution or "").lower())
                deg_norm = re.sub(r"[^a-z0-9]", "", (e.degree or "").lower())
                if (inst_norm and (college_norm in inst_norm or inst_norm in college_norm)) or (college_norm in deg_norm):
                    if not e.institution:
                        e.institution = onboarding.ug_college
                    if onboarding.ug_cgpa and not e.gpa:
                        e.gpa = onboarding.ug_cgpa
                    found = True
                    break
        else:
            if out.education:
                found = True
        if not found:
            out.education.insert(0, ug)

    if onboarding.pg_degree or onboarding.pg_college:
        from core.resume_parser import Education as Edu
        pg = Edu(
            degree=onboarding.pg_degree, institution=onboarding.pg_college,
            gpa=onboarding.pg_cgpa,
        )
        college_norm = re.sub(r"[^a-z0-9]", "", (onboarding.pg_college or "").lower())
        found = False
        if college_norm:
            for e in out.education:
                inst_norm = re.sub(r"[^a-z0-9]", "", (e.institution or "").lower())
                deg_norm = re.sub(r"[^a-z0-9]", "", (e.degree or "").lower())
                if (inst_norm and (college_norm in inst_norm or inst_norm in college_norm)) or (college_norm in deg_norm):
                    if not e.institution:
                        e.institution = onboarding.pg_college
                    if onboarding.pg_cgpa and not e.gpa:
                        e.gpa = onboarding.pg_cgpa
                    found = True
                    break
        if not found and (onboarding.pg_college or onboarding.pg_degree):
            out.education.insert(0, pg)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Gap Analyst (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def agent_gap_analysis(resume: ParsedResume, jd: ParsedJD, scores: ScoreResult) -> List[str]:
    resume_lower = resume.raw_text.lower()
    missing = [
        kw for kw in jd.all_keywords
        if kw.lower() not in resume_lower
    ]
    missing_priority = sorted(
        missing,
        key=lambda k: (k in jd.must_have, len(k)),
        reverse=True,
    )
    return missing_priority[:20]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4a — Summary Rewriter
# ══════════════════════════════════════════════════════════════════════════════

def agent_rewrite_summary(resume: ParsedResume, jd: ParsedJD) -> str:
    existing = resume.summary or ""
    bullets_preview = " | ".join(
        b for exp in resume.experience[:2] for b in exp.bullets[:2]
    )

    prompt = f"""Rewrite this professional summary for a {jd.role_title} role ({jd.domain}, {jd.seniority}-level).

Existing summary: {existing[:400] if existing else "None"}
Candidate skills: {", ".join(resume.skills[:20])}
Key experience snippet: {bullets_preview[:300]}
JD must-have skills: {", ".join(jd.must_have[:8])}
JD domain: {jd.domain}
Seniority: {jd.seniority}

Write a 3-sentence ATS-optimised summary. Naturally include: {", ".join(jd.must_have[:5])}.
Use strong action-oriented language. Do NOT invent metrics or facts not present above.
Output ONLY the summary paragraph. No quotes, no labels."""

    result = _llm(prompt, max_tokens=200)
    result = _CONTACT_RE.sub("", result).strip()
    if len(result) < 30:
        return existing or f"Experienced {jd.seniority}-level professional in {jd.domain} with skills in {', '.join(resume.skills[:5])}."
    return result


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4b — Bullet Rewriter
# ══════════════════════════════════════════════════════════════════════════════

def agent_rewrite_bullets(
    bullets: List[str],
    job_title: str,
    company: str,
    jd: ParsedJD,
    missing: List[str],
    max_bullets: int = 4,
) -> List[str]:
    if not bullets:
        return []

    bullets_text = "\n".join(f"- {b}" for b in bullets[:6])
    inject_kw = ", ".join(missing[:6])

    prompt = f"""Rewrite these resume bullet points for a {job_title} at {company} applying to: {jd.role_title}.

Original bullets:
{bullets_text}

JD keywords to naturally incorporate if relevant: {inject_kw}
Role domain: {jd.domain}

Rules:
- Start each bullet with a strong past-tense action verb (Built, Developed, Led, Reduced, Increased, etc.)
- Add or preserve specific metrics wherever possible (%, x, users, ms, K, M)
- Keep each bullet under 20 words
- Do NOT invent facts or numbers not present in the originals
- Output {max_bullets} bullets, one per line, no bullet markers

Output ONLY the rewritten bullets, one per line."""

    raw = _llm(prompt, max_tokens=max_bullets * 50)
    return _parse_bullets(raw, max_n=max_bullets) or bullets[:max_bullets]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4c — Project Enhancer
# ══════════════════════════════════════════════════════════════════════════════

def agent_enhance_project(proj: Project, jd: ParsedJD, missing: List[str]) -> Project:
    prompt = f"""Enhance this project description for a {jd.role_title} ({jd.domain}) resume.

Project name: {proj.name}
Current description: {proj.description or "Not provided"}
Current tech stack: {", ".join(proj.tech_used) if proj.tech_used else "Not specified"}
Current outcome: {proj.outcome or "Not provided"}
JD keywords to incorporate naturally: {", ".join(missing[:8])}

Output in EXACTLY this format (3 lines, no extra text):
DESCRIPTION: [one strong sentence describing what was built and its scale/purpose]
TECH: [comma-separated tech stack, include JD keywords where genuine]
OUTCOME: [one sentence with a measurable result or clear business impact]"""

    resp   = _llm(prompt, max_tokens=200)
    desc_m = re.search(r"DESCRIPTION:\s*(.+)", resp)
    tech_m = re.search(r"TECH:\s*(.+)", resp)
    out_m  = re.search(r"OUTCOME:\s*(.+)", resp)

    desc    = _CONTACT_RE.sub("", desc_m.group(1).strip()) if desc_m else proj.description
    tech    = [t.strip() for t in tech_m.group(1).split(",")] if tech_m else proj.tech_used
    outcome = _CONTACT_RE.sub("", out_m.group(1).strip()) if out_m else proj.outcome

    return Project(
        name=proj.name,
        description=desc,
        tech_used=tech[:8],
        outcome=outcome,
        link=proj.link,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Skill Booster (targets ATS ≥ 90)
# ══════════════════════════════════════════════════════════════════════════════

def agent_boost_skills(resume: ParsedResume, jd: ParsedJD, missing: List[str]) -> List[str]:
    prompt = f"""A candidate is applying for: {jd.role_title} ({jd.domain})

Their current skills: {", ".join(resume.skills[:20])}
Skills missing from JD: {", ".join(missing[:12])}

Which missing skills would this candidate very likely know given their existing stack?
Reasoning: if they know PyTorch → likely know NumPy; if AWS → likely know S3/EC2.

Return ONLY a comma-separated list of inferred skills (max 10). If none, return NONE."""

    resp = _llm(prompt, max_tokens=150).strip()
    inferable = [] if resp.upper().strip() == "NONE" else [
        s.strip() for s in resp.split(",") if s.strip() and len(s.strip()) < 40
    ]

    jd_kw_lower = {k.lower() for k in jd.all_keywords}
    matched     = [s for s in resume.skills if s.lower() in jd_kw_lower]
    inferable_  = [s for s in inferable if s.lower() not in {m.lower() for m in matched}]
    rest        = [s for s in resume.skills if s not in matched]

    # Dynamically inject all missing required skills from the JD
    all_required = [k for k in jd.required_skills if k.lower() not in {s.lower() for s in matched + inferable_}]
    
    # Dynamically inject up to 6 missing preferred skills from the JD
    all_preferred = [k for k in jd.preferred_skills if k.lower() not in {s.lower() for s in matched + inferable_ + all_required}][:6]
    
    # Dynamically inject up to 4 implicit skills from the JD
    all_implicit = [k for k in jd.implicit_skills if k.lower() not in {s.lower() for s in matched + inferable_ + all_required + all_preferred}][:4]

    combined = matched + all_required + all_preferred + all_implicit + inferable_ + rest
    return list(dict.fromkeys(combined))[:50]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — QA Guard
# ══════════════════════════════════════════════════════════════════════════════

def agent_qa(enhanced: EnhancedResume, original_contact: dict) -> EnhancedResume:
    enhanced.name      = original_contact.get("name",      enhanced.name)
    enhanced.email     = original_contact.get("email",     enhanced.email)
    enhanced.phone     = original_contact.get("phone",     enhanced.phone)
    enhanced.linkedin  = original_contact.get("linkedin",  enhanced.linkedin)
    enhanced.github    = original_contact.get("github",    enhanced.github)
    enhanced.portfolio = original_contact.get("portfolio", enhanced.portfolio)

    enhanced.enhanced_summary = _CONTACT_RE.sub("", enhanced.enhanced_summary).strip()

    # ── Filter experience: reject placeholder or contact-bleed entries ────────────
    _PLACEHOLDER_RE = re.compile(
        r"(recent employer|company name|job title|professional role|N/A"
        r"|\d{7,})",  # phone numbers leaked as company names
        re.I,
    )
    _PHONE_ONLY_RE = re.compile(r"^[\d\s\-().+]+$")

    clean_exp = []
    for exp in enhanced.enhanced_experience:
        title   = (exp.title or "").strip()
        company = (exp.company or "").strip()
        # Skip if title or company is a placeholder / pure phone number
        if _PLACEHOLDER_RE.search(title) or _PLACEHOLDER_RE.search(company):
            continue
        if _PHONE_ONLY_RE.match(title) or _PHONE_ONLY_RE.match(company):
            continue
        # Skip if both are empty
        if not title and not company:
            continue
        exp.bullets = [
            _CONTACT_RE.sub("", b).strip()
            for b in exp.bullets
            if b.strip() and not _CONTACT_RE.fullmatch(b.strip())
        ]
        clean_exp.append(exp)
    enhanced.enhanced_experience = clean_exp

    for proj in enhanced.enhanced_projects:
        proj.description = _CONTACT_RE.sub("", proj.description or "").strip()
        proj.outcome     = _CONTACT_RE.sub("", proj.outcome or "").strip()

    enhanced.accomplishments = [
        a for a in enhanced.accomplishments
        if a.strip() and not _CONTACT_RE.search(a)
    ]

    # ── Deduplicate skills case-insensitively ─────────────────────────────────────
    seen_s: set = set()
    deduped_skills = []
    for s in enhanced.enhanced_skills:
        key = s.lower().strip()
        if key and key not in seen_s:
            seen_s.add(key)
            deduped_skills.append(s)
    enhanced.enhanced_skills = deduped_skills

    return enhanced


# ══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def enhance_resume(
    resume: ParsedResume,
    jd: ParsedJD,
    scores: ScoreResult,
    onboarding: Optional[OnboardingData] = None,
    rewrite_all: bool = True,
) -> EnhancedResume:
    out = EnhancedResume()

    original_contact = {
        "name":      resume.name,
        "email":     resume.email,
        "phone":     resume.phone,
        "linkedin":  resume.linkedin,
        "github":    resume.github,
        "portfolio": getattr(resume, "portfolio", ""),
    }

    out.name      = resume.name
    out.email     = resume.email
    out.phone     = resume.phone
    out.linkedin  = resume.linkedin
    out.github    = resume.github
    out.portfolio = getattr(resume, "portfolio", "")
    out.target_role = jd.role_title
    out.domain      = jd.domain

    # Agent 2
    if onboarding:
        logger.info("Agent 2: Merging onboarding data...")
        resume = agent_validate_and_merge(resume, onboarding)

    out.education      = resume.education
    out.certifications = resume.certifications
    out.publications   = resume.publications
    clubs = (onboarding.clubs_societies + onboarding.volunteering) if onboarding else []
    out.accomplishments = [
        a for a in resume.accomplishments
        if a.strip() and not _CONTACT_RE.search(a)
    ] + clubs

    # Agent 3
    logger.info("Agent 3: Gap analysis...")
    missing = agent_gap_analysis(resume, jd, scores)

    # Agent 4a
    logger.info("Agent 4a: Rewriting summary...")
    out.enhanced_summary = agent_rewrite_summary(resume, jd)

    # Agent 4b
    logger.info("Agent 4b: Rewriting experience bullets...")
    enhanced_exp = []
    for exp in resume.experience[:4]:
        we = deepcopy(exp)
        bullets_src = exp.bullets if (rewrite_all or not exp.bullets) else \
            [b for b in exp.bullets if b in scores.weak_bullets] or exp.bullets[:4]
        we.bullets = agent_rewrite_bullets(
            bullets_src, exp.title, exp.company, jd, missing,
            max_bullets=min(4, len(exp.bullets) + 1),
        )
        enhanced_exp.append(we)
    out.enhanced_experience = enhanced_exp

    # Agent 4c
    logger.info("Agent 4c: Enhancing projects...")
    major_projs = resume.projects[:3]
    out.enhanced_projects = [agent_enhance_project(p, jd, missing) for p in major_projs]
    for p in resume.projects[3:6]:
        out.enhanced_projects.append(deepcopy(p))

    # Agent 5
    logger.info("Agent 5: Boosting skills for ATS 90+...")
    out.enhanced_skills = agent_boost_skills(resume, jd, missing)

    out.improvement_notes = scores.recommendations[:5]

    # Agent 6
    logger.info("Agent 6: QA validation...")
    out = agent_qa(out, original_contact)

    logger.info("Enhancement pipeline complete")
    return out


def optimize_resume_loop(
    resume: ParsedResume,
    jd: ParsedJD,
    scores: ScoreResult,
    onboarding: Optional[OnboardingData] = None,
    rewrite_all: bool = True,
    progress_callback = None,
    force_one_page: bool = True
) -> tuple[EnhancedResume, list[dict]]:
    """
    ATS Optimization Loop.
    Repeatedly enhances and re-scores the resume until target ATS score is reached
    or max iterations are exhausted, while monitoring authenticity.
    """
    from core.scoring_engine import score_resume
    from core.authenticity_engine import score_authenticity
    from core.compression_agent import compress_resume
    from config import OPTIMIZATION_TARGET_SCORE, OPTIMIZATION_MAX_ITERATIONS, OPTIMIZATION_MIN_AUTHENTICITY
    
    iteration_log = []
    
    # ── Iteration 0: Initial enhancement ──────────────────────────────────────
    if progress_callback:
        progress_callback(0, "Running primary enhancement pass...")
        
    enhanced = enhance_resume(resume, jd, scores, onboarding, rewrite_all)
    
    # Evaluate score
    pr_post = ParsedResume()
    pr_post.name = enhanced.name
    pr_post.email = enhanced.email
    pr_post.phone = enhanced.phone
    pr_post.linkedin = enhanced.linkedin
    pr_post.github = enhanced.github
    pr_post.portfolio = getattr(enhanced, "portfolio", "")
    pr_post.summary = enhanced.enhanced_summary
    pr_post.skills = enhanced.enhanced_skills
    pr_post.experience = enhanced.enhanced_experience
    pr_post.projects = enhanced.enhanced_projects
    pr_post.education = enhanced.education
    pr_post.certifications = enhanced.certifications
    pr_post.accomplishments = enhanced.accomplishments
    pr_post.publications = enhanced.publications
    
    sections = ["experience", "skills", "projects", "education", "summary"]
    if enhanced.certifications: sections.append("certifications")
    if enhanced.accomplishments: sections.append("accomplishments")
    if enhanced.publications: sections.append("publications")
    pr_post.sections_found = sections
    pr_post.raw_text = (
        " ".join(enhanced.enhanced_skills) + " " +
        enhanced.enhanced_summary + " " +
        " ".join(b for e in enhanced.enhanced_experience for b in e.bullets)
    )
    score_post = score_resume(pr_post, jd)
    
    # Evaluate authenticity
    orig_bullets = [b for exp in resume.experience for b in exp.bullets]
    enh_bullets = [b for exp in enhanced.enhanced_experience for b in exp.bullets]
    orig_metrics = resume.metrics_found
    orig_skills = resume.skills
    enh_skills = enhanced.enhanced_skills
    allowed_tech = set(jd.all_keywords)
    
    auth_score = score_authenticity(
        orig_bullets, enh_bullets, orig_metrics, 0, orig_skills, enh_skills, allowed_tech
    )
    
    iteration_log.append({
        "iteration": 1,
        "ats_score": score_post.overall_score * 100,
        "authenticity_score": auth_score.authenticity_score,
        "trust_level": auth_score.recruiter_trust,
        "msg": "Primary enhancement pass completed."
    })
    
    logger.info(f"Loop Iteration 1 | Score: {score_post.overall_score:.2%} | Auth: {auth_score.authenticity_score:.1f}%")
    
    # ── Loop optimization passes ─────────────────────────────────────────────
    best_enhanced = deepcopy(enhanced)
    best_score = score_post.overall_score
    
    it = 1
    while score_post.overall_score < OPTIMIZATION_TARGET_SCORE and it < OPTIMIZATION_MAX_ITERATIONS:
        it += 1
        if progress_callback:
            progress_callback(it, f"Optimization pass {it}: Target {OPTIMIZATION_TARGET_SCORE*100:.0f}%, Current {score_post.overall_score*100:.1f}%...")
            
        # Target remaining gaps: focus on missing skills and keyword injection
        remaining_missing = score_post.missing_skills
        if not remaining_missing:
            logger.info("No missing keywords left to optimize.")
            break
            
        # Step A: Inject high-priority missing skills
        new_skills = agent_boost_skills(pr_post, jd, remaining_missing)
        enhanced.enhanced_skills = list(dict.fromkeys(enhanced.enhanced_skills + new_skills))[:50]
        
        # Step B: Select weakest bullet points and rewrite them to incorporate missing keywords
        weakest_bullets = score_post.weak_bullets
        if weakest_bullets:
            if enhanced.enhanced_experience:
                first_exp = enhanced.enhanced_experience[0]
                first_exp.bullets = agent_rewrite_bullets(
                    first_exp.bullets, first_exp.title, first_exp.company, jd, remaining_missing,
                    max_bullets=len(first_exp.bullets)
                )
                
        # Re-score after modifications
        pr_post.skills = enhanced.enhanced_skills
        pr_post.experience = enhanced.enhanced_experience
        pr_post.raw_text = (
            " ".join(enhanced.enhanced_skills) + " " +
            enhanced.enhanced_summary + " " +
            " ".join(b for e in enhanced.enhanced_experience for b in e.bullets)
        )
        new_score_post = score_resume(pr_post, jd)
        
        # Re-evaluate authenticity
        new_enh_bullets = [b for exp in enhanced.enhanced_experience for b in exp.bullets]
        new_auth_score = score_authenticity(
            orig_bullets, new_enh_bullets, orig_metrics, 0, orig_skills, enhanced.enhanced_skills, allowed_tech
        )
        
        logger.info(f"Loop Iteration {it} | Score: {new_score_post.overall_score:.2%} | Auth: {new_auth_score.authenticity_score:.1f}%")
        
        # Check authenticity safety threshold
        if new_auth_score.authenticity_score < OPTIMIZATION_MIN_AUTHENTICITY or new_auth_score.recruiter_trust == "Low":
            logger.warning(f"Optimization halted at iteration {it}: Authenticity ({new_auth_score.authenticity_score:.1f}%) fell below limit.")
            iteration_log.append({
                "iteration": it,
                "ats_score": new_score_post.overall_score * 100,
                "authenticity_score": new_auth_score.authenticity_score,
                "trust_level": new_auth_score.recruiter_trust,
                "msg": f"Halted: Authenticity score dropped below {OPTIMIZATION_MIN_AUTHENTICITY}%. Rolled back to iteration {it-1}."
            })
            # Roll back to the previous best stable version and stop
            enhanced = deepcopy(best_enhanced)
            break
            
        # Update best tracking
        if new_score_post.overall_score > best_score:
            best_score = new_score_post.overall_score
            best_enhanced = deepcopy(enhanced)
            
        score_post = new_score_post
        auth_score = new_auth_score
        
        iteration_log.append({
            "iteration": it,
            "ats_score": score_post.overall_score * 100,
            "authenticity_score": auth_score.authenticity_score,
            "trust_level": auth_score.recruiter_trust,
            "msg": f"Optimization pass {it} completed successfully."
        })
        
    # ── Step C: Run One-Page Compression Agent on the optimized content ──────
    if progress_callback:
        progress_callback(it + 1, "Enforcing single-page constraints...")
        
    final_compressed = compress_resume(best_enhanced, jd, force_one_page=force_one_page)
    
    return final_compressed, iteration_log

