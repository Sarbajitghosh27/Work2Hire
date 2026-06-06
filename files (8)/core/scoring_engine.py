"""
core/scoring_engine.py
────────────────────────────────────────────────────────
Module 3 — Semantic Matching Engine + ATS Scoring Engine

Computes 5 scores using a hybrid approach:
  1. ATS Keyword Score    — BM25 keyword overlap
  2. Semantic Match Score — Sentence-BERT cosine similarity
  3. Technical Depth      — tech keyword density & coverage
  4. Recruiter Readability— formatting, structure, action verbs
  5. Achievement Impact   — quantified metrics presence

Also identifies:
  - Missing skills (gap analysis)
  - Matched skills
  - Weak bullet points that need rewriting
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from config import SCORE_WEIGHTS, EMBEDDING_MODEL
from core.resume_parser import ParsedResume
from core.jd_engine import ParsedJD

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy embedding model loader
# ──────────────────────────────────────────────────────────────────────────────

_embedder = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ──────────────────────────────────────────────────────────────────────────────
# Action verb list (strong verbs improve readability score)
# ──────────────────────────────────────────────────────────────────────────────

STRONG_VERBS = {
    "built","developed","designed","implemented","optimized","reduced",
    "improved","increased","led","created","launched","deployed","automated",
    "architected","engineered","scaled","integrated","delivered","achieved",
    "accelerated","collaborated","mentored","migrated","refactored",
    "researched","published","contributed","resolved","streamlined","established",
}

WEAK_VERBS = {
    "worked","helped","assisted","involved","participated","responsible",
    "handled","used","did","made","was part of","tried","attempted",
}

METRIC_RE = re.compile(
    r"(\d+[\.,]?\d*\s*(%|x|X|\+|k|K|M|B|ms|sec|hrs?|days?|"
    r"users?|customers?|requests?|transactions?))",
    re.I,
)


# ──────────────────────────────────────────────────────────────────────────────
# Result data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    ats_keyword_score      : float = 0.0
    semantic_score         : float = 0.0
    technical_depth_score  : float = 0.0
    readability_score      : float = 0.0
    achievement_score      : float = 0.0
    overall_score          : float = 0.0

    matched_skills         : List[str] = field(default_factory=list)
    missing_skills         : List[str] = field(default_factory=list)
    weak_bullets           : List[str] = field(default_factory=list)
    strong_bullets         : List[str] = field(default_factory=list)
    recommendations        : List[str] = field(default_factory=list)
    skill_coverage_pct     : float = 0.0

    def to_dict(self) -> dict:
        return {
            "ATS Keyword Score"       : round(self.ats_keyword_score * 100, 1),
            "Semantic Match"          : round(self.semantic_score * 100, 1),
            "Technical Depth"         : round(self.technical_depth_score * 100, 1),
            "Recruiter Readability"   : round(self.readability_score * 100, 1),
            "Achievement Impact"      : round(self.achievement_score * 100, 1),
            "Overall Score"           : round(self.overall_score * 100, 1),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Score 1: ATS keyword score (BM25)
# ──────────────────────────────────────────────────────────────────────────────

def compute_ats_score(resume: ParsedResume, jd: ParsedJD) -> Tuple[float, List[str], List[str]]:
    """
    Tokenise resume and score against JD keywords using BM25.
    Returns (score_0_1, matched_keywords, missing_keywords).
    """
    jd_keywords = [kw.lower() for kw in jd.all_keywords]
    if not jd_keywords:
        return 0.5, [], []

    resume_text_lower = resume.raw_text.lower()
    resume_tokens = re.findall(r"\b\w+\b", resume_text_lower)

    matched = [kw for kw in jd_keywords if kw in resume_text_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_text_lower]

    # BM25 scoring
    corpus = [resume_tokens]
    bm25   = BM25Okapi(corpus)
    query  = re.findall(r"\b\w+\b", " ".join(jd_keywords))
    raw_scores = bm25.get_scores(query)

    # Normalise: coverage ratio weighted with BM25
    coverage = len(matched) / max(len(jd_keywords), 1)
    bm25_norm = min(raw_scores[0] / 50.0, 1.0) if raw_scores[0] > 0 else 0.0
    ats_score = 0.6 * coverage + 0.4 * bm25_norm

    return min(ats_score, 1.0), matched, missing


# ──────────────────────────────────────────────────────────────────────────────
# Score 2: Semantic similarity (Sentence-BERT)
# ──────────────────────────────────────────────────────────────────────────────

def compute_semantic_score(resume: ParsedResume, jd: ParsedJD) -> float:
    """
    Embed resume text and JD text, compute cosine similarity.
    Uses chunking to handle long resumes.
    """
    embedder = get_embedder()

    # Chunk resume (experience + skills + projects)
    resume_chunks = []
    for exp in resume.experience:
        resume_chunks.append(f"{exp.title} at {exp.company}: " + " ".join(exp.bullets))
    resume_chunks.append("Skills: " + ", ".join(resume.skills[:40]))
    for proj in resume.projects:
        resume_chunks.append(f"Project {proj.name}: {proj.description}")

    if not resume_chunks:
        resume_chunks = [resume.raw_text[:1000]]

    # JD chunks
    jd_chunks = [
        jd.role_title + " " + jd.domain,
        "Required: " + ", ".join(jd.required_skills[:20]),
        " ".join(jd.responsibilities[:5]),
    ]
    jd_chunks = [c for c in jd_chunks if c.strip()]

    # Embed
    resume_embeddings = embedder.encode(resume_chunks, convert_to_numpy=True)
    jd_embeddings     = embedder.encode(jd_chunks, convert_to_numpy=True)

    # Mean pool
    resume_vec = resume_embeddings.mean(axis=0, keepdims=True)
    jd_vec     = jd_embeddings.mean(axis=0, keepdims=True)

    sim = cosine_similarity(resume_vec, jd_vec)[0][0]
    return float(np.clip(sim, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Score 3: Technical depth
# ──────────────────────────────────────────────────────────────────────────────

def compute_technical_depth(resume: ParsedResume, jd: ParsedJD) -> float:
    """
    Measures how technically aligned the resume is.
    Factors: JD tech coverage, project tech density, keyword variety.
    """
    jd_tech  = set(jd.tech_stack)
    res_tech = set(resume.skills)

    if not jd_tech:
        return 0.5

    # Coverage of JD tech stack
    coverage = len(jd_tech & res_tech) / len(jd_tech)

    # Project tech richness
    total_proj_tech = sum(len(p.tech_used) for p in resume.projects)
    proj_density = min(total_proj_tech / max(len(resume.projects) * 3, 1), 1.0)

    # Raw skill count bonus
    skill_variety = min(len(resume.skills) / 30, 1.0)

    return 0.5 * coverage + 0.3 * proj_density + 0.2 * skill_variety


# ──────────────────────────────────────────────────────────────────────────────
# Score 4: Recruiter readability
# ──────────────────────────────────────────────────────────────────────────────

def compute_readability_score(resume: ParsedResume) -> Tuple[float, List[str], List[str]]:
    """
    Checks for:
    - Action verb usage in bullet points
    - Bullet structure quality
    - Section completeness
    - Word count in sweet spot
    """
    all_bullets = []
    for exp in resume.experience:
        all_bullets.extend(exp.bullets)
    for proj in resume.projects:
        if proj.description:
            all_bullets.append(proj.description)

    weak   = []
    strong = []
    score  = 0.0
    factors = []

    # Factor 1: Action verb quality
    if all_bullets:
        strong_count = 0
        for bullet in all_bullets:
            first_word = bullet.split()[0].lower().rstrip("ed") if bullet.split() else ""
            if any(verb.startswith(first_word) for verb in STRONG_VERBS):
                strong_count += 1
                strong.append(bullet)
            elif any(verb in bullet.lower()[:30] for verb in WEAK_VERBS):
                weak.append(bullet)
        verb_score = strong_count / max(len(all_bullets), 1)
        factors.append(("action_verbs", verb_score))
    else:
        factors.append(("action_verbs", 0.3))

    # Factor 2: Section completeness
    required_sections = {"experience", "skills", "education", "projects"}
    found_sections    = set(resume.sections_found)
    section_score     = len(required_sections & found_sections) / len(required_sections)
    factors.append(("sections", section_score))

    # Factor 3: Word count  (550–800 is ideal for 1-page resume)
    wc = resume.word_count
    if 400 <= wc <= 900:
        wc_score = 1.0
    elif wc < 400:
        wc_score = wc / 400
    else:
        wc_score = max(0.5, 1.0 - (wc - 900) / 900)
    factors.append(("word_count", wc_score))

    # Factor 4: Contact completeness
    contact_score  = sum([
        bool(resume.email), bool(resume.phone),
        bool(resume.linkedin or resume.github or getattr(resume, "portfolio", "")),
    ]) / 3
    factors.append(("contact", contact_score))

    # Weighted average
    weights = {"action_verbs": 0.4, "sections": 0.3, "word_count": 0.2, "contact": 0.1}
    score   = sum(weights[k] * v for k, v in factors)

    return min(score, 1.0), weak, strong


# ──────────────────────────────────────────────────────────────────────────────
# Score 5: Achievement impact
# ──────────────────────────────────────────────────────────────────────────────

def compute_achievement_score(resume: ParsedResume) -> float:
    """
    Measures quantification of achievements.
    More metrics = higher score.
    """
    all_bullets = []
    for exp in resume.experience:
        all_bullets.extend(exp.bullets)

    if not all_bullets:
        return 0.2

    metric_bullets = [b for b in all_bullets if METRIC_RE.search(b)]
    ratio = len(metric_bullets) / len(all_bullets)

    # Also check for standalone metrics found
    metric_bonus = min(len(resume.metrics_found) / 10, 0.3)

    return min(ratio + metric_bonus, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Recommendations generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    result: ScoreResult,
    resume: ParsedResume,
    jd: ParsedJD,
) -> List[str]:
    recs = []

    if result.ats_keyword_score < 0.6:
        top_missing = result.missing_skills[:5]
        recs.append(f"Add these missing ATS keywords: {', '.join(top_missing)}")

    if result.semantic_score < 0.5:
        recs.append(f"Tailor your summary and experience descriptions more closely to the {jd.domain} role")

    if result.technical_depth_score < 0.5:
        recs.append("Add more technical projects with specific technologies used and outcomes")

    if result.readability_score < 0.6:
        if result.weak_bullets:
            recs.append(f"Replace weak verbs ('worked on', 'helped with') with action verbs ('{', '.join(list(STRONG_VERBS)[:4])}')")
        if not resume.email or not resume.phone:
            recs.append("Ensure contact information (email, phone, LinkedIn) is clearly visible")

    if result.achievement_score < 0.4:
        recs.append("Quantify more achievements: add numbers, percentages, or scale metrics to at least 40% of bullets")

    if jd.seniority in ("senior", "lead", "staff") and len(resume.experience) < 2:
        recs.append(f"This is a {jd.seniority}-level role — emphasise leadership, architecture decisions, and mentorship")

    if result.missing_skills:
        recs.append(f"Consider adding a certification or side project demonstrating: {', '.join(result.missing_skills[:3])}")

    return recs


# ──────────────────────────────────────────────────────────────────────────────
# Master scoring function
# ──────────────────────────────────────────────────────────────────────────────

def score_resume(resume: ParsedResume, jd: ParsedJD) -> ScoreResult:
    """
    Runs all 5 scoring modules and returns a ScoreResult.
    """
    result = ScoreResult()

    logger.info("Computing ATS keyword score...")
    result.ats_keyword_score, result.matched_skills, result.missing_skills = \
        compute_ats_score(resume, jd)

    logger.info("Computing semantic similarity score...")
    result.semantic_score = compute_semantic_score(resume, jd)

    logger.info("Computing technical depth score...")
    result.technical_depth_score = compute_technical_depth(resume, jd)

    logger.info("Computing readability score...")
    result.readability_score, result.weak_bullets, result.strong_bullets = \
        compute_readability_score(resume)

    logger.info("Computing achievement impact score...")
    result.achievement_score = compute_achievement_score(resume)

    # Weighted overall score
    w = SCORE_WEIGHTS
    result.overall_score = (
        w["ats_keyword"]     * result.ats_keyword_score     +
        w["semantic_match"]  * result.semantic_score         +
        w["technical_depth"] * result.technical_depth_score  +
        w["readability"]     * result.readability_score      +
        w["achievement"]     * result.achievement_score
    )

    result.skill_coverage_pct = (
        len(result.matched_skills) / max(len(jd.all_keywords), 1) * 100
    )

    result.recommendations = generate_recommendations(result, resume, jd)

    logger.info(f"Overall score: {result.overall_score:.2%}")
    return result
