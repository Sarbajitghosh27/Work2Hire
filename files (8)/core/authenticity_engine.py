"""
core/authenticity_engine.py — v1
──────────────────────────────────
Upgrade 2: Resume Authenticity Scoring
Upgrade 3: Recruiter Realism Layer
Upgrade 9: Confidence Scoring System

Provides:
  - AuthenticityScore   dataclass
  - ConfidenceScore     dataclass
  - score_authenticity()
  - score_confidence()
  - realism_filter()
  - filter_ai_phrases()
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from config import REALISM_FILTER, REALISM_BANNED_PHRASES, CONFIDENCE_SCORING

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Realism — banned phrase patterns (Upgrade 3)
# ──────────────────────────────────────────────────────────────────────────────

_REALISM_RE = re.compile(
    "|".join(re.escape(p) for p in REALISM_BANNED_PHRASES),
    re.I,
)

# Patterns that indicate genuine, human-written engineering bullets
_ENGINEERING_TONE_RE = re.compile(
    r"\b(implemented|built|designed|deployed|optimized|reduced|improved|"
    r"integrated|migrated|refactored|automated|architected|engineered|"
    r"validated|trained|fine-tuned|debugged|profiled|benchmarked)\b",
    re.I,
)

# Patterns that still sound AI-generated even after fluff filter
_AI_TELL_RE = re.compile(
    r"\b(seamlessly|effortlessly|robust|comprehensive|leveraged the power of|"
    r"utilizing cutting|highly scalable|state-of-the-art|next-generation|"
    r"revolutionized|transformative|impactful solution|end-to-end solution)\b",
    re.I,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AuthenticityScore:
    """Resume authenticity / recruiter trust metrics."""
    authenticity_score      : float = 0.0   # 0-100
    hallucination_risk      : str   = ""    # "Low" / "Medium" / "High"
    recruiter_trust         : str   = ""    # "High" / "Medium" / "Low"

    preserved_bullets_pct   : float = 0.0
    rewritten_bullets_pct   : float = 0.0
    hallucination_violations : int  = 0
    tech_consistency_score  : float = 0.0
    metric_preservation_pct : float = 0.0

    ai_phrase_count         : int   = 0
    engineering_tone_score  : float = 0.0

    details                 : List[str] = field(default_factory=list)


@dataclass
class ConfidenceScore:
    """Pipeline generation confidence estimate."""
    overall_confidence      : float = 0.0   # 0-100
    rewrite_aggressiveness  : str   = ""    # "Conservative" / "Moderate" / "Aggressive"
    hallucination_probability: float = 0.0  # 0-100

    ats_improvement_estimate: float = 0.0   # Expected delta
    data_richness           : float = 0.0   # How much input data we had (0-100)
    details                 : List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Realism filter (Upgrade 3)
# ──────────────────────────────────────────────────────────────────────────────

def filter_ai_phrases(text: str) -> tuple[str, int]:
    """
    Remove AI-sounding phrases from generated text.
    Returns (cleaned_text, number_of_removals).
    """
    if not REALISM_FILTER:
        return text, 0

    count = len(_REALISM_RE.findall(text))
    count += len(_AI_TELL_RE.findall(text))

    text = _REALISM_RE.sub("", text)
    text = _AI_TELL_RE.sub("", text)

    # Clean up double spaces left by removal
    text = re.sub(r"  +", " ", text).strip()
    # Clean up leading commas or conjunctions
    text = re.sub(r"^[,\s]+", "", text)

    return text, count


def realism_filter_bullets(bullets: List[str]) -> List[str]:
    """
    Apply realism filter to a list of bullets.
    Prefers engineering-focused, concise bullets.
    """
    cleaned = []
    for bullet in bullets:
        text, n = filter_ai_phrases(bullet)
        if text and len(text) > 10:
            cleaned.append(text)
        elif bullet and len(bullet) > 10:
            cleaned.append(bullet)  # fallback to original if filter destroyed it
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Authenticity scorer (Upgrade 2)
# ──────────────────────────────────────────────────────────────────────────────

def score_authenticity(
    original_bullets: List[str],
    enhanced_bullets: List[str],
    original_metrics: List[str],
    hallucination_violations: int,
    original_skills: List[str],
    enhanced_skills: List[str],
    allowed_tech: set,
) -> AuthenticityScore:
    """
    Computes an authenticity score measuring how trustworthy/human the
    enhanced resume looks vs the original.

    Factors:
      - Preserved bullets %             (higher = more authentic)
      - Metric preservation %           (must be 100% for trust)
      - Hallucination violations        (0 = trustworthy)
      - Tech consistency                (enhanced skills ⊆ allowed_tech)
      - AI phrase count                 (0 = clean)
      - Engineering tone               (action-verb density)
    """
    result = AuthenticityScore()

    # ── Factor 1: Bullet preservation ─────────────────────────────────────────
    orig_set = set(b.strip().lower() for b in original_bullets if b)
    enh_set  = set(b.strip().lower() for b in enhanced_bullets if b)
    total    = len(original_bullets) if original_bullets else 1

    if orig_set and enh_set:
        preserved = len(orig_set & enh_set)
        rewritten = len(enh_set - orig_set)
        result.preserved_bullets_pct = round(preserved / max(total, 1) * 100, 1)
        result.rewritten_bullets_pct = round(rewritten / max(total, 1) * 100, 1)
    else:
        result.preserved_bullets_pct = 100.0
        result.rewritten_bullets_pct = 0.0

    # ── Factor 2: Metric preservation ─────────────────────────────────────────
    _metric_re = re.compile(
        r"\b(\d+[\.,]?\d*\s*(%|x|X|\+|k|K|M|B|ms|sec|hrs?|days?|users?|requests?))\b",
        re.I,
    )
    enhanced_text = " ".join(enhanced_bullets)
    enhanced_metrics = set(m[0] for m in _metric_re.findall(enhanced_text))
    original_metrics_set = set(original_metrics)

    if original_metrics_set:
        preserved_m = enhanced_metrics & original_metrics_set
        result.metric_preservation_pct = round(
            len(preserved_m) / len(original_metrics_set) * 100, 1
        )
    else:
        result.metric_preservation_pct = 100.0  # nothing to preserve

    # Check for invented metrics
    invented = enhanced_metrics - original_metrics_set
    if invented:
        result.hallucination_violations += len(invented)
        result.details.append(f"Invented metrics detected: {list(invented)[:3]}")

    # ── Factor 3: Hallucination violations ────────────────────────────────────
    result.hallucination_violations += hallucination_violations

    # ── Factor 4: Tech consistency ────────────────────────────────────────────
    if enhanced_skills and allowed_tech:
        valid_tech = [s for s in enhanced_skills if s.lower() in allowed_tech]
        result.tech_consistency_score = round(len(valid_tech) / max(len(enhanced_skills), 1) * 100, 1)
    else:
        result.tech_consistency_score = 100.0

    # ── Factor 5: AI phrase detection ─────────────────────────────────────────
    all_text = " ".join(enhanced_bullets)
    _, ai_count = filter_ai_phrases(all_text)
    result.ai_phrase_count = ai_count

    # ── Factor 6: Engineering tone ────────────────────────────────────────────
    if enhanced_bullets:
        tone_bullets = [b for b in enhanced_bullets if _ENGINEERING_TONE_RE.search(b)]
        result.engineering_tone_score = round(
            len(tone_bullets) / max(len(enhanced_bullets), 1) * 100, 1
        )
    else:
        result.engineering_tone_score = 0.0

    # ── Composite authenticity score ──────────────────────────────────────────
    score = 0.0
    score += 0.20 * (result.preserved_bullets_pct / 100)      # preservation
    score += 0.25 * (result.metric_preservation_pct / 100)    # metric integrity
    score += 0.20 * max(0, (1 - result.hallucination_violations / 5))  # no hallucinations
    score += 0.15 * (result.tech_consistency_score / 100)     # tech honesty
    score += 0.10 * max(0, (1 - result.ai_phrase_count / 10)) # no AI phrases
    score += 0.10 * (result.engineering_tone_score / 100)     # sounds human

    result.authenticity_score = round(min(score * 100, 100), 1)

    # ── Categorise risk / trust ────────────────────────────────────────────────
    viol = result.hallucination_violations
    if viol == 0 and result.metric_preservation_pct >= 95:
        result.hallucination_risk = "Low"
    elif viol <= 2:
        result.hallucination_risk = "Medium"
    else:
        result.hallucination_risk = "High"
        result.details.append("Multiple hallucination violations — manual review recommended")

    if result.authenticity_score >= 85 and result.hallucination_risk == "Low":
        result.recruiter_trust = "High"
    elif result.authenticity_score >= 65:
        result.recruiter_trust = "Medium"
    else:
        result.recruiter_trust = "Low"
        result.details.append("Authenticity score below threshold — consider more conservative rewriting")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Confidence scorer (Upgrade 9)
# ──────────────────────────────────────────────────────────────────────────────

def score_confidence(
    original_score: float,          # 0-100
    enhanced_score: Optional[float],# 0-100
    num_experience: int,
    num_projects: int,
    num_skills: int,
    hallucination_violations: int,
    bullets_total: int,
    bullets_rewritten: int,
    authenticity: AuthenticityScore,
) -> ConfidenceScore:
    """
    Estimates confidence in the pipeline's output quality.
    """
    result = ConfidenceScore()

    # ── Data richness — how much we had to work with ───────────────────────────
    richness = 0.0
    richness += min(num_experience / 3, 1.0) * 30    # 3+ roles = full credit
    richness += min(num_projects / 3, 1.0) * 25      # 3+ projects
    richness += min(num_skills / 15, 1.0) * 20       # 15+ skills
    richness += min(bullets_total / 8, 1.0) * 25     # 8+ bullets
    result.data_richness = round(richness, 1)

    # ── Hallucination probability ──────────────────────────────────────────────
    base_risk = min(hallucination_violations / 5, 1.0) * 60
    rewrite_risk = (bullets_rewritten / max(bullets_total, 1)) * 20
    result.hallucination_probability = round(min(base_risk + rewrite_risk, 100), 1)

    # ── Rewrite aggressiveness ─────────────────────────────────────────────────
    pct = bullets_rewritten / max(bullets_total, 1)
    if pct < 0.3:
        result.rewrite_aggressiveness = "Conservative"
    elif pct < 0.6:
        result.rewrite_aggressiveness = "Moderate"
    else:
        result.rewrite_aggressiveness = "Aggressive"

    # ── ATS improvement estimate ───────────────────────────────────────────────
    if enhanced_score is not None and original_score is not None:
        result.ats_improvement_estimate = round(enhanced_score - original_score, 1)
    else:
        # Estimate based on how many missing JD keywords we could've added
        result.ats_improvement_estimate = round(result.data_richness * 0.15, 1)

    # ── Overall confidence ────────────────────────────────────────────────────
    conf = 0.0
    conf += (result.data_richness / 100) * 30
    conf += max(0, (1 - result.hallucination_probability / 100)) * 35
    conf += (authenticity.authenticity_score / 100) * 25
    conf += (authenticity.tech_consistency_score / 100) * 10

    result.overall_confidence = round(min(conf * 100 / 100 * 100, 100), 1)  # normalize

    # Clamp to 0-100 (the multiplication above is redundant but explicit)
    result.overall_confidence = round(min(conf * 100, 100), 1)

    result.details.append(
        f"Data richness: {result.data_richness:.0f}% | "
        f"Hallucination risk: {result.hallucination_probability:.0f}% | "
        f"Rewriting: {result.rewrite_aggressiveness}"
    )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Format helpers for Streamlit display
# ──────────────────────────────────────────────────────────────────────────────

def format_authenticity_badge(auth: AuthenticityScore) -> dict:
    """Returns a dict of display-ready strings for the UI."""
    risk_color = {"Low": "#4ade80", "Medium": "#facc15", "High": "#f87171"}.get(
        auth.hallucination_risk, "#94a3b8"
    )
    trust_color = {"High": "#4ade80", "Medium": "#facc15", "Low": "#f87171"}.get(
        auth.recruiter_trust, "#94a3b8"
    )
    return {
        "authenticity": f"{auth.authenticity_score:.0f}%",
        "hallucination_risk": auth.hallucination_risk,
        "recruiter_trust": auth.recruiter_trust,
        "metric_pct": f"{auth.metric_preservation_pct:.0f}%",
        "preserved_pct": f"{auth.preserved_bullets_pct:.0f}%",
        "rewritten_pct": f"{auth.rewritten_bullets_pct:.0f}%",
        "tech_consistency": f"{auth.tech_consistency_score:.0f}%",
        "risk_color": risk_color,
        "trust_color": trust_color,
    }


def format_confidence_badge(conf: ConfidenceScore) -> dict:
    col = "#4ade80" if conf.overall_confidence >= 80 else (
          "#facc15" if conf.overall_confidence >= 60 else "#f87171")
    return {
        "confidence": f"{conf.overall_confidence:.0f}%",
        "aggressiveness": conf.rewrite_aggressiveness,
        "hallucination_prob": f"{conf.hallucination_probability:.0f}%",
        "ats_delta": f"+{conf.ats_improvement_estimate:.1f}" if conf.ats_improvement_estimate >= 0 else f"{conf.ats_improvement_estimate:.1f}",
        "color": col,
    }
