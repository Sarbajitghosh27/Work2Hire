"""
core/dataset_logger.py — v1
────────────────────────────
Upgrade 10: Logging & Dataset Collection

Logs every generation run as a JSONL record for future fine-tuning.
Stores: original resume, generated resume, ATS delta, hallucination
violations, accepted/rejected bullets, authenticity scores.

Records are stored in logs/generation_log.jsonl
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

from config import LOGGING_ENABLED, LOG_DIR

logger = logging.getLogger(__name__)

os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "generation_log.jsonl")


# ──────────────────────────────────────────────────────────────────────────────
# Data model for a single run
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationRecord:
    run_id                  : str   = ""
    timestamp               : str   = ""
    resume_hash             : str   = ""  # SHA256 of original raw_text (no PII)

    # Input metadata
    original_ats_score      : float = 0.0
    original_skill_count    : int   = 0
    original_bullet_count   : int   = 0
    original_word_count     : int   = 0
    jd_domain               : str   = ""
    jd_seniority            : str   = ""
    jd_keyword_count        : int   = 0

    # Output metadata
    enhanced_ats_score      : float = 0.0
    ats_delta               : float = 0.0
    enhanced_skill_count    : int   = 0
    enhanced_bullet_count   : int   = 0

    # Quality metrics
    authenticity_score      : float = 0.0
    hallucination_risk      : str   = ""
    recruiter_trust         : str   = ""
    hallucination_violations: int   = 0
    metric_preservation_pct : float = 0.0
    preserved_bullets_pct   : float = 0.0
    rewritten_bullets_pct   : float = 0.0
    confidence_score        : float = 0.0
    rewrite_aggressiveness  : str   = ""

    # Bullet-level data (anonymised — no personal content)
    bullets_preserved       : int   = 0
    bullets_rewritten       : int   = 0
    bullets_full_rewrite    : int   = 0
    bullets_light_rewrite   : int   = 0

    # Violations log
    violation_details       : List[str] = field(default_factory=list)

    # Model info
    llm_model               : str   = ""
    llm_temperature         : float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

def log_generation(record: GenerationRecord) -> None:
    """Append a GenerationRecord to the JSONL log file."""
    if not LOGGING_ENABLED:
        return

    try:
        record_dict = asdict(record)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict) + "\n")
        logger.info(f"Logged run {record.run_id} | ATS delta: +{record.ats_delta:.1f} | Auth: {record.authenticity_score:.0f}%")
    except Exception as e:
        logger.warning(f"Dataset logging failed (non-fatal): {e}")


def build_generation_record(
    raw_text: str,
    original_score: float,
    enhanced_score: Optional[float],
    original_skill_count: int,
    original_bullet_count: int,
    original_word_count: int,
    jd_domain: str,
    jd_seniority: str,
    jd_keyword_count: int,
    enhanced_skill_count: int,
    enhanced_bullet_count: int,
    authenticity_score: float,
    hallucination_risk: str,
    recruiter_trust: str,
    hallucination_violations: int,
    metric_preservation_pct: float,
    preserved_bullets_pct: float,
    rewritten_bullets_pct: float,
    confidence_score: float,
    rewrite_aggressiveness: str,
    bullets_preserved: int,
    bullets_rewritten: int,
    bullets_full_rewrite: int,
    bullets_light_rewrite: int,
    violation_details: List[str],
    llm_model: str,
    llm_temperature: float,
) -> GenerationRecord:
    """Build a complete GenerationRecord from pipeline outputs."""
    resume_hash = hashlib.sha256(raw_text.encode("utf-8", errors="ignore")).hexdigest()[:16]

    return GenerationRecord(
        run_id                  = f"{resume_hash[:8]}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        timestamp               = datetime.utcnow().isoformat(),
        resume_hash             = resume_hash,
        original_ats_score      = round(original_score, 1),
        original_skill_count    = original_skill_count,
        original_bullet_count   = original_bullet_count,
        original_word_count     = original_word_count,
        jd_domain               = jd_domain,
        jd_seniority            = jd_seniority,
        jd_keyword_count        = jd_keyword_count,
        enhanced_ats_score      = round(enhanced_score or 0.0, 1),
        ats_delta               = round((enhanced_score or original_score) - original_score, 1),
        enhanced_skill_count    = enhanced_skill_count,
        enhanced_bullet_count   = enhanced_bullet_count,
        authenticity_score      = round(authenticity_score, 1),
        hallucination_risk      = hallucination_risk,
        recruiter_trust         = recruiter_trust,
        hallucination_violations= hallucination_violations,
        metric_preservation_pct = round(metric_preservation_pct, 1),
        preserved_bullets_pct   = round(preserved_bullets_pct, 1),
        rewritten_bullets_pct   = round(rewritten_bullets_pct, 1),
        confidence_score        = round(confidence_score, 1),
        rewrite_aggressiveness  = rewrite_aggressiveness,
        bullets_preserved       = bullets_preserved,
        bullets_rewritten       = bullets_rewritten,
        bullets_full_rewrite    = bullets_full_rewrite,
        bullets_light_rewrite   = bullets_light_rewrite,
        violation_details       = violation_details[:10],  # cap
        llm_model               = llm_model,
        llm_temperature         = llm_temperature,
    )


def get_log_stats() -> Dict:
    """Read log file and return summary stats. Used in admin/debug view."""
    if not os.path.exists(LOG_PATH):
        return {"total_runs": 0}

    records = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        return {"error": str(e)}

    if not records:
        return {"total_runs": 0}

    ats_deltas = [r["ats_delta"] for r in records if "ats_delta" in r]
    auth_scores = [r["authenticity_score"] for r in records if "authenticity_score" in r]
    halluc = [r["hallucination_violations"] for r in records if "hallucination_violations" in r]
    domains = [r["jd_domain"] for r in records if "jd_domain" in r]

    return {
        "total_runs"           : len(records),
        "avg_ats_delta"        : round(sum(ats_deltas) / len(ats_deltas), 1) if ats_deltas else 0,
        "avg_authenticity"     : round(sum(auth_scores) / len(auth_scores), 1) if auth_scores else 0,
        "avg_hallucinations"   : round(sum(halluc) / len(halluc), 2) if halluc else 0,
        "hallucination_rate_pct": round(sum(1 for h in halluc if h > 0) / len(halluc) * 100, 1) if halluc else 0,
        "top_domains"          : list({d: domains.count(d) for d in set(domains)}.items())[:5],
        "log_path"             : LOG_PATH,
    }
