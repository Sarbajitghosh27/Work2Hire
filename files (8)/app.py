"""
app.py — Rozgar24x7 AI Resume Builder
─────────────────────────────────────────────────────────────────────────────
Work2Hire-style two-panel layout.

LLM Backends (configured in config.py — zero external API, zero tokenization):
  ● Ollama      — mistral:7b / llama3.1:8b / phi3:mini / gemma2:9b
  ● HuggingFace — any model from HF Hub, loaded locally (4-bit / 8-bit)
  ● Fine-tuned  — your own LoRA/QLoRA adapter or merged model

Run:
  streamlit run app.py
"""

import io
import sys
import logging
import textwrap
from pathlib import Path
from typing import Optional

import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import ROLE_DOMAINS, LLM_BACKEND, OLLAMA_MODEL, HF_MODEL_PATH, canonical_skill_name
from core.resume_parser import parse_resume, ParsedResume, is_valid_certification, is_valid_accomplishment
from core.jd_engine import parse_jd, ParsedJD
from core.scoring_engine import score_resume, ScoreResult
from core.enhancement_engine import enhance_resume, OnboardingData, EnhancedResume
from core.resume_generator import generate_html_resume, html_to_pdf_bytes
from core.llm_backend import get_backend_status, check_ollama, call_llm
from core.background_classifier import classify_background
from core.non_tech_ats_architect import run_ats_architect, ATSArchitectReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Work2Hire — AI Resume Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f7f6f2; }
[data-testid="stHeader"] { display: none; }

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  background: #fff; border-bottom: 1px solid #e8e6e0;
  padding: 0 28px; height: 54px; position: sticky; top: 0; z-index: 999;
}
.topbar-logo { font-size: 20px; font-weight: 700; color: #1a1a1a; }
.topbar-logo span { color: #0ea271; }
.backend-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
  border: 1px solid #e8e6e0;
}
.badge-ollama    { background: #e6f7f1; color: #085041; border-color: #0ea271; }
.badge-hf        { background: #fff0f6; color: #7a2a7a; border-color: #d070d0; }
.badge-finetuned { background: #fff8e6; color: #7a5000; border-color: #e0a020; }

/* Remove default Streamlit block container padding */
.block-container {
  padding-top: 0rem !important;
  padding-bottom: 0rem !important;
  padding-left: 0rem !important;
  padding-right: 0rem !important;
}

div[data-testid="stMainBlockContainer"] {
  padding-top: 0rem !important;
  padding-bottom: 0rem !important;
  padding-left: 0rem !important;
  padding-right: 0rem !important;
}

/* Remove spacing between topbar and columns */
div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] > .stHorizontalBlock {
  margin-top: 0rem !important;
}

/* Style top-level layout columns directly for full height panel layout */
div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] > .stHorizontalBlock > .stColumn:nth-of-type(1) {
  background: #fff !important;
  border-right: 1px solid #e8e6e0 !important;
  padding: 18px 20px !important;
  min-height: 100vh !important;
}

div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] > .stHorizontalBlock > .stColumn:nth-of-type(2) {
  background: #f0eeea !important;
  padding: 18px 20px !important;
  min-height: 100vh !important;
}

.score-card {
  background: #fff; border-radius: 10px; border: 1px solid #e8e6e0;
  padding: 14px 18px; margin-bottom: 12px;
}
.score-number { font-size: 42px; font-weight: 800; color: #0ea271; line-height: 1; }
.score-label  { font-size: 12px; color: #888; margin-top: 2px; }
.score-badge  {
  display: inline-block; padding: 3px 10px; border-radius: 20px;
  font-size: 11px; font-weight: 700; margin-top: 6px;
}
.badge-green  { background: #e6f9f2; color: #0ea271; }
.badge-yellow { background: #fff8e1; color: #f59e0b; }
.badge-red    { background: #fef2f2; color: #ef4444; }

.agent-row {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 0; border-bottom: 1px solid #f3f3f3; font-size: 12px;
}
.agent-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-wait { background: #d1d5db; }
.dot-run  { background: #f59e0b; }
.dot-done { background: #0ea271; }
.dot-err  { background: #ef4444; }

.ollama-ok  { background:#e6f9f2; border:1px solid #0ea271; border-radius:6px; padding:8px 14px; font-size:12px; color:#085041; }
.ollama-err { background:#fef2f2; border:1px solid #ef4444; border-radius:6px; padding:8px 14px; font-size:12px; color:#ef4444; }
.hf-ok      { background:#fff0f6; border:1px solid #d070d0; border-radius:6px; padding:8px 14px; font-size:12px; color:#7a2a7a; }
.ft-ok      { background:#fff8e6; border:1px solid #e0a020; border-radius:6px; padding:8px 14px; font-size:12px; color:#7a5000; }

.resume-wrapper {
  background: #fff; border-radius: 8px; border: 1px solid #ddd;
  overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}

div[data-testid="column"] { padding: 0 6px !important; }
.stButton > button {
  background: #0ea271 !important; color: #fff !important;
  border: none !important; border-radius: 6px !important;
  font-weight: 600 !important; width: 100%;
}
.stButton > button:hover { background: #0b8a60 !important; }
label { font-size: 12px !important; font-weight: 500 !important; }
.stExpander { border: 1px solid #e8e6e0 !important; border-radius: 8px !important; }

.edit-section-wrap {
  border: 1px solid #e8e6e0; border-radius: 8px;
  padding: 14px; margin-bottom: 10px; background: #fff;
}
.edit-section-header {
  font-size: 13px; font-weight: 700; color: #1a1a1a;
  margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
}
.section-icon { font-size: 16px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════════════════════════

_backend_label = {
    "ollama":      f"<span class='backend-badge badge-ollama'>🟢 Ollama — {OLLAMA_MODEL}</span>",
    "huggingface": f"<span class='backend-badge badge-hf'>🟣 HuggingFace — {HF_MODEL_PATH.split('/')[-1]}</span>",
    "finetuned":   f"<span class='backend-badge badge-finetuned'>🟡 Fine-tuned model</span>",
}.get(LLM_BACKEND.lower(), "")

st.markdown(f"""
<div class="topbar">
  <div class="topbar-logo">Work<span>2</span>Hire</div>
  <div>{_backend_label}</div>
  <div style="font-size:12px;color:#aaa">Local LLM · Zero API · Zero Cost</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _blank_html():
    return """<!DOCTYPE html><html><head><style>
body{font-family:'Times New Roman',serif;color:#888;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0;font-size:15px;background:#fafafa;}
</style></head><body>
<div style="text-align:center">
  <div style="font-size:48px;margin-bottom:16px">📄</div>
  <div style="font-weight:600;color:#1a1a1a;margin-bottom:6px">Live Resume Preview</div>
  <div style="color:#aaa;font-size:13px">Upload your CV, paste a JD,<br>then click <strong>Generate & Enhance Resume</strong></div>
</div></body></html>"""

def _init():
    defaults = {
        "parsed_resume":        None,
        "parsed_jd":            None,
        "score_result":         None,
        "score_result_input":   None,
        "enhanced_resume":      None,
        "html_resume":          _blank_html(),
        "agent_log":            [],
        "onboarding":           OnboardingData(),
        "pdf_bytes":            None,
        "edit_mode":            False,
        "edit_resume":          None,
        "sufficiency_report":   None,
        "compatibility_report": None,
        "optimization_log":     [],
        # ATS Architect additions
        "bg_classification":    None,   # result of classify_background()
        "ats_architect_report": None,   # ATSArchitectReport for non-tech CVs
        "da_target_role":       "Data Analyst",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

def recalculate_sufficiency_and_background():
    parsed = st.session_state.parsed_resume
    if not parsed:
        return
        
    bg = classify_background(parsed)
    
    # Determine target domain (selected in selectbox or parsed from JD)
    target_dom = st.session_state.get("jd_domain", "")
    if st.session_state.get("parsed_jd"):
        target_dom = st.session_state.parsed_jd.domain or target_dom
        
    TECH_DOMAINS = {
        "Artificial Intelligence / ML",
        "Data Science / Analytics",
        "Software Engineering (Backend)",
        "Software Engineering (Frontend)",
        "Full Stack Development",
        "DevOps / Cloud Engineering",
        "Embedded Systems / IoT",
        "VLSI / Hardware Engineering",
        "Cybersecurity",
    }
    if target_dom in TECH_DOMAINS:
        bg["is_non_technical"] = False
        bg["background_domain"] = ""
        
    st.session_state.bg_classification = bg
    
    from core.sufficiency_agent import evaluate_sufficiency
    bg_domain = bg["background_domain"] if bg["is_non_technical"] else ""
    st.session_state.sufficiency_report = evaluate_sufficiency(parsed, bg_domain)

def ensure_input_score():
    parsed = st.session_state.parsed_resume
    jd = st.session_state.parsed_jd
    if not parsed or not jd:
        st.session_state.score_result_input = None
        return
    if getattr(st.session_state, "score_result_input", None) is None:
        try:
            st.session_state.score_result_input = score_resume(parsed, jd)
        except Exception as e:
            logger.error(f"Input CV scoring failed: {e}")
            pass

def _log_agent(name, status, msg=""):
    log = st.session_state.agent_log
    for i, (n, s, m) in enumerate(log):
        if n == name:
            log[i] = (name, status, msg)
            return
    log.append((name, status, msg))

def _score_badge(score):
    pct = round(score * 100)
    if pct >= 80:   return f'<span class="score-badge badge-green">ATS Ready ✓</span>'
    elif pct >= 60: return f'<span class="score-badge badge-yellow">Needs Work</span>'
    return f'<span class="score-badge badge-red">Low Match</span>'


def merge_questionnaire_responses(pr: ParsedResume) -> ParsedResume:
    from copy import deepcopy
    from core.resume_parser import WorkExperience, Project
    
    out = deepcopy(pr)
    report = st.session_state.sufficiency_report
    if not report or not report.get("questions"):
        return out
        
    for q in report["questions"]:
        qid = q["id"]
        ans = st.session_state.get(f"q_resp_{qid}", "").strip()
        if not ans:
            continue
            
        sec = q["section"]
        field = q["field"]
        idx = q["item_index"]
        
        if sec == "experience":
            if idx == -1: # general
                out.experience.append(WorkExperience(company="Recent Employer", title="Professional Role", bullets=[ans]))
            else:
                if idx < len(out.experience):
                    if field == "bullets":
                        bullets = [b.strip() for b in ans.split("\n") if b.strip()]
                        out.experience[idx].bullets.extend(bullets)
                    elif field == "metrics":
                        out.experience[idx].bullets.append(ans)
        elif sec == "projects":
            if idx == -1: # general
                import re
                project_name = "Key Project"
                ans_clean = ans.strip()
                
                def clean_project_name(name: str) -> str:
                    name = name.strip()
                    if not name:
                        return "Key Project"
                    # Remove trailing punctuation
                    name = re.sub(r"[.,;:!]+$", "", name).strip()
                    # Capitalize first letter of each word unless it's already uppercase/mixed case
                    words = name.split()
                    capitalized_words = []
                    for w in words:
                        if w.isupper() or any(c.isupper() for c in w[1:]):
                            capitalized_words.append(w)
                        else:
                            capitalized_words.append(w.capitalize())
                    return " ".join(capitalized_words)
                
                # Heuristic 1: Look for common prefixes like "Project Name: [Name]"
                prefix_pattern = re.compile(r"^(project\s+name|project|name)\s*:\s*(.+)$", re.I)
                match = prefix_pattern.match(ans_clean)
                if match:
                    first_part = match.group(2).split(".")[0].split("\n")[0].strip()
                    if first_part:
                        project_name = clean_project_name(first_part)
                else:
                    # Heuristic 2: Check if there's a period or newline early in the text that splits a short title
                    first_sentence = ans_clean.split(".")[0].split("\n")[0].strip()
                    if len(first_sentence) < 40 and "built" not in first_sentence.lower() and "developed" not in first_sentence.lower():
                        project_name = clean_project_name(first_sentence)
                        
                # Heuristic 3: Parse noun phrase from action-verb description (e.g. "Developed an AI-based system for...")
                if project_name == "Key Project":
                    action_pattern = re.compile(
                        r"^(developed|built|created|designed|implemented|worked\s+on|engineered|coded)\s+(a|an|the)?\s*(.+?)\s+(for|using|that|which|to|with|in|to\s+optimize)\b",
                        re.I
                    )
                    action_match = action_pattern.match(ans_clean)
                    if action_match:
                        extracted = action_match.group(3).strip()
                        if extracted:
                            project_name = clean_project_name(extracted)
                
                # If heuristics didn't yield a custom name, use LLM
                if project_name == "Key Project" and len(ans_clean) > 10:
                    try:
                        prompt = f"""You are a resume assistant. Given this project description, generate a short, professional project title (3-5 words).
Examples:
- "Built a movie recommendation system using collaborative filtering" -> "Movie Recommendation System"
- "Developed a full-stack e-commerce web app using React and Node" -> "E-Commerce Web Application"

Description: {ans_clean}
Return ONLY the project title. Do not include quotes, labels, or extra text."""
                        llm_name = call_llm(prompt, max_tokens=20).strip()
                        # Clean LLM response labels if any
                        llm_name = re.sub(r"^(project\s+title|title|project\s+name|name)\s*:\s*", "", llm_name, flags=re.I).strip()
                        llm_name = llm_name.strip('"').strip("'").strip()
                        if llm_name and len(llm_name) > 3:
                            # If it's too long, truncate to first 5 words
                            if len(llm_name) > 60:
                                llm_name = " ".join(llm_name.split()[:5])
                            project_name = clean_project_name(llm_name)
                    except Exception as e:
                        logger.warning(f"Failed to generate project name via LLM: {e}")
                
                out.projects.append(Project(name=project_name, description=ans))
            else:
                if idx < len(out.projects):
                    if field == "description":
                        out.projects[idx].description = ans
                    elif field == "tech":
                        out.projects[idx].tech_used = [canonical_skill_name(t.strip()) for t in ans.split(",") if t.strip()]
                    elif field == "outcome":
                        out.projects[idx].outcome = ans
        elif sec == "skills":
            skills = [canonical_skill_name(s.strip()) for s in ans.split(",") if s.strip()]
            out.skills = list(dict.fromkeys(out.skills + skills))
        elif sec == "certifications":
            certs = [c.strip() for c in ans.split("\n") if c.strip() and is_valid_certification(c.strip())]
            out.certifications = list(dict.fromkeys(out.certifications + certs))
        elif sec == "accomplishments":
            accs = [a.strip() for a in ans.split("\n") if a.strip() and is_valid_accomplishment(a.strip())]
            out.accomplishments = list(dict.fromkeys(out.accomplishments + accs))
            
    return out


def apply_edits_and_recalculate():
    if not st.session_state.edit_mode or not st.session_state.edit_resume:
        return
        
    er = st.session_state.edit_resume
    from core.resume_parser import WorkExperience, Project, Education as Edu
    
    # Update er from st.session_state widget keys
    if "ed_name" in st.session_state: er["name"] = st.session_state.ed_name
    if "ed_phone" in st.session_state: er["phone"] = st.session_state.ed_phone
    if "ed_li" in st.session_state: er["linkedin"] = st.session_state.ed_li
    if "ed_email" in st.session_state: er["email"] = st.session_state.ed_email
    if "ed_gh" in st.session_state: er["github"] = st.session_state.ed_gh
    if "ed_port" in st.session_state: er["portfolio"] = st.session_state.ed_port
    if "ed_summary" in st.session_state: er["summary"] = st.session_state.ed_summary
    
    for i in range(len(er["experience"])):
        if f"ed_et_{i}" in st.session_state: er["experience"][i]["title"] = st.session_state[f"ed_et_{i}"]
        if f"ed_ec_{i}" in st.session_state: er["experience"][i]["company"] = st.session_state[f"ed_ec_{i}"]
        if f"ed_ed_{i}" in st.session_state: er["experience"][i]["duration"] = st.session_state[f"ed_ed_{i}"]
        if f"ed_eb_{i}" in st.session_state:
            bullets_text = st.session_state[f"ed_eb_{i}"]
            er["experience"][i]["bullets"] = [b.strip() for b in bullets_text.split("\n") if b.strip()]
            
    for i in range(len(er["projects"])):
        if f"ed_pn_{i}" in st.session_state: er["projects"][i]["name"] = st.session_state[f"ed_pn_{i}"]
        if f"ed_pt_{i}" in st.session_state:
            tech_str = st.session_state[f"ed_pt_{i}"]
            er["projects"][i]["tech_used"] = [canonical_skill_name(t.strip()) for t in tech_str.split(",") if t.strip()]
        if f"ed_pd_{i}" in st.session_state: er["projects"][i]["description"] = st.session_state[f"ed_pd_{i}"]
        if f"ed_po_{i}" in st.session_state: er["projects"][i]["outcome"] = st.session_state[f"ed_po_{i}"]
        if f"ed_pl_{i}" in st.session_state: er["projects"][i]["link"] = st.session_state[f"ed_pl_{i}"]
        if f"ed_pdu_{i}" in st.session_state: er["projects"][i]["duration"] = st.session_state[f"ed_pdu_{i}"]
        
    if "ed_skills" in st.session_state:
        skills_str = st.session_state.ed_skills
        er["skills"] = [canonical_skill_name(s.strip()) for s in skills_str.split(",") if s.strip()]
        
    for i in range(len(er["education"])):
        if f"ed_ded_{i}" in st.session_state: er["education"][i]["degree"] = st.session_state[f"ed_ded_{i}"]
        if f"ed_dei_{i}" in st.session_state: er["education"][i]["institution"] = st.session_state[f"ed_dei_{i}"]
        if f"ed_dey_{i}" in st.session_state: er["education"][i]["year"] = st.session_state[f"ed_dey_{i}"]
        if f"ed_deg_{i}" in st.session_state: er["education"][i]["gpa"] = st.session_state[f"ed_deg_{i}"]
        
    if "ed_certs" in st.session_state:
        certs_text = st.session_state.ed_certs
        er["certifications"] = [c.strip() for c in certs_text.split("\n") if c.strip() and is_valid_certification(c.strip())]
        
    if "ed_accs" in st.session_state:
        accs_text = st.session_state.ed_accs
        er["accomplishments"] = [a.strip() for a in accs_text.split("\n") if a.strip() and is_valid_accomplishment(a.strip())]
        
    if "ed_pubs" in st.session_state:
        pubs_text = st.session_state.ed_pubs
        er["publications"] = [p.strip() for p in pubs_text.split("\n") if p.strip()]
    
    # Rebuild EnhancedResume
    enh = st.session_state.enhanced_resume
    if not enh:
        return
        
    enh.name      = er["name"]
    enh.email     = er["email"]
    enh.phone     = er["phone"]
    enh.linkedin  = er["linkedin"]
    enh.github    = er["github"]
    enh.portfolio = er.get("portfolio", "")
    enh.enhanced_summary = er["summary"]
    enh.enhanced_skills  = er["skills"]
    enh.enhanced_experience = [
        WorkExperience(
            title=e["title"], company=e["company"],
            duration=e["duration"], bullets=e["bullets"]
        ) for e in er["experience"]
    ]
    enh.enhanced_projects = [
        Project(
            name=p["name"], tech_used=p["tech_used"],
            description=p["description"], outcome=p["outcome"], link=p["link"],
            duration=p.get("duration", "")
        ) for p in er["projects"]
    ]
    enh.education = [
        Edu(degree=e["degree"], institution=e["institution"],
            year=e["year"], gpa=e["gpa"])
        for e in er["education"]
    ]
    enh.certifications  = er["certifications"]
    enh.accomplishments = er["accomplishments"]
    enh.publications    = er.get("publications", [])
    
    st.session_state.enhanced_resume = enh
    st.session_state.html_resume = generate_html_resume(enh)
    
    # Re-score
    pr_post = ParsedResume()
    pr_post.name      = enh.name
    pr_post.email     = enh.email
    pr_post.phone     = enh.phone
    pr_post.linkedin  = enh.linkedin
    pr_post.github    = enh.github
    pr_post.portfolio = getattr(enh, "portfolio", "")
    pr_post.summary   = enh.enhanced_summary
    pr_post.skills    = enh.enhanced_skills
    pr_post.experience = enh.enhanced_experience
    pr_post.projects   = enh.enhanced_projects
    pr_post.education  = enh.education
    pr_post.certifications = enh.certifications
    pr_post.accomplishments = enh.accomplishments
    pr_post.publications   = enh.publications
    
    sections = ["experience","skills","projects","education","summary"]
    if enh.certifications: sections.append("certifications")
    if enh.accomplishments: sections.append("accomplishments")
    if getattr(enh, "publications", None): sections.append("publications")
    pr_post.sections_found = sections
    pr_post.word_count = (
        len(enh.enhanced_summary.split()) +
        sum(len(b.split()) for e in enh.enhanced_experience for b in e.bullets) +
        len(enh.enhanced_skills) * 2
    )
    pr_post.raw_text = (
        " ".join(enh.enhanced_skills) + " " +
        enh.enhanced_summary + " " +
        " ".join(b for e in enh.enhanced_experience for b in e.bullets)
    )
    
    jd = st.session_state.parsed_jd
    if not jd:
        from core.jd_engine import ParsedJD
        jd = ParsedJD()
        jd.role_title = enh.target_role
        jd.domain     = enh.domain
        
    st.session_state.score_result = score_resume(pr_post, jd)


# Recalculate background classification and sufficiency report dynamically based on target domain / JD
recalculate_sufficiency_and_background()

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL
# ══════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1, 1], gap="small")

with col_left:

    # ── CONTACT ──────────────────────────────────────────────────────────────
    with st.expander("📋  CONTACT", expanded=True):
        pr = st.session_state.parsed_resume
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("NAME",     value=pr.name if pr else "", placeholder="Your Name",          key="inp_name")
            phone     = st.text_input("PHONE",    value=pr.phone if pr else "", placeholder="+91 98765 43210",    key="inp_phone")
            linkedin  = st.text_input("LINKEDIN", value=pr.linkedin if pr else "", placeholder="linkedin.com/in/you",key="inp_li")
        with c2:
            email     = st.text_input("EMAIL",    value=pr.email if pr else "", placeholder="you@example.com",    key="inp_email")
            location  = st.text_input("LOCATION", value="", placeholder="City, State",        key="inp_loc")
            github    = st.text_input("GITHUB",   value=pr.github if pr else "", placeholder="github.com/you",     key="inp_gh")
            portfolio = st.text_input("PORTFOLIO", value=getattr(pr, "portfolio", "") if pr else "", placeholder="yourportfolio.com", key="inp_port")

    # ── UPLOAD ───────────────────────────────────────────────────────────────
    with st.expander("📂  UPLOAD RESUME (PDF / DOCX)", expanded=True):
        resume_file = st.file_uploader(
            "Drag & drop or click to upload",
            type=["pdf", "docx"],
            label_visibility="collapsed",
            key="resume_upload",
        )
        if resume_file:
            try:
                parsed = parse_resume(resume_file.getvalue(), resume_file.name)
                st.session_state.parsed_resume = parsed
                st.session_state.score_result_input = None
                st.session_state.score_result = None

                # ── Dynamic Sufficiency & Background Check ──────────────────
                recalculate_sufficiency_and_background()
                bg = st.session_state.bg_classification

                # ── Non-tech detection banner ──────────────────────────────
                if bg["is_non_technical"]:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#0ea271,#0077b6);border-radius:8px;
                                padding:12px 16px;margin-top:8px;color:#fff">
                        <div style="font-weight:700;font-size:13px;margin-bottom:4px">
                            📊 ATS Architect Mode Activated
                        </div>
                        <div style="font-size:11.5px;opacity:0.92">
                            Non-technical background detected: <strong>{bg['background_domain']}</strong>
                            (confidence: {round(bg['confidence']*100)}%)<br>
                            Your CV will be transformed into a <strong>Data Analyst resume</strong>
                            using our Transferability Engine — no experience invented.
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.session_state.ats_architect_report = None  # will be built on generate
                else:
                    st.success(f"✅ Parsed: {parsed.name or 'resume'} — {len(parsed.skills)} skills, {len(parsed.experience)} roles, {len(parsed.projects)} projects")

                if parsed.sparse_sections:
                    st.warning(f"⚠️ Sparse: {', '.join(parsed.sparse_sections)} — please fill in below")
            except Exception as ex:
                st.error(f"Parse error: {ex}")

    # ── EXPERIENCE ───────────────────────────────────────────────────────────
    with st.expander("💼  EXPERIENCE", expanded=False):
        num_jobs = st.number_input("Number of roles", 0, 5, 0, key="num_jobs")
        fulltime_details = []
        for i in range(int(num_jobs)):
            st.markdown(f"**Role {i+1}**")
            r1, r2 = st.columns(2)
            with r1:
                jt = st.text_input("Job Title", key=f"jt_{i}", placeholder="ML Engineer")
                co = st.text_input("Company",   key=f"co_{i}", placeholder="Acme Corp")
            with r2:
                du = st.text_input("Duration",  key=f"du_{i}", placeholder="Jun 2022 – Present")
            bl = st.text_area("Key bullets (one per line)", key=f"bl_{i}", height=70,
                               placeholder="Built model reducing latency by 40%\nLed team of 4")
            fulltime_details.append({
                "title": jt, "company": co, "duration": du,
                "bullets": [b.strip() for b in bl.split("\n") if b.strip()],
            })
        st.session_state.onboarding.fulltime_details = fulltime_details
        st.session_state.onboarding.num_fulltime_jobs = len(fulltime_details)

    # ── PROJECTS ─────────────────────────────────────────────────────────────
    with st.expander("🚀  PROJECTS", expanded=False):
        num_proj = st.number_input("Number of projects", 0, 6, 0, key="num_proj")
        major_project_details = []
        for i in range(int(num_proj)):
            st.markdown(f"**Project {i+1}**")
            p1, p2 = st.columns(2)
            with p1:
                pn = st.text_input("Project Name", key=f"pn_{i}", placeholder="Fraud Detection System")
                pt = st.text_input("Tech Stack",   key=f"pt_{i}", placeholder="PyTorch, FastAPI, Docker")
            with p2:
                pl = st.text_input("Link (optional)", key=f"pl_{i}", placeholder="github.com/you/project")
            pd_ = st.text_area("Description", key=f"pd_{i}", height=55,
                                placeholder="Built real-time fraud detection using LSTM, 97% accuracy")
            po  = st.text_area("Outcome / Impact", key=f"po_{i}", height=45,
                                placeholder="Reduced false positives by 35%, deployed to 500K+ users")
            major_project_details.append({"name": pn, "tech": pt, "link": pl, "description": pd_, "outcome": po})
        st.session_state.onboarding.major_project_details = major_project_details
        st.session_state.onboarding.num_major_projects = len(major_project_details)

    # ── EDUCATION ────────────────────────────────────────────────────────────
    with st.expander("🎓  EDUCATION", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            ug_deg  = st.text_input("UG Degree",  placeholder="B.Tech Computer Science", key="ug_deg")
            ug_col  = st.text_input("UG College", placeholder="IIT Kharagpur",            key="ug_col")
            ug_cgpa = st.text_input("UG CGPA",    placeholder="8.7 / 10",                  key="ug_cgpa")
        with c2:
            pg_deg  = st.text_input("PG Degree",  placeholder="M.Tech AI",                key="pg_deg")
            pg_col  = st.text_input("PG College", placeholder="IISc Bangalore",            key="pg_col")
            pg_cgpa = st.text_input("PG CGPA",    placeholder="9.1 / 10",                  key="pg_cgpa")
        st.session_state.onboarding.ug_degree  = ug_deg
        st.session_state.onboarding.ug_college = ug_col
        st.session_state.onboarding.ug_cgpa    = ug_cgpa
        st.session_state.onboarding.pg_degree  = pg_deg
        st.session_state.onboarding.pg_college = pg_col
        st.session_state.onboarding.pg_cgpa    = pg_cgpa

    # ── SKILLS ───────────────────────────────────────────────────────────────
    with st.expander("🛠️  SKILLS (optional override)", expanded=False):
        skills_raw = st.text_area(
            "Comma-separated (leave blank to auto-extract from resume)",
            height=65, key="skills_raw",
            placeholder="Python, PyTorch, TensorFlow, SQL, Docker, FastAPI, AWS, LangChain",
        )
        st.caption("Agent 5 automatically injects JD-matched keywords on top.")

    # ── CERTIFICATIONS ───────────────────────────────────────────────────────
    with st.expander("🏅  CERTIFICATIONS & EXTRAS", expanded=False):
        certs_val = "\n".join(pr.certifications) if (pr and pr.certifications) else ""
        certs_raw = st.text_area("Certifications (one per line)", value=certs_val, height=65, key="certs_raw",
            placeholder="AWS Certified ML Specialty — 2024\nDeep Learning Specialization — Coursera")
        
        clubs_val = "\n".join(pr.accomplishments) if (pr and pr.accomplishments) else ""
        clubs_raw = st.text_area("Clubs / Volunteering (one per line)", value=clubs_val, height=55, key="clubs_raw",
            placeholder="ML Club Secretary — IIT KGP\nMentor — Google DSC")
        
        pubs_val = "\n".join(pr.publications) if (pr and pr.publications) else ""
        pubs_raw  = st.text_area("Publications / Research (one per line)", value=pubs_val, height=55, key="pubs_raw",
            placeholder="Title of paper — Journal/Conference, Year")
        st.session_state.onboarding.clubs_societies = [
            l.strip() for l in clubs_raw.split("\n") if l.strip() and is_valid_accomplishment(l.strip())
        ]

    # ── JD ───────────────────────────────────────────────────────────────────
    with st.expander("📝  JOB DESCRIPTION", expanded=True):
        jd_company    = st.text_input("Company Name", placeholder="Google, Flipkart, Swiggy…", key="jd_company")
        target_domain = st.selectbox("Target Domain", ROLE_DOMAINS, key="jd_domain")
        jd_text = st.text_area(
            "Paste full JD",
            height=170,
            key="jd_text",
            placeholder=(
                "Senior Machine Learning Engineer\n\n"
                "Required: Python, PyTorch, TensorFlow, MLOps, Docker, Kubernetes\n"
                "5+ years in ML model development and deployment...\n"
                "Must have: LLM fine-tuning, RAG pipelines, distributed training"
            ),
        )
        if jd_text and len(jd_text) > 50:
            try:
                jd = parse_jd(jd_text, company_name=jd_company)
                if getattr(st.session_state, "parsed_jd", None) != jd:
                    st.session_state.parsed_jd = jd
                    st.session_state.score_result_input = None
                    st.session_state.score_result = None
                st.success(f"✅ JD: **{jd.role_title}** | {jd.seniority} | {len(jd.all_keywords)} keywords | {jd.domain}")
                
                # Dynamic domain compatibility pre-check
                if st.session_state.parsed_resume:
                    from core.role_intelligence import calculate_compatibility
                    st.session_state.compatibility_report = calculate_compatibility(
                        st.session_state.parsed_resume, jd
                    )
            except Exception as ex:
                st.error(f"JD parse error: {ex}")

    # ── INTERACTIVE QUESTIONNAIRE ─────────────────────────────────────────────
    if st.session_state.parsed_resume and st.session_state.sufficiency_report and not st.session_state.sufficiency_report.get("is_sufficient", True):
        is_nt = st.session_state.sufficiency_report.get("is_non_tech_mode", False)
        banner_label = (
            "📊  DA TRANSITION QUESTIONNAIRE (REQUIRED FOR ATS 85+)"
            if is_nt else
            "❓  INTERACTIVE QUESTIONNAIRE (REQUIRED FOR ATS 90+)"
        )
        with st.expander(banner_label, expanded=True):
            if is_nt:
                st.markdown(
                    "<p style='font-size:12px;color:#0ea271;font-weight:600'>"
                    "🔄 Data Analyst Transition Mode — answer these questions to help reframe "
                    "your experience in analytics language. No experience will be invented."
                    "</p>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<p style='font-size:12px;color:#666'>Your resume is missing details that are critical for ranking well. Please answer the following questions to help our agents optimize your resume:</p>", unsafe_allow_html=True)
            for q in st.session_state.sufficiency_report["questions"]:
                st.text_area(
                    q["label"],
                    placeholder=q["placeholder"],
                    key=f"q_resp_{q['id']}",
                    height=80
                )

    # ── TARGET ROLE selector (non-tech only) ──────────────────────────────────
    bg = st.session_state.get("bg_classification")
    if bg and bg.get("is_non_technical"):
        from config import DA_TARGET_ROLES
        with st.expander("🎯  TARGET ROLE (Role Transition)", expanded=True):
            st.session_state.da_target_role = st.selectbox(
                "Select your target role:",
                DA_TARGET_ROLES,
                index=0,
                key="da_target_selector",
            )
            st.caption("The resume will be optimized specifically for this role.")

    # ── LAYOUT & PAGE SETTINGS ───────────────────────────────────────────────
    with st.expander("📐  LAYOUT & PAGE SETTINGS", expanded=True):
        page_target = st.radio(
            "Page Target:",
            ["Strict 1-Page (Recommended)", "Allow 2-Pages (For extensive experience)"],
            index=0,
            key="page_target_selector"
        )
        st.caption("Strict 1-Page will compress and trim content to fit on a single A4 page.")

    # ── LLM SETTINGS ─────────────────────────────────────────────────────────
    with st.expander("⚙️  LLM BACKEND SETTINGS", expanded=False):
        st.markdown(f"**Active backend:** `{LLM_BACKEND.upper()}`")

        if LLM_BACKEND.lower() == "ollama":
            st.markdown(f"**Model:** `{OLLAMA_MODEL}`")
            st.caption(
                "To change: edit `config.py` → `OLLAMA_MODEL`\n\n"
                "Available models: mistral:7b · llama3.1:8b · phi3:mini · gemma2:9b · codellama:7b"
            )
            if st.button("🔗 Check Ollama Connection", key="check_ollama"):
                status = check_ollama()
                if status["ok"] and status.get("model_ready"):
                    st.markdown(f'<div class="ollama-ok">✅ Ollama running — model <strong>{OLLAMA_MODEL}</strong> ready<br>Available: {", ".join(status["models"][:5])}</div>', unsafe_allow_html=True)
                elif status["ok"]:
                    st.markdown(f'<div class="ollama-err">⚠️ Ollama running but <strong>{OLLAMA_MODEL}</strong> not pulled.<br>Run: <code>ollama pull {OLLAMA_MODEL}</code></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ollama-err">❌ Ollama not reachable.<br>1. <code>ollama serve</code><br>2. <code>ollama pull {OLLAMA_MODEL}</code><br><small>{status["error"]}</small></div>', unsafe_allow_html=True)

            with st.expander("📖 Ollama quick-start"):
                st.code(
                    "# Install Ollama: https://ollama.com\n"
                    "ollama serve                  # start server\n"
                    f"ollama pull {OLLAMA_MODEL}          # pull model (~4GB)\n"
                    "# Edit config.py to switch model:\n"
                    "# OLLAMA_MODEL = 'llama3.1:8b'   # best quality\n"
                    "# OLLAMA_MODEL = 'phi3:mini'      # fastest (~2GB)\n"
                    "# OLLAMA_MODEL = 'gemma2:9b'      # Google, very capable",
                    language="bash"
                )

        elif LLM_BACKEND.lower() == "huggingface":
            from config import HF_MODEL_PATH, HF_LOAD_IN_4BIT, HF_LOAD_IN_8BIT, HF_DEVICE
            st.markdown(f"**Model:** `{HF_MODEL_PATH}`")
            st.markdown(f"**Device:** `{HF_DEVICE}` | **4-bit:** `{HF_LOAD_IN_4BIT}` | **8-bit:** `{HF_LOAD_IN_8BIT}`")
            st.caption(
                "Model loads from HuggingFace Hub (or local path) on first run.\n"
                "Edit `config.py` → `HF_MODEL_PATH` to switch models."
            )
            with st.expander("📖 HuggingFace setup"):
                st.code(
                    "# Install dependencies:\n"
                    "pip install transformers torch accelerate bitsandbytes\n\n"
                    "# config.py settings:\n"
                    "LLM_BACKEND = 'huggingface'\n"
                    "HF_MODEL_PATH = 'microsoft/phi-3-mini-4k-instruct'   # ~2GB\n"
                    "# or: 'mistralai/Mistral-7B-Instruct-v0.2'           # ~4GB\n"
                    "# or: 'meta-llama/Llama-3.1-8B-Instruct'             # needs HF_TOKEN\n"
                    "HF_LOAD_IN_4BIT = True    # saves GPU memory\n"
                    "HF_DEVICE = 'auto'        # picks GPU automatically",
                    language="python"
                )

        elif LLM_BACKEND.lower() == "finetuned":
            from config import FINETUNED_BASE_MODEL, FINETUNED_ADAPTER_PATH, FINETUNED_MERGED_PATH
            if FINETUNED_MERGED_PATH:
                st.markdown(f"**Merged model:** `{FINETUNED_MERGED_PATH}`")
            elif FINETUNED_ADAPTER_PATH:
                st.markdown(f"**Base:** `{FINETUNED_BASE_MODEL}`")
                st.markdown(f"**LoRA adapter:** `{FINETUNED_ADAPTER_PATH}`")
            else:
                st.warning("⚠️ Set `FINETUNED_ADAPTER_PATH` or `FINETUNED_MERGED_PATH` in config.py")
            with st.expander("📖 Fine-tuned model setup"):
                st.code(
                    "# config.py — LoRA adapter:\n"
                    "LLM_BACKEND = 'finetuned'\n"
                    "FINETUNED_BASE_MODEL   = 'mistralai/Mistral-7B-Instruct-v0.2'\n"
                    "FINETUNED_ADAPTER_PATH = '/path/to/your/lora_adapter'\n"
                    "FINETUNED_MERGED_PATH  = ''   # leave empty when using adapter\n\n"
                    "# config.py — fully merged model:\n"
                    "LLM_BACKEND = 'finetuned'\n"
                    "FINETUNED_BASE_MODEL   = ''\n"
                    "FINETUNED_ADAPTER_PATH = ''\n"
                    "FINETUNED_MERGED_PATH  = '/path/to/merged_model'\n\n"
                    "# Install:\n"
                    "pip install transformers torch accelerate bitsandbytes peft",
                    language="python"
                )

    st.markdown("---")

    # ── GENERATE BUTTON ───────────────────────────────────────────────────────
    generate_clicked = st.button("✨  Generate & Enhance Resume", key="generate_btn", type="primary")

    # ── EDIT MODE TOGGLE ──────────────────────────────────────────────────────
    if st.session_state.enhanced_resume is not None:
        if st.button("✏️  Edit Resume Section-by-Section", key="edit_btn"):
            enh = st.session_state.enhanced_resume
            st.session_state.edit_mode = not st.session_state.edit_mode
            if st.session_state.edit_mode:
                # Initialise editable dict from enhanced resume
                st.session_state.edit_resume = {
                    "name":      enh.name,
                    "email":     enh.email,
                    "phone":     enh.phone,
                    "linkedin":  enh.linkedin,
                    "github":    enh.github,
                    "portfolio": getattr(enh, "portfolio", ""),
                    "summary":   enh.enhanced_summary,
                    "skills":   enh.enhanced_skills,
                    "experience": [
                        {"title": e.title, "company": e.company,
                         "duration": e.duration, "bullets": list(e.bullets)}
                        for e in enh.enhanced_experience
                    ],
                    "projects": [
                        {"name": p.name, "tech_used": list(p.tech_used),
                         "description": p.description, "outcome": p.outcome, "link": p.link,
                         "duration": getattr(p, "duration", "")}
                        for p in enh.enhanced_projects
                    ],
                    "education": [
                        {"degree": ed.degree, "institution": ed.institution,
                         "year": ed.year, "gpa": ed.gpa}
                        for ed in enh.education
                    ],
                    "certifications": list(enh.certifications),
                    "accomplishments": list(enh.accomplishments),
                    "publications": list(enh.publications) if getattr(enh, "publications", None) else [],
                }

    # ── DOWNLOAD BUTTONS ──────────────────────────────────────────────────────
    if st.session_state.enhanced_resume is not None:
        if st.session_state.pdf_bytes:
            st.download_button(
                "⬇️ Download PDF",
                data=st.session_state.pdf_bytes,
                file_name=f"{(st.session_state.enhanced_resume.name or 'resume').replace(' ','_')}_ats.pdf",
                mime="application/pdf",
                key="dl_pdf",
                use_container_width=True,
            )
        else:
            st.error("⚠️ PDF generation failed. Ensure Microsoft Edge or Google Chrome is installed on the system.")





# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL
# ══════════════════════════════════════════════════════════════════════════════

with col_right:

    # ── Real-time Recalculation Call ──────────────────────────────────────────
    if st.session_state.edit_mode and st.session_state.edit_resume:
        apply_edits_and_recalculate()

    # ── Domain Compatibility Analysis ─────────────────────────────────────────
    if st.session_state.parsed_resume and st.session_state.parsed_jd:
        from core.role_intelligence import calculate_compatibility
        comp = calculate_compatibility(st.session_state.parsed_resume, st.session_state.parsed_jd)
        st.session_state.compatibility_report = comp
        
        score = comp["compatibility_score"]
        is_mismatch = comp["is_mismatch"]
        color = "#ef4444" if is_mismatch else ("#f59e0b" if score < 60 else "#0ea271")
        bg_color = "#fef2f2" if is_mismatch else ("#fff8e1" if score < 60 else "#e6f9f2")
        
        st.markdown(f"""
        <div style="background:{bg_color};border:1px solid {color};border-radius:8px;padding:12px;margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="font-weight:700;font-size:13.5px;color:#1a1a1a">Domain Compatibility Analysis</span>
                <span style="background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700">{score}% Match</span>
            </div>
            <p style="font-size:11px;color:#444;margin:2px 0 6px 0">Candidate primary domain: <strong>{comp['candidate_primary_domain']}</strong> &rarr; Target JD domain: <strong>{st.session_state.parsed_jd.domain}</strong></p>
        """, unsafe_allow_html=True)
        
        if is_mismatch:
            st.markdown("""
            <div style="font-size:11px;color:#ef4444;font-weight:600;margin-bottom:6px">
                ⚠️ Domain Mismatch Warning: Blind transition may yield high hallucination risk and low recruiter trust.
            </div>
            """, unsafe_allow_html=True)
            
        for reason in comp["reasons"]:
            st.markdown(f"<div style='font-size:11px;color:#555'>• {reason}</div>", unsafe_allow_html=True)
            
        if comp["missing_foundations"]:
            st.markdown(f"<div style='font-size:11px;color:#777;margin-top:4px'><strong>Missing foundations:</strong> {', '.join(comp['missing_foundations'])}</div>", unsafe_allow_html=True)
        if comp["crossover_skills"]:
            st.markdown(f"<div style='font-size:11px;color:#777'><strong>Crossover skills:</strong> {', '.join(comp['crossover_skills'])}</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ATS ARCHITECT REPORT (non-technical CVs) ──────────────────────────────
    arch: Optional[ATSArchitectReport] = st.session_state.get("ats_architect_report")
    if arch:
        cur = arch.current_ats_score
        pred = arch.predicted_ats_score

        st.markdown("""
        <div style="background:linear-gradient(135deg,#0ea271 0%,#0077b6 100%);
                    border-radius:10px;padding:14px 18px;margin-bottom:12px;color:#fff">
            <div style="font-size:14px;font-weight:800;letter-spacing:.5px;margin-bottom:2px">
                📊 ATS ARCHITECT REPORT
            </div>
            <div style="font-size:11px;opacity:0.88">Non-Technical → Data Analyst Transition</div>
        </div>""", unsafe_allow_html=True)

        # ── Before / After score cards ────────────────────────────────────────
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown(f"""
            <div style="background:#fff;border-radius:8px;border:2px solid #ef4444;
                        padding:12px;text-align:center;margin-bottom:10px">
                <div style="font-size:32px;font-weight:800;color:#ef4444">{cur.overall:.0f}</div>
                <div style="font-size:10px;color:#888">Current ATS Score</div>
                <div style="font-size:9px;color:#aaa;margin-top:3px">Before Enhancement</div>
            </div>""", unsafe_allow_html=True)
        with ac2:
            st.markdown(f"""
            <div style="background:#fff;border-radius:8px;border:2px solid #0ea271;
                        padding:12px;text-align:center;margin-bottom:10px">
                <div style="font-size:32px;font-weight:800;color:#0ea271">{pred.overall:.0f}</div>
                <div style="font-size:10px;color:#888">Predicted ATS Score</div>
                <div style="font-size:9px;color:#aaa;margin-top:3px">After Enhancement</div>
            </div>""", unsafe_allow_html=True)

        # ── 7-dimension breakdown table ───────────────────────────────────────
        dimensions = [
            ("Keyword Match",        cur.keyword_match,        pred.keyword_match,        "25%"),
            ("Skill Match",          cur.skill_match,          pred.skill_match,          "20%"),
            ("Semantic Similarity",  cur.semantic_similarity,  pred.semantic_similarity,  "20%"),
            ("Project Relevance",    cur.project_relevance,    pred.project_relevance,    "15%"),
            ("Exp. Alignment",       cur.experience_alignment, pred.experience_alignment, "10%"),
            ("Education",            cur.education_alignment,  pred.education_alignment,  "5%"),
            ("Formatting",           cur.formatting_quality,   pred.formatting_quality,   "5%"),
        ]
        for label, before, after, weight in dimensions:
            delta = after - before
            delta_color = "#0ea271" if delta >= 0 else "#ef4444"
            bar_before = int(before * 0.8)
            bar_after  = int(after * 0.8)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;font-size:11px">
                <span style="min-width:120px;color:#444">{label} <span style="color:#aaa">({weight})</span></span>
                <span style="color:#ef4444;min-width:32px;text-align:right">{before:.0f}</span>
                <div style="flex:1;height:8px;background:#f0f0f0;border-radius:4px;position:relative">
                    <div style="position:absolute;left:0;top:0;height:8px;width:{bar_before}%;
                                background:#ef4444;border-radius:4px;opacity:.6"></div>
                    <div style="position:absolute;left:0;top:0;height:8px;width:{bar_after}%;
                                background:#0ea271;border-radius:4px;opacity:.5"></div>
                </div>
                <span style="color:{delta_color};min-width:32px;text-align:right">{after:.0f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Section 3: Transferable Skills ───────────────────────────────────
        with st.expander("✅  Transferable Skills Identified", expanded=False):
            if arch.transferable_skills:
                cols = st.columns(3)
                for i, skill in enumerate(arch.transferable_skills):
                    cols[i % 3].markdown(
                        f"<span style='background:#e6f9f2;color:#085041;padding:2px 8px;"
                        f"border-radius:12px;font-size:11px;display:inline-block;margin:2px'>"
                        f"✓ {skill}</span>",
                        unsafe_allow_html=True
                    )

        # ── Knowledge Graph Path ──────────────────────────────────────────────
        with st.expander("🔗  Knowledge Graph — Your Transition Path", expanded=False):
            for i, step in enumerate(arch.knowledge_graph_path):
                color = "#0ea271" if i == len(arch.knowledge_graph_path) - 1 else "#1a1a1a"
                weight = "800" if i == 0 or i == len(arch.knowledge_graph_path) - 1 else "400"
                st.markdown(
                    f"<div style='font-size:12px;color:{color};font-weight:{weight};"
                    f"padding:2px 0'>{step}</div>",
                    unsafe_allow_html=True
                )

        # ── Section 2: Skill Gap ──────────────────────────────────────────────
        with st.expander("❌  Skill Gap Analysis", expanded=False):
            gap = arch.skill_gap
            tier_colors = {"core": "#ef4444", "intermediate": "#f59e0b", "advanced": "#6366f1"}
            for tier, color in tier_colors.items():
                missing = gap.get(tier, [])
                if missing:
                    st.markdown(
                        f"<div style='font-size:11px;font-weight:700;color:{color};"
                        f"margin:4px 0 2px'>{tier.capitalize()} Skills Missing:</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<span style='font-size:11px;color:#555'>" +
                        " • ".join(missing[:8]) + "</span>",
                        unsafe_allow_html=True
                    )

        # ── Section 4: Recommended DA Skills ─────────────────────────────────
        with st.expander("🎯  Recommended Data Analyst Skills to Learn", expanded=False):
            for i, skill in enumerate(arch.recommended_da_skills[:12]):
                priority = "🔴 Must Have" if i < 4 else ("🟡 Should Have" if i < 8 else "🟢 Nice to Have")
                st.markdown(f"<div style='font-size:11px;padding:2px 0'>**{i+1}.** {skill} — <span style='color:#888'>{priority}</span></div>", unsafe_allow_html=True)

        # ── Section 5: Certifications ─────────────────────────────────────────
        with st.expander("📜  Recommended Certifications", expanded=False):
            certs = arch.recommended_certs
            p_labels = {"priority_1": "🥇 Start Here", "priority_2": "🥈 Next Step", "priority_3": "🥉 Advanced"}
            for key, label in p_labels.items():
                items = certs.get(key, [])
                if items:
                    st.markdown(f"**{label}**")
                    for cert in items:
                        st.markdown(f"<div style='font-size:11px;color:#555;padding:1px 0'>• {cert}</div>", unsafe_allow_html=True)

        # ── Section 6: Recommended Projects ──────────────────────────────────
        with st.expander("🚀  Recommended Projects (Build These First)", expanded=False):
            for proj in arch.recommended_projects:
                st.markdown(f"""
                <div style="background:#f9f9f9;border:1px solid #e8e6e0;border-radius:8px;
                            padding:10px 14px;margin-bottom:8px">
                    <div style="font-weight:700;font-size:12px;color:#1a1a1a">{proj['name']}</div>
                    <div style="font-size:10px;color:#0ea271;margin:2px 0">
                        Targets: <strong>{proj.get('skill_targeted','').title()}</strong> •
                        Difficulty: {proj.get('difficulty','')}
                    </div>
                    <div style="font-size:11px;color:#555;margin-top:4px">{proj['description']}</div>
                    <div style="font-size:10px;color:#888;margin-top:3px">
                        🛠 {', '.join(proj.get('tools', [])[:3])} •
                        📊 Dataset: {proj.get('dataset','')}
                    </div>
                    <div style="font-size:10px;color:#0ea271;margin-top:3px">
                        ✅ Outcome: {proj.get('outcome','')}
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Section 7: Reframed Resume Bullets ───────────────────────────────
        with st.expander("✍️  Reframed Resume Bullets (Analytics Language)", expanded=False):
            st.caption("These rewrites maintain factual accuracy while repositioning your experience in data analytics language.")
            for role, bullets in arch.reframed_bullets.items():
                if bullets:
                    st.markdown(f"**{role}**")
                    for b in bullets:
                        st.markdown(
                            f"<div style='font-size:11px;color:#333;padding:2px 0 2px 12px;"
                            f"border-left:2px solid #0ea271;margin-bottom:4px'>{b}</div>",
                            unsafe_allow_html=True
                        )

        # ── Section 9: Recruiter Feedback ────────────────────────────────────
        with st.expander("💼  Recruiter Feedback Report", expanded=True):
            for fb in arch.recruiter_feedback:
                st.markdown(
                    f"<div style='font-size:11.5px;color:#333;padding:6px 0;"
                    f"border-bottom:1px solid #f0f0f0'>{fb}</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")

    # ── Score dashboard ───────────────────────────────────────────────────────
    ensure_input_score()
    sr_opt = st.session_state.score_result
    sr_in = st.session_state.score_result_input
    sr = sr_opt if sr_opt else sr_in

    if sr_opt and sr_in:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0ea271 0%,#0077b6 100%);
                    border-radius:10px;padding:8px 12px;margin-bottom:12px;color:#fff">
            <div style="font-size:13px;font-weight:800;letter-spacing:.5px;text-align:center">
                📊 ATS SCORE COMPARISON
            </div>
        </div>""", unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            inp_pct = round(sr_in.overall_score * 100)
            inp_color = "#ef4444" if inp_pct < 60 else ("#f59e0b" if inp_pct < 80 else "#0ea271")
            st.markdown(f"""
            <div style="background:#fff;border-radius:8px;border:2px solid {inp_color};
                        padding:10px;text-align:center;margin-bottom:10px">
                <div style="font-size:30px;font-weight:800;color:{inp_color}">{inp_pct}</div>
                <div style="font-size:10px;font-weight:700;color:#555">Input CV ATS Score</div>
                <div style="font-size:8px;color:#aaa;margin-top:2px">Original Uploaded File</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            opt_pct = round(sr_opt.overall_score * 100)
            opt_color = "#0ea271"
            st.markdown(f"""
            <div style="background:#fff;border-radius:8px;border:2px solid {opt_color};
                        padding:10px;text-align:center;margin-bottom:10px">
                <div style="font-size:30px;font-weight:800;color:{opt_color}">{opt_pct}</div>
                <div style="font-size:10px;font-weight:700;color:#555">Generated CV ATS Score</div>
                <div style="font-size:8px;color:#aaa;margin-top:2px">Optimized by Work2Hire</div>
            </div>""", unsafe_allow_html=True)

        metrics_comparison = [
            ("ATS Keywords",    sr_in.ats_keyword_score,   sr_opt.ats_keyword_score),
            ("Semantic Match",  sr_in.semantic_score,      sr_opt.semantic_score),
            ("Tech Depth",      sr_in.technical_depth_score, sr_opt.technical_depth_score),
            ("Readability",     sr_in.readability_score,    sr_opt.readability_score),
            ("Impact",          sr_in.achievement_score,    sr_opt.achievement_score),
        ]
        
        comparison_html = """
        <table style="width:100%;font-size:11px;border-collapse:collapse;margin-bottom:15px">
          <thead>
            <tr style="border-bottom:1px solid #ddd;color:#666;text-align:left">
              <th style="padding:4px">Dimension</th>
              <th style="padding:4px;text-align:center">Input CV</th>
              <th style="padding:4px;text-align:center">Generated CV</th>
            </tr>
          </thead>
          <tbody>
        """
        for label, val_in, val_out in metrics_comparison:
            pct_in = round(val_in * 100)
            pct_out = round(val_out * 100)
            color_in = "#ef4444" if pct_in < 60 else ("#f59e0b" if pct_in < 80 else "#0ea271")
            color_out = "#ef4444" if pct_out < 60 else ("#f59e0b" if pct_out < 80 else "#0ea271")
            diff = pct_out - pct_in
            diff_badge = f'<span style="color:#0ea271;font-weight:700;font-size:9px;margin-left:3px">+{diff}%</span>' if diff > 0 else ""
            
            comparison_html += f"""
            <tr style="border-bottom:1px solid #f9f9f9">
              <td style="padding:5px;font-weight:500;color:#333">{label}</td>
              <td style="padding:5px;text-align:center;color:{color_in};font-weight:700">{pct_in}%</td>
              <td style="padding:5px;text-align:center;color:{color_out};font-weight:700">{pct_out}% {diff_badge}</td>
            </tr>
            """
        comparison_html += "</tbody></table>"
        st.markdown(comparison_html, unsafe_allow_html=True)

        metrics_radar = [
            ("ATS Keywords",   sr_opt.ats_keyword_score),
            ("Semantic Match", sr_opt.semantic_score),
            ("Tech Depth",     sr_opt.technical_depth_score),
            ("Readability",    sr_opt.readability_score),
            ("Impact",         sr_opt.achievement_score),
        ]
        fig = go.Figure(go.Scatterpolar(
            r=[round(v*100) for _, v in metrics_radar],
            theta=[l for l, _ in metrics_radar],
            fill="toself",
            fillcolor="rgba(14,162,113,0.12)",
            line=dict(color="#0ea271", width=2),
            marker=dict(color="#0ea271", size=5),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=9))),
            showlegend=False,
            margin=dict(l=30, r=30, t=20, b=20),
            height=180,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    elif sr:
        pct = round(sr.overall_score * 100)
        badge = _score_badge(sr.overall_score)
        label_text = "Overall ATS Score" if sr == sr_opt else "Input CV ATS Score"

        dash_cols = st.columns(5)
        metrics = [
            ("ATS Keywords",   sr.ats_keyword_score),
            ("Semantic Match", sr.semantic_score),
            ("Tech Depth",     sr.technical_depth_score),
            ("Readability",    sr.readability_score),
            ("Impact",         sr.achievement_score),
        ]
        for col, (label, val) in zip(dash_cols, metrics):
            with col:
                color = "#0ea271" if val >= 0.7 else ("#f59e0b" if val >= 0.5 else "#ef4444")
                st.markdown(f"""
                <div style="background:#fff;border-radius:8px;border:1px solid #e8e6e0;
                            padding:10px 6px;text-align:center;margin-bottom:8px">
                  <div style="font-size:20px;font-weight:800;color:{color}">{round(val*100)}</div>
                  <div style="font-size:9px;color:#888;margin-top:1px">{label}</div>
                </div>""", unsafe_allow_html=True)

        fig = go.Figure(go.Scatterpolar(
            r=[round(v*100) for _, v in metrics],
            theta=[l for l, _ in metrics],
            fill="toself",
            fillcolor="rgba(14,162,113,0.12)",
            line=dict(color="#0ea271", width=2),
            marker=dict(color="#0ea271", size=5),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=9))),
            showlegend=False,
            margin=dict(l=30, r=30, t=20, b=20),
            height=180,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div class="score-card" style="text-align:center;margin-bottom:8px">
          <div class="score-number">{pct}</div>
          <div class="score-label">{label_text}</div>
          {badge}
        </div>""", unsafe_allow_html=True)

        if getattr(st.session_state, "optimization_log", None):
            with st.expander("🔄 Optimization Loop History", expanded=False):
                for log in st.session_state.optimization_log:
                    st.markdown(f"""
                    **Iteration {log['iteration']}**
                    - Score: `{log['ats_score']:.1f}%` | Authenticity: `{log['authenticity_score']:.1f}%` (`{log['trust_level']}`)
                    - Details: {log['msg']}
                    ---
                    """)

        if sr.recommendations:
            with st.expander("💡 Recommendations", expanded=False):
                for rec in sr.recommendations:
                    st.markdown(f"• {rec}")

        if sr.missing_skills:
            with st.expander(f"❌ {len(sr.missing_skills)} missing JD keywords", expanded=False):
                st.write(", ".join(sr.missing_skills[:20]))

    # ── Agent pipeline log ────────────────────────────────────────────────────
    if st.session_state.agent_log:
        with st.expander("🤖 Agent Pipeline Log", expanded=False):
            dot_map = {"wait": "dot-wait", "run": "dot-run", "done": "dot-done", "err": "dot-err"}
            for name, status, msg in st.session_state.agent_log:
                dot_cls = dot_map.get(status, "dot-wait")
                icon = {"wait":"⏳","run":"🔄","done":"✅","err":"❌"}.get(status,"⏳")
                st.markdown(f"""
                <div class="agent-row">
                  <div class="agent-dot {dot_cls}"></div>
                  <span style="font-weight:600;min-width:175px">{icon} {name}</span>
                  <span style="color:#666">{msg}</span>
                </div>""", unsafe_allow_html=True)

    # ── Section-by-section editor ─────────────────────────────────────────────
    if st.session_state.edit_mode and st.session_state.edit_resume:
        er = st.session_state.edit_resume
        st.markdown("### ✏️ Edit Resume Sections")

        with st.expander("👤 Contact Info", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                er["name"]      = st.text_input("Full Name",   value=er["name"],     key="ed_name")
                er["phone"]     = st.text_input("Phone",       value=er["phone"],    key="ed_phone")
                er["linkedin"]  = st.text_input("LinkedIn URL",value=er["linkedin"], key="ed_li")
            with c2:
                er["email"]     = st.text_input("Email",      value=er["email"],  key="ed_email")
                er["github"]    = st.text_input("GitHub URL", value=er["github"], key="ed_gh")
                er["portfolio"] = st.text_input("Portfolio URL", value=er.get("portfolio", ""), key="ed_port")

        with st.expander("📋 Professional Summary", expanded=False):
            er["summary"] = st.text_area("Summary", value=er["summary"], height=100, key="ed_summary")
            st.caption("3-4 sentences. Include JD-matched keywords naturally.")

        with st.expander("💼 Experience", expanded=False):
            for i, exp in enumerate(er["experience"]):
                st.markdown(f"**Role {i+1}**")
                c1, c2, c3 = st.columns(3)
                with c1: exp["title"]    = st.text_input("Title",    value=exp["title"],    key=f"ed_et_{i}")
                with c2: exp["company"]  = st.text_input("Company",  value=exp["company"],  key=f"ed_ec_{i}")
                with c3: exp["duration"] = st.text_input("Duration", value=exp["duration"], key=f"ed_ed_{i}")
                bullets_text = "\n".join(exp["bullets"])
                new_bullets = st.text_area(
                    "Bullets (one per line — start each with a strong action verb)",
                    value=bullets_text, height=110, key=f"ed_eb_{i}"
                )
                exp["bullets"] = [b.strip() for b in new_bullets.split("\n") if b.strip()]
                st.markdown("---")

        with st.expander("🚀 Projects", expanded=False):
            for i, proj in enumerate(er["projects"]):
                st.markdown(f"**Project {i+1}: {proj['name']}**")
                c1_p, c2_p = st.columns([2, 1])
                with c1_p:
                    proj["name"] = st.text_input("Project Name", value=proj["name"], key=f"ed_pn_{i}")
                with c2_p:
                    proj["duration"] = st.text_input("Duration", value=proj.get("duration", ""), key=f"ed_pdu_{i}")
                tech_str = ", ".join(proj["tech_used"])
                new_tech = st.text_input("Tech Stack (comma-separated)", value=tech_str, key=f"ed_pt_{i}")
                proj["tech_used"] = [t.strip() for t in new_tech.split(",") if t.strip()]
                proj["description"] = st.text_area("Description", value=proj["description"] or "", height=70, key=f"ed_pd_{i}")
                proj["outcome"]     = st.text_area("Outcome / Metric", value=proj["outcome"] or "", height=55, key=f"ed_po_{i}")
                proj["link"]        = st.text_input("Link (optional)", value=proj["link"] or "", key=f"ed_pl_{i}")
                st.markdown("---")

        with st.expander("🛠️ Skills", expanded=False):
            skills_str = ", ".join(er["skills"])
            new_skills = st.text_area(
                "Comma-separated skills (Agent 5 has already injected JD keywords)",
                value=skills_str, height=80, key="ed_skills"
            )
            er["skills"] = [s.strip() for s in new_skills.split(",") if s.strip()]

        with st.expander("🎓 Education", expanded=False):
            for i, edu in enumerate(er["education"]):
                c1, c2 = st.columns(2)
                with c1:
                    edu["degree"]      = st.text_input("Degree",      value=edu["degree"] or "",      key=f"ed_ded_{i}")
                    edu["institution"] = st.text_input("Institution",  value=edu["institution"] or "", key=f"ed_dei_{i}")
                with c2:
                    edu["year"] = st.text_input("Year",      value=edu["year"] or "", key=f"ed_dey_{i}")
                    edu["gpa"]  = st.text_input("CGPA / GPA", value=edu["gpa"] or "", key=f"ed_deg_{i}")
                st.markdown("---")

        with st.expander("🏅 Certifications & Activities", expanded=False):
            certs_text = "\n".join(er["certifications"])
            new_certs = st.text_area("Certifications (one per line)", value=certs_text, height=80, key="ed_certs")
            er["certifications"] = [c.strip() for c in new_certs.split("\n") if c.strip()]
            accs_text = "\n".join(er["accomplishments"])
            new_accs = st.text_area("Accomplishments / Activities (one per line)", value=accs_text, height=70, key="ed_accs")
            er["accomplishments"] = [a.strip() for a in new_accs.split("\n") if a.strip()]
            pubs_text = "\n".join(er.get("publications", []))
            new_pubs = st.text_area("Publications / Research (one per line)", value=pubs_text, height=70, key="ed_pubs")
            er["publications"] = [p.strip() for p in new_pubs.split("\n") if p.strip()]

        if st.button("💾 Apply Edits & Regenerate Preview", key="apply_edits"):
            # Rebuild EnhancedResume from edit dict
            from core.resume_parser import WorkExperience, Project, Education as Edu
            if "ed_port" in st.session_state: er["portfolio"] = st.session_state.ed_port
            if "ed_pubs" in st.session_state:
                er["publications"] = [p.strip() for p in st.session_state.ed_pubs.split("\n") if p.strip()]
            enh = st.session_state.enhanced_resume
            enh.name      = er["name"]
            enh.email     = er["email"]
            enh.phone     = er["phone"]
            enh.linkedin  = er["linkedin"]
            enh.github    = er["github"]
            enh.portfolio = er.get("portfolio", "")
            enh.enhanced_summary = er["summary"]
            enh.enhanced_skills  = er["skills"]
            enh.enhanced_experience = [
                WorkExperience(
                    title=e["title"], company=e["company"],
                    duration=e["duration"], bullets=e["bullets"]
                ) for e in er["experience"]
            ]
            enh.enhanced_projects = [
                Project(
                    name=p["name"], tech_used=p["tech_used"],
                    description=p["description"], outcome=p["outcome"], link=p["link"],
                    duration=p.get("duration", "")
                ) for p in er["projects"]
            ]
            enh.education = [
                Edu(degree=e["degree"], institution=e["institution"],
                    year=e["year"], gpa=e["gpa"])
                for e in er["education"]
            ]
            enh.certifications  = er["certifications"]
            enh.accomplishments = er["accomplishments"]
            enh.publications    = er.get("publications", [])
            st.session_state.enhanced_resume = enh
            st.session_state.html_resume = generate_html_resume(enh)
            st.success("✅ Edits applied! Preview updated.")
            st.rerun()

    # ── Live resume preview ───────────────────────────────────────────────────
    if not st.session_state.edit_mode:
        st.markdown(
            '<div style="color:#0ea271;font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;margin-bottom:3px">LIVE RESUME</div>',
            unsafe_allow_html=True
        )
        st.components.v1.html(st.session_state.html_resume, height=900, scrolling=True)




# ══════════════════════════════════════════════════════════════════════════════
# GENERATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if generate_clicked:
    pr: ParsedResume = st.session_state.parsed_resume or ParsedResume()

    if full_name.strip():  pr.name      = full_name.strip()
    if email.strip():      pr.email     = email.strip()
    if phone.strip():      pr.phone     = phone.strip()
    if linkedin.strip():   pr.linkedin  = linkedin.strip()
    if github.strip():     pr.github    = github.strip()
    if portfolio.strip():  pr.portfolio = portfolio.strip()

    if skills_raw.strip():
        manual_skills = [canonical_skill_name(s.strip()) for s in skills_raw.split(",") if s.strip()]
        pr.skills = list(dict.fromkeys(manual_skills + pr.skills))

    if certs_raw.strip():
        pr.certifications = [l.strip() for l in certs_raw.split("\n") if l.strip() and is_valid_certification(l.strip())]

    if clubs_raw.strip():
        pr.accomplishments = [l.strip() for l in clubs_raw.split("\n") if l.strip() and is_valid_accomplishment(l.strip())]

    if pubs_raw.strip():
        pr.publications = [l.strip() for l in pubs_raw.split("\n") if l.strip()]

    if not pr.name:
        st.error("Please enter your name in the Contact section.")
        st.stop()

    jd_parsed: Optional[ParsedJD] = st.session_state.parsed_jd
    if jd_parsed is None:
        from core.jd_engine import ParsedJD
        jd_parsed = ParsedJD()
        jd_parsed.role_title = target_domain
        jd_parsed.domain     = target_domain
        jd_parsed.seniority  = "mid"

    agent_names = [
        "Agent 1 — Intake",
        "Agent 2 — Validator",
        "Agent 3 — Gap Analyst",
        "Agent 4a — Summary Rewriter",
        "Agent 4b — Bullet Rewriter",
        "Agent 4c — Project Enhancer",
        "Agent 5 — Skill Booster",
        "Agent 6 — QA Guard",
    ]
    st.session_state.agent_log = [(n, "wait", "") for n in agent_names]
    _log_agent("Agent 1 — Intake", "done", f"Name={pr.name}, roles={len(st.session_state.onboarding.fulltime_details)}")

    backend_label = {
        "ollama":      f"Ollama / {OLLAMA_MODEL}",
        "huggingface": f"HuggingFace / {HF_MODEL_PATH.split('/')[-1]}",
        "finetuned":   "Fine-tuned model",
    }.get(LLM_BACKEND.lower(), LLM_BACKEND)

    progress_bar = st.progress(0, text=f"Running multi-agent pipeline ({backend_label}, zero API cost)…")

    try:
        progress_bar.progress(10, "Scoring original resume…")
        _log_agent("Agent 2 — Validator", "run", "Merging onboarding data…")

        # Merge questionnaire responses before scoring and optimizing
        pr_merged = merge_questionnaire_responses(pr)

        # ── ATS ARCHITECT PIPELINE (non-technical CVs) ─────────────────────────
        bg = st.session_state.get("bg_classification")
        if bg and bg.get("is_non_technical"):
            bg_domain = bg["background_domain"]
            da_target = st.session_state.get("da_target_role", "Data Analyst")
            progress_bar.progress(15, f"🔍 ATS Architect: Analyzing {bg_domain} background…")
            _log_agent("Agent 1 — Intake", "done",
                       f"Non-tech mode: {bg_domain} → {da_target}")

            # Run the 9-section architect report
            arch_report = run_ats_architect(
                resume=pr_merged,
                background_domain=bg_domain,
                target_role=da_target,
                jd=jd_parsed,
            )
            st.session_state.ats_architect_report = arch_report
            progress_bar.progress(22, "✅ ATS Architect report generated — applying reframes…")

            # Apply reframed content to pr_merged so the enhancement pipeline
            # receives the already-reframed experience bullets and DA summary
            from core.resume_parser import WorkExperience as WE
            new_experience = []
            for exp in pr_merged.experience:
                role_key = f"{exp.title or 'Role'} @ {exp.company or 'Company'}"
                reframed_bullets = arch_report.reframed_bullets.get(role_key, exp.bullets)
                new_exp = WE(
                    title=exp.title,
                    company=exp.company,
                    duration=exp.duration,
                    bullets=reframed_bullets if reframed_bullets else exp.bullets,
                )
                new_experience.append(new_exp)
            pr_merged.experience = new_experience

            # Inject DA-oriented summary
            if arch_report.reframed_summary:
                pr_merged.summary = arch_report.reframed_summary

            # Inject reframed skills (DA skills take priority)
            pr_merged.skills = list(dict.fromkeys(arch_report.reframed_skills + pr_merged.skills))[:40]

            # Update JD to target DA role if no JD was provided
            if not st.session_state.parsed_jd:
                from core.jd_engine import ParsedJD
                da_jd = ParsedJD()
                da_jd.role_title = da_target
                da_jd.domain     = "Data Analytics"
                da_jd.seniority  = "junior"
                from core.non_tech_ats_architect import ALL_DA_KEYWORDS
                da_jd.all_keywords   = ALL_DA_KEYWORDS
                da_jd.required_skills = ALL_DA_KEYWORDS[:20]
                da_jd.must_have      = ["sql", "excel", "data visualization", "reporting", "power bi"]
                da_jd.tech_stack     = ["sql", "excel", "python", "tableau", "power bi"]
                da_jd.responsibilities = [
                    "Analyze data from multiple sources to generate business insights",
                    "Build dashboards and reports for stakeholders",
                    "Perform data cleaning, transformation, and EDA",
                    "Present findings to non-technical stakeholders",
                ]
                jd_parsed = da_jd


        score_pre = score_resume(pr_merged, jd_parsed)
        st.session_state.score_result = score_pre
        st.session_state.score_result_input = score_pre

        _log_agent("Agent 2 — Validator",        "done", f"Skills={len(pr_merged.skills)}, bullets={sum(len(e.bullets) for e in pr_merged.experience)}")
        _log_agent("Agent 3 — Gap Analyst",       "run",  f"Missing={len(score_pre.missing_skills)} keywords…")
        progress_bar.progress(25, "Gap analysis (Agent 3)…")

        _log_agent("Agent 4a — Summary Rewriter", "run",  f"Rewriting via {backend_label}…")
        progress_bar.progress(35, f"Rewriting summary ({backend_label})…")

        _log_agent("Agent 4b — Bullet Rewriter",  "run",  "Rewriting experience bullets…")
        progress_bar.progress(50, "Rewriting bullets (Agent 4b)…")

        _log_agent("Agent 4c — Project Enhancer", "run",  "Enhancing project descriptions…")
        progress_bar.progress(65, "Enhancing projects (Agent 4c)…")

        _log_agent("Agent 5 — Skill Booster",     "run",  "Injecting JD keywords for ATS 90+…")
        progress_bar.progress(78, "Skill boost (Agent 5)…")

        def loop_progress_callback(iteration, text):
            val = min(78 + iteration * 5, 95)
            progress_bar.progress(val, text)

        force_one_page = (st.session_state.page_target_selector == "Strict 1-Page (Recommended)")
        from core.enhancement_engine import optimize_resume_loop
        enhanced, opt_log = optimize_resume_loop(
            resume=pr_merged,
            jd=jd_parsed,
            scores=score_pre,
            onboarding=st.session_state.onboarding,
            rewrite_all=True,
            progress_callback=loop_progress_callback,
            force_one_page=force_one_page
        )
        st.session_state.enhanced_resume = enhanced
        st.session_state.optimization_log = opt_log

        _log_agent("Agent 3 — Gap Analyst",       "done", f"{len(score_pre.missing_skills)} gaps identified")
        _log_agent("Agent 4a — Summary Rewriter", "done", "Summary ✓")
        _log_agent("Agent 4b — Bullet Rewriter",  "done", f"{sum(len(e.bullets) for e in enhanced.enhanced_experience)} bullets ✓")
        _log_agent("Agent 4c — Project Enhancer", "done", f"{len(enhanced.enhanced_projects)} projects enhanced")
        _log_agent("Agent 5 — Skill Booster",     "done", f"{len(enhanced.enhanced_skills)} skills in output")

        progress_bar.progress(95, "QA validation (Agent 6)…")
        _log_agent("Agent 6 — QA Guard", "done", "Contact integrity ✓")

        progress_bar.progress(97, "Scoring optimized resume…")
        pr_post = ParsedResume()
        pr_post.name      = enhanced.name
        pr_post.email     = enhanced.email
        pr_post.phone     = enhanced.phone
        pr_post.linkedin  = enhanced.linkedin
        pr_post.github    = enhanced.github
        pr_post.portfolio = getattr(enhanced, "portfolio", "")
        pr_post.summary   = enhanced.enhanced_summary
        pr_post.skills    = enhanced.enhanced_skills
        pr_post.experience = enhanced.enhanced_experience
        pr_post.projects   = enhanced.enhanced_projects
        pr_post.education  = enhanced.education
        pr_post.certifications = enhanced.certifications
        pr_post.accomplishments = enhanced.accomplishments
        pr_post.publications = enhanced.publications
        
        sections = ["experience","skills","projects","education","summary"]
        if enhanced.certifications: sections.append("certifications")
        if enhanced.accomplishments: sections.append("accomplishments")
        if enhanced.publications: sections.append("publications")
        pr_post.sections_found = sections
        pr_post.word_count = (
            len(enhanced.enhanced_summary.split()) +
            sum(len(b.split()) for e in enhanced.enhanced_experience for b in e.bullets) +
            len(enhanced.enhanced_skills) * 2
        )
        pr_post.raw_text = (
            " ".join(enhanced.enhanced_skills) + " " +
            enhanced.enhanced_summary + " " +
            " ".join(b for e in enhanced.enhanced_experience for b in e.bullets)
        )
        score_post = score_resume(pr_post, jd_parsed)
        st.session_state.score_result = score_post

        progress_bar.progress(97, "Generating HTML resume…")
        html = generate_html_resume(enhanced)
        st.session_state.html_resume = html

        try:
            st.session_state.pdf_bytes = html_to_pdf_bytes(html)
        except Exception:
            st.session_state.pdf_bytes = None

        progress_bar.progress(100, "Done!")
        st.success(f"✅ Resume enhanced via **{backend_label}** (zero API cost)! ATS Score: **{round(score_post.overall_score*100)}/100**")
        st.rerun()

    except RuntimeError as e:
        err_msg = str(e)
        progress_bar.empty()
        for name in agent_names:
            _log_agent(name, "err", "Backend unreachable")

        if LLM_BACKEND.lower() == "ollama":
            st.error(f"""
**Ollama not running or model not available.**

1. Open terminal: `ollama serve`
2. Pull model: `ollama pull {OLLAMA_MODEL}`
3. Click Generate again

Error: `{err_msg}`
""")
        elif LLM_BACKEND.lower() == "huggingface":
            st.error(f"""
**HuggingFace model load failed.**

Check:
- `pip install transformers torch accelerate bitsandbytes`
- HF_MODEL_PATH in config.py is correct
- Enough RAM/VRAM (use HF_LOAD_IN_4BIT=True to reduce memory)

Error: `{err_msg}`
""")
        elif LLM_BACKEND.lower() == "finetuned":
            st.error(f"""
**Fine-tuned model load failed.**

Check:
- FINETUNED_ADAPTER_PATH or FINETUNED_MERGED_PATH in config.py
- `pip install transformers torch accelerate bitsandbytes peft`

Error: `{err_msg}`
""")
        else:
            st.error(f"Pipeline error: {err_msg}")

    except Exception as e:
        progress_bar.empty()
        st.error(f"Unexpected error: {e}")
        import traceback
        st.code(traceback.format_exc())
