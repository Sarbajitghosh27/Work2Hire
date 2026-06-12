"""
core/resume_generator.py — v4
──────────────────────────────
Generates a pixel-perfect HTML resume matching the Jitin Nair LaTeX template.
Produces a real PDF via wkhtmltopdf (no LaTeX needed).
Contact info injected ONLY from structured fields — never from section content.

Upgrade 7: Domain-aware section ordering.
"""

import html as _html
import subprocess
import tempfile
import os
import logging
from dataclasses import dataclass, field
from typing import List

from core.enhancement_engine import EnhancedResume
from config import canonical_skill_name

logger = logging.getLogger(__name__)



def _e(text) -> str:
    """HTML-escape."""
    return _html.escape(str(text or ""), quote=True)


def _link(url: str, label: str = "") -> str:
    if not url:
        return ""
    url   = url.strip()
    clean = url.replace("https://", "").replace("http://", "").replace("www.", "")
    disp  = _e(label or clean)
    return f'<a href="{_e(url)}" target="_blank">{disp}</a>'


def is_valid_certification(c: str) -> bool:
    c_low = c.lower().strip()
    if len(c) > 100 or len(c.split()) > 12:
        return False
    banned_keywords = [
        "does not list",
        "no certifications",
        "no separate certifications",
        "however, it does include",
        "not specify certifications",
        "not list certifications",
        "no online courses",
        "does not specify",
        "no formal certifications",
        "n/a",
        "none",
        "not applicable",
        "available upon request",
        "refer to",
        "see project",
        "no direct certifications",
        "not yet certified",
        "has not listed",
        "does not have",
        "not listed",
        "no separate",
        "no online",
        "will be provided",
        "explanatory text",
        "source document",
        "explicitly mentioned",
        "no direct",
        "no certifications are",
        "refer to ",
        "nil"
    ]
    for kw in banned_keywords:
        if kw in c_low:
            return False
    return True


def is_valid_accomplishment(a: str) -> bool:
    a_low = a.lower().strip()
    if len(a) > 150 or len(a.split()) > 20:
        return False
    banned_keywords = [
        "does not list",
        "no accomplishments",
        "no separate accomplishments",
        "no achievements",
        "no awards",
        "however, it does include",
        "not specify accomplishments",
        "not list accomplishments",
        "does not specify",
        "no formal accomplishments",
        "no formal achievements",
        "n/a",
        "none",
        "not applicable",
        "available upon request",
        "refer to",
        "see project",
        "no direct achievements",
        "no direct accomplishments",
        "has not listed",
        "does not have",
        "not listed",
        "will be provided",
        "explanatory text",
        "source document",
        "explicitly mentioned",
        "nil"
    ]
    for kw in banned_keywords:
        if kw in a_low:
            return False
    return True



# ══════════════════════════════════════════════════════════════════════════════
# HTML SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _header(e: EnhancedResume) -> str:
    name = _e(e.name or "Your Name")
    contact_parts = []
    
    phone_svg = '<svg viewBox="0 0 24 24" class="icon" xmlns="http://www.w3.org/2000/svg"><path d="M6.62 10.79a15.15 15.15 0 0 0 6.57 6.57l2.2-2.2a1 1 0 0 1 .76-.29c1.07.12 2.18.12 3.23-.23a1 1 0 0 1 1.25.95v3.42a1 1 0 0 1-1.07 1A17 17 0 0 1 3 4.07a1 1 0 0 1 1-1.07h3.42a1 1 0 0 1 .95 1.25c-.35 1.05-.35 2.16-.23 3.23a1 1 0 0 1-.29.76l-2.2 2.2z"/></svg>'
    email_svg = '<svg viewBox="0 0 24 24" class="icon" xmlns="http://www.w3.org/2000/svg"><path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>'
    linkedin_svg = '<svg viewBox="0 0 24 24" class="icon" xmlns="http://www.w3.org/2000/svg"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.79M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>'
    github_svg = '<svg viewBox="0 0 24 24" class="icon" xmlns="http://www.w3.org/2000/svg"><path d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.9-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"/></svg>'
    portfolio_svg = '<svg viewBox="0 0 24 24" class="icon" xmlns="http://www.w3.org/2000/svg"><path d="M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 5-5s-2.24-5-5-5z"/></svg>'

    import re
    if e.phone:
        disp_phone = re.sub(r"[^\d+\s()-]", "", e.phone).strip()
        clean_phone = re.sub(r"[^\d+]", "", e.phone)
        if disp_phone:
            contact_parts.append(f'<a href="tel:{_e(clean_phone)}">{phone_svg}{_e(disp_phone)}</a>')
    if e.email:
        disp_email = re.sub(r"[^\w@.+-]", "", e.email).strip()
        if disp_email:
            contact_parts.append(f'<a href="mailto:{_e(disp_email)}">{email_svg}{_e(disp_email)}</a>')
    if e.linkedin:
        li = e.linkedin.replace("https://","").replace("http://","").replace("www.","")
        li = re.sub(r"^[^a-zA-Z0-9]+", "", li).strip()
        if li.lower().startswith("in "):
            li = li[3:].strip()
        li_url = e.linkedin.strip()
        if not li_url.startswith("http"):
            li_url = "https://" + li_url
        if li:
            contact_parts.append(f'<a href="{_e(li_url)}" target="_blank">{linkedin_svg}{_e(li)}</a>')
    if e.github:
        gh = e.github.replace("https://","").replace("http://","").replace("www.","")
        gh = re.sub(r"^[^a-zA-Z0-9]+", "", gh).strip()
        gh_url = e.github.strip()
        if not gh_url.startswith("http"):
            gh_url = "https://" + gh_url
        if gh:
            contact_parts.append(f'<a href="{_e(gh_url)}" target="_blank">{github_svg}{_e(gh)}</a>')
    if getattr(e, "portfolio", None):
        pf = e.portfolio.replace("https://","").replace("http://","").replace("www.","")
        pf = re.sub(r"^[^a-zA-Z0-9]+", "", pf).strip()
        pf_url = e.portfolio.strip()
        if not pf_url.startswith("http"):
            pf_url = "https://" + pf_url
        if pf:
            contact_parts.append(f'<a href="{_e(pf_url)}" target="_blank">{portfolio_svg}{_e(pf)}</a>')

    contact_line = " &nbsp;|&nbsp; ".join(contact_parts)
    return f"""
<div class="header">
  <div class="name">{name}</div>
  <div class="contact">{contact_line}</div>
</div>"""


def _section_title(title: str) -> str:
    return f'<div class="section-title">{_e(title)}</div>'


def _summary(e: EnhancedResume) -> str:
    if not e.enhanced_summary:
        return ""
    return f"""
<div class="section">
  {_section_title("Professional Summary")}
  <p class="summary-text">{_e(e.enhanced_summary)}</p>
</div>"""


def _experience(e: EnhancedResume) -> str:
    if not e.enhanced_experience:
        return ""
    items = []
    for exp in e.enhanced_experience:
        title   = _e(exp.title or "")
        company = _e(exp.company or "")
        dur     = _e(exp.duration or "")
        # Skip completely empty entries
        if not title and not company:
            continue
        # Format: Title — Company (preferred) or just whichever is present
        if title and company:
            heading = f"{title} &#8212; {company}"
        elif title:
            heading = title
        else:
            heading = company
        bullets_html = "".join(
            f"<li>{_e(b)}</li>"
            for b in exp.bullets if b and len(b) > 5
        )
        items.append(f"""
  <div class="job">
    <div class="job-header">
      <span class="job-title">{heading}</span>
      <span class="job-date">{dur}</span>
    </div>
    {"<ul>" + bullets_html + "</ul>" if bullets_html else ""}
  </div>""")
    if not items:
        return ""
    return f"""
<div class="section">
  {_section_title("Professional Experience")}
  {"".join(items)}
</div>"""


def _projects(e: EnhancedResume) -> str:
    if not e.enhanced_projects:
        return ""
    items = []
    for proj in e.enhanced_projects:
        name    = _e(proj.name or "")
        tech    = ", ".join(_e(canonical_skill_name(t)) for t in proj.tech_used if t)
        desc    = _e(proj.description or "")
        outcome = _e(proj.outcome or "")
        link_html = (f' <span class="proj-link">— {_link(proj.link, "link")}</span>'
                     if proj.link else "")

        content = ""
        if desc:
            content += f"<li>{desc}</li>"
        if outcome and outcome != desc:
            content += f"<li>{outcome}</li>"

        items.append(f"""
  <div class="job">
    <div class="job-header">
      <span class="job-title">{name}{link_html}</span>
      <span class="job-date">{_e(proj.duration or "")}</span>
    </div>
    {f'<div class="proj-tech">Tech Stack: {tech}</div>' if tech else ""}
    {"<ul>" + content + "</ul>" if content else ""}
  </div>""")
    return f"""
<div class="section">
  {_section_title("Projects")}
  {"".join(items)}
</div>"""


def _skills(e: EnhancedResume) -> str:
    if not e.enhanced_skills:
        return ""

    # Case-insensitive dedup before rendering + apply capitalization
    seen: set = set()
    skills: list = []
    for s in e.enhanced_skills:
        s_cap = canonical_skill_name(s)
        key = s_cap.lower().strip()
        if key and key not in seen:
            seen.add(key)
            skills.append(s_cap)

    # Group into logical buckets for a clean, ATS-friendly layout
    LANGUAGES_KW = {
        "python", "r", "java", "javascript", "typescript", "c++", "c", "c#",
        "go", "rust", "kotlin", "swift", "scala", "bash", "matlab", "ruby", "php", "perl", "dart", "html", "css", "solidity", "haskell", "assembly"
    }
    FRAMEWORKS_KW = {
        "react", "next.js", "nextjs", "angular", "vue", "node.js", "nodejs", "express", "flask", "django", "fastapi",
        "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "opencv", "nltk", "spacy", "huggingface", "transformers", "langchain", "spring boot", "spring", "dotnet", ".net",
        "laravel", "ruby on rails", "rails", "flutter", "react native", "svelte", "jquery", "bootstrap", "tailwind",
        "tailwindcss", "graphql", "redux", "jest", "pytest", "junit"
    }
    DATABASES_KW = {
        "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra", "dynamodb", "oracle", "ms sql",
        "sql server", "mariadb", "neo4j", "couchdb", "elasticsearch", "firebase", "bigquery", "snowflake", "redshift",
        "hive", "impala", "db2"
    }
    CLOUD_KW = {
        "aws", "gcp", "azure", "google cloud", "amazon web services", "microsoft azure", "heroku", "digitalocean",
        "cloudflare", "lambda", "ec2", "s3", "rds", "route 53", "iam", "ecs", "eks"
    }

    languages = []
    frameworks = []
    databases = []
    clouds = []
    tools = []

    for s in skills:
        s_low = s.lower().strip()
        if s_low in LANGUAGES_KW:
            languages.append(s)
        elif s_low in FRAMEWORKS_KW:
            frameworks.append(s)
        elif s_low in DATABASES_KW:
            databases.append(s)
        elif s_low in CLOUD_KW:
            clouds.append(s)
        else:
            tools.append(s)

    rows = []
    if languages:
        rows.append(("Programming Languages", ", ".join(_e(s) for s in languages)))
    if frameworks:
        rows.append(("Frameworks &amp; Libraries", ", ".join(_e(s) for s in frameworks)))
    if databases:
        rows.append(("Databases", ", ".join(_e(s) for s in databases)))
    if tools:
        rows.append(("Tools &amp; Technologies", ", ".join(_e(s) for s in tools)))
    if clouds:
        rows.append(("Cloud Platforms", ", ".join(_e(s) for s in clouds)))

    if not rows:
        all_skills = ", ".join(_e(s) for s in skills)
        return f"""
<div class="section">
  {_section_title("Technical Skills")}
  <ul class="skills-list"><li>{all_skills}</li></ul>
</div>"""

    rows_html = "".join(
        f"<li><strong>{label}:</strong> {vals}</li>"
        for label, vals in rows
    )
    return f"""
<div class="section">
  {_section_title("Technical Skills")}
  <ul class="skills-list">{rows_html}</ul>
</div>"""


def _education(e: EnhancedResume) -> str:
    import re
    if not e.education:
        return ""
        
    # Deduplicate e.education entries before rendering (case-insensitive, alphanumeric check)
    cleaned_entries = []
    for edu in e.education:
        found_idx = -1
        edu_deg = (edu.degree or "").strip()
        edu_inst = (edu.institution or "").strip()
        edu_yr = (edu.year or "").strip()
        edu_gpa = (edu.gpa or "").strip()
        edu_norm = re.sub(r"[^a-z0-9]", "", (edu_deg + edu_inst).lower())

        for idx, existing in enumerate(cleaned_entries):
            ex_deg = (existing.degree or "").strip()
            ex_inst = (existing.institution or "").strip()
            ex_norm = re.sub(r"[^a-z0-9]", "", (ex_deg + ex_inst).lower())

            # Check year conflict
            yr1 = re.sub(r"[^0-9]", "", edu_yr)
            yr2 = re.sub(r"[^0-9]", "", (existing.year or ""))
            years_conflict = False
            if yr1 and yr2:
                y1_set = set(re.findall(r"\b\d{4}\b", edu_yr))
                y2_set = set(re.findall(r"\b\d{4}\b", existing.year or ""))
                if y1_set and y2_set and not (y1_set & y2_set):
                    years_conflict = True

            if years_conflict:
                continue

            is_match = False
            if edu_norm and ex_norm:
                if edu_norm == ex_norm or edu_norm in ex_norm or ex_norm in edu_norm:
                    is_match = True

            inst1 = re.sub(r"[^a-z0-9]", "", edu_inst.lower())
            inst2 = re.sub(r"[^a-z0-9]", "", ex_inst.lower())
            if inst1 and inst2 and (inst1 in inst2 or inst2 in inst1):
                is_match = True

            if not inst1 and inst2:
                deg1_norm = re.sub(r"[^a-z0-9]", "", edu_deg.lower())
                if inst2 in deg1_norm:
                    is_match = True
            if not inst2 and inst1:
                deg2_norm = re.sub(r"[^a-z0-9]", "", ex_deg.lower())
                if inst1 in deg2_norm:
                    is_match = True

            if is_match:
                found_idx = idx
                break

        if found_idx != -1:
            existing = cleaned_entries[found_idx]
            if len(edu_deg) > len(existing.degree or ""):
                existing.degree = edu_deg
            if len(edu_inst) > len(existing.institution or ""):
                existing.institution = edu_inst
            if not existing.year and edu_yr:
                existing.year = edu.year
            if not existing.gpa and edu_gpa:
                existing.gpa = edu.gpa
        else:
            from copy import copy
            cleaned_entries.append(copy(edu))

    rows = []
    for edu in cleaned_entries:
        deg  = (edu.degree or "").strip()
        inst = (edu.institution or "").strip()
        yr   = (edu.year or "").strip()
        
        # Clean empty parentheses ()
        deg = re.sub(r"\s*\(\s*\)", "", deg).strip()
        inst = re.sub(r"\s*\(\s*\)", "", inst).strip()
        
        # Prevent duplication if inst is identical or contained in deg (case-insensitive, alphanumeric check)
        if deg and inst:
            deg_comp = re.sub(r"[^a-z0-9]", "", deg.lower())
            inst_comp = re.sub(r"[^a-z0-9]", "", inst.lower())
            
            def make_fuzzy_pattern(s: str) -> str:
                chunks = re.findall(r"[a-z0-9]+", s.lower())
                if not chunks:
                    return ""
                return r"[^a-z0-9]*".join(re.escape(c) for c in chunks)
                
            inst_pat = make_fuzzy_pattern(inst)
            deg_pat = make_fuzzy_pattern(deg)
            
            if deg_comp == inst_comp:
                deg = ""
            elif inst_comp in deg_comp and inst_pat:
                deg = re.sub(inst_pat, "", deg, flags=re.I).strip()
                deg = re.sub(r"\s*[-–—|/,\s]+\s*$", "", deg).strip()
                deg = re.sub(r"^\s*[-–—|/,\s]+", "", deg).strip()
            elif deg_comp in inst_comp and deg_pat:
                inst = re.sub(deg_pat, "", inst, flags=re.I).strip()
                inst = re.sub(r"\s*[-–—|/,\s]+\s*$", "", inst).strip()
                inst = re.sub(r"^\s*[-–—|/,\s]+", "", inst).strip()
                
        deg  = _e(deg)
        inst = _e(inst)
        yr   = _e(yr)
        gpa  = f" &mdash; {_e(edu.gpa)}" if edu.gpa else ""
        
        # Skip completely empty entries
        if not deg and not inst:
            continue
        inst_part = f"<strong>{inst}</strong>" if inst else ""
        deg_part  = f"{deg}{gpa}" if deg else ""
        if deg_part and inst_part:
            content = f"{deg_part} &mdash; {inst_part}"
        else:
            content = deg_part or inst_part
        rows.append(f"""
  <div class="edu-row">
    <span>{yr}</span>
    <span>{content}</span>
  </div>""")
    if not rows:
        return ""
    return f"""
<div class="section">
  {_section_title("Education")}
  <div class="edu-table">{"".join(rows)}</div>
</div>"""


def _accomplishments(e: EnhancedResume) -> str:
    items = [a for a in e.accomplishments if a and len(a) > 5 and is_valid_accomplishment(a)]
    if not items:
        return ""
    li = "".join(f"<li>{_e(a)}</li>" for a in items)
    return f"""
<div class="section">
  {_section_title("Achievements & Awards")}
  <ul>{li}</ul>
</div>"""


def _certifications(e: EnhancedResume) -> str:
    items = [c for c in e.certifications if c and len(c) > 3 and is_valid_certification(c)]
    if not items:
        return ""
    li = "".join(f"<li>{_e(c)}</li>" for c in items)
    return f"""
<div class="section">
  {_section_title("Certifications")}
  <ul>{li}</ul>
</div>"""


def _publications(e: EnhancedResume) -> str:
    items = [p for p in e.publications if p and len(p) > 5]
    if not items:
        return ""
    li = "".join(f"<li>{_e(p)}</li>" for p in items)
    return f"""
<div class="section">
  {_section_title("Publications & Research")}
  <ul>{li}</ul>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# CSS — Jitin Nair template style
# ══════════════════════════════════════════════════════════════════════════════

def generate_dynamic_css(p: dict) -> str:
    return f"""
/* @page margin:0 — we own ALL whitespace via body padding. */
@page {{ size: A4 portrait; margin: 0; }}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

html {{ width: 210mm; }}

body {{
  font-family: "Times New Roman", Times, serif;
  font-size: {p["body_font_size"]}pt;
  color: #000;
  background: #fff;
  width: 210mm;
  height: 297mm;
  padding: {p["body_padding_top"]}mm {p["body_padding_horizontal"]}mm {p["body_padding_bottom"]}mm {p["body_padding_horizontal"]}mm;
  line-height: {p["body_line_height"]};
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
  overflow: hidden;
}}

/* Header */
.header {{ text-align: center; margin-bottom: {p["header_margin_bottom"]}pt; }}
.name {{
  font-size: {p["name_font_size"]}pt;
  font-weight: bold;
  font-variant: small-caps;
  letter-spacing: 0.6px;
  margin-bottom: 2pt;
}}
.contact {{ font-size: 9.8pt; color: #111; }}
.contact a {{ color: #002266; text-decoration: none; }}
.contact span {{ color: #111; }}
.icon {{
  width: 9.8pt;
  height: 9.8pt;
  vertical-align: -1.2pt;
  fill: currentColor;
  display: inline-block;
  margin-right: 3.5pt;
}}

/* Sections */
.section {{ margin-bottom: {p["section_margin_bottom"]}pt; }}
.section-title {{
  font-size: {p["section_title_font_size"]}pt;
  font-variant: small-caps;
  font-weight: bold;
  border-bottom: {p.get("section_title_border_width", 1.2)}pt solid #000;
  padding-bottom: {p["section_title_padding_bottom"]}pt;
  margin-bottom: {p["section_title_margin_bottom"]}pt;
  letter-spacing: 0.4px;
}}

/* Summary */
.summary-text {{ font-size: {p["summary_font_size"]}pt; margin-left: 2mm; text-align: justify; line-height: {p["summary_line_height"]}; }}

/* Jobs / Projects */
.job {{ margin-bottom: {p["job_margin_bottom"]}pt; margin-left: 2mm; page-break-inside: avoid; }}
.job-header {{ display: flex; justify-content: space-between; align-items: baseline; }}
.job-title  {{ font-weight: bold; font-size: {p["job_title_font_size"]}pt; }}
.job-date   {{ font-size: {p["job_date_font_size"]}pt; white-space: nowrap; margin-left: 8pt; }}
.proj-tech  {{ font-size: {p["proj_tech_font_size"]}pt; font-style: italic; margin: 1.5pt 0; }}
.proj-link  {{ font-weight: normal; font-size: 9.2pt; }}
.proj-link a {{ color: #002266; }}

/* Lists */
ul {{ padding-left: 4.5mm; margin-top: 1.5pt; list-style-type: none; }}
ul li {{
  font-size: {p["list_item_font_size"]}pt;
  margin-bottom: {p["list_item_margin_bottom"]}pt;
  position: relative;
  padding-left: 2.5mm;
  line-height: {p["body_line_height"]};
}}
ul li::before {{ content: "\\2013"; position: absolute; left: 0; }}

/* Skills */
.skills-list {{ list-style: none; padding-left: 2mm; }}
.skills-list li::before {{ content: ""; }}
.skills-list li {{ font-size: {p["list_item_font_size"]}pt; margin-bottom: {p["list_item_margin_bottom"]}pt; line-height: {p["body_line_height"]}; }}

/* Education */
.edu-table {{ margin-left: 2mm; }}
.edu-row {{ display: flex; gap: 8pt; margin-bottom: {p["edu_row_margin_bottom"]}pt; font-size: {p["edu_row_font_size"]}pt; line-height: {p["body_line_height"]}; }}
.edu-row span:first-child {{ min-width: 75pt; font-weight: bold; }}

/* Screen preview */
@media screen {{
  body {{ margin: 20px auto; box-shadow: 0 2px 20px rgba(0,0,0,0.20); overflow: auto !important; }}
}}


/* Print */
@media print {{
  body {{ margin: 0; box-shadow: none; height: 297mm; overflow: hidden; }}
  body.two-page {{ height: auto !important; overflow: visible !important; }}
  a {{ color: #000 !important; }}
  .job {{ page-break-inside: avoid; }}
}}

/* Two-page support */
body.two-page {{
  height: auto !important;
  overflow: visible !important;
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# FULL HTML ASSEMBLER
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_resume(enhanced: EnhancedResume) -> str:
    """
    Assembles complete self-contained HTML resume.
    Upgrade 7: Section order is determined by enhanced.section_order (domain-aware ATS).
    Falls back to default order if not set.
    """
    from config import SECTION_ORDER_BY_DOMAIN

    # Section renderers keyed by section name
    section_renderers = {
        "summary"        : lambda: _summary(enhanced),
        "experience"     : lambda: _experience(enhanced),
        "projects"       : lambda: _projects(enhanced),
        "skills"         : lambda: _skills(enhanced),
        "education"      : lambda: _education(enhanced),
        "accomplishments": lambda: _accomplishments(enhanced),
        "certifications" : lambda: _certifications(enhanced),
        "publications"   : lambda: _publications(enhanced),
    }

    # Use domain-aware order if available, else default
    order = getattr(enhanced, "section_order", None)
    if not order:
        order = SECTION_ORDER_BY_DOMAIN.get(
            enhanced.domain or "",
            SECTION_ORDER_BY_DOMAIN["default"]
        )

    # Build body in order
    parts = [_header(enhanced)]
    for sec in order:
        renderer = section_renderers.get(sec)
        if renderer:
            parts.append(renderer())

    body = "\n".join(parts)
    
    # Retrieve dynamic layout parameters from the compression agent if present
    layout_params = getattr(enhanced, "layout_params", None)
    if not layout_params:
        # Fallback to normal layout parameters
        from core.compression_agent import LAYOUTS
        layout_params = LAYOUTS["normal"]
        
    compiled_css = generate_dynamic_css(layout_params)
    
    body_classes = []
    if getattr(enhanced, "is_two_page", False):
        body_classes.append("two-page")
    body_class = " ".join(body_classes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_e(enhanced.name)} — Resume</title>
<style>{compiled_css}</style>
</head>
<body class="{body_class}">
{body}
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION via wkhtmltopdf
# ══════════════════════════════════════════════════════════════════════════════

def get_chromium_browser_path() -> str:
    import platform
    system = platform.system()
    if system == "Windows":
        paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe")
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    elif system == "Darwin":
        paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    else: # Linux
        paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/microsoft-edge",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium"
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    return None


def html_to_pdf_bytes(html: str) -> bytes:
    """
    Converts HTML string to PDF bytes.
    First tries Edge or Chrome in headless mode (very high fidelity, default on Windows/Mac).
    Falls back to wkhtmltopdf if no browser is found or if browser print fails.
    """
    html_path = None
    pdf_path = None
    temp_user_data_dir = None
    
    try:
        # Create unique temp files in a safe location (mkstemp closes descriptor to avoid Win sharing lock)
        fd_html, html_path = tempfile.mkstemp(suffix=".html")
        os.close(fd_html)
        pdf_path = html_path.replace(".html", ".pdf")
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
            
        # Try browser print first
        browser_path = get_chromium_browser_path()
        if browser_path:
            # Create a temporary user data directory to prevent profile locking issues on Windows
            temp_user_data_dir = tempfile.mkdtemp(prefix="chrome_profile_")
            cmd = [
                browser_path,
                "--headless",
                "--disable-gpu",
                f"--print-to-pdf={pdf_path}",
                f"--user-data-dir={temp_user_data_dir}",
                "--no-margins",
                html_path
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=20)
                if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                    with open(pdf_path, "rb") as f:
                        return f.read()
            except Exception as e:
                logger.warning(f"Browser PDF generation failed, falling back to wkhtmltopdf: {e}")
                
        # Fallback to wkhtmltopdf
        cmd = [
            "wkhtmltopdf",
            "--page-size", "A4",
            "--margin-top", "0mm",
            "--margin-right", "0mm",
            "--margin-bottom", "0mm",
            "--margin-left", "0mm",
            "--enable-local-file-access",
            "--quiet",
            html_path,
            pdf_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            with open(pdf_path, "rb") as f:
                return f.read()
                
        raise RuntimeError(
            f"PDF generation failed. Headless browser and wkhtmltopdf both failed.\n"
            f"Browser path: {browser_path}\n"
            f"wkhtmltopdf output: {result.stderr.decode() if 'result' in locals() else 'N/A'}"
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate PDF: {e}")
        
    finally:
        for p in [html_path, pdf_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass
        if temp_user_data_dir and os.path.exists(temp_user_data_dir):
            import shutil
            try:
                shutil.rmtree(temp_user_data_dir)
            except Exception:
                pass


