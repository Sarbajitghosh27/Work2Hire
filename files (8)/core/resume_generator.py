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
from dataclasses import dataclass, field
from typing import List

from core.enhancement_engine import EnhancedResume


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


# ══════════════════════════════════════════════════════════════════════════════
# HTML SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _header(e: EnhancedResume) -> str:
    name = _e(e.name or "Your Name")
    contact_parts = []
    if e.phone:
        contact_parts.append(f'<span>📱 {_e(e.phone)}</span>')
    if e.email:
        contact_parts.append(f'<a href="mailto:{_e(e.email)}">✉ {_e(e.email)}</a>')
    if e.linkedin:
        li = e.linkedin.replace("https://","").replace("http://","").replace("www.","")
        contact_parts.append(f'<a href="{_e(e.linkedin)}" target="_blank">in {_e(li)}</a>')
    if e.github:
        gh = e.github.replace("https://","").replace("http://","").replace("www.","")
        contact_parts.append(f'<a href="{_e(e.github)}" target="_blank">⌥ {_e(gh)}</a>')
    if getattr(e, "portfolio", None):
        pf = e.portfolio.replace("https://","").replace("http://","").replace("www.","")
        contact_parts.append(f'<a href="{_e(e.portfolio)}" target="_blank">🔗 {_e(pf)}</a>')

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
        tech    = ", ".join(_e(t) for t in proj.tech_used if t)
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
      <span class="job-date"></span>
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

    # Case-insensitive dedup before rendering
    seen: set = set()
    skills: list = []
    for s in e.enhanced_skills:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            skills.append(s)

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
    items = [a for a in e.accomplishments if a and len(a) > 5]
    if not items:
        return ""
    li = "".join(f"<li>{_e(a)}</li>" for a in items)
    return f"""
<div class="section">
  {_section_title("Achievements & Awards")}
  <ul>{li}</ul>
</div>"""


def _certifications(e: EnhancedResume) -> str:
    items = [c for c in e.certifications if c and len(c) > 3]
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

CSS = """
/* @page margin:0 — we own ALL whitespace via body padding. */
@page { size: A4 portrait; margin: 0; }

* { box-sizing: border-box; margin: 0; padding: 0; }

html { width: 210mm; }

body {
  font-family: "Times New Roman", Times, serif;
  font-size: 11pt;
  color: #000;
  background: #fff;
  width: 210mm;
  height: 297mm;
  padding: 10mm 14mm 8mm 14mm;
  line-height: 1.38;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
  overflow: hidden;
}

/* Header */
.header { text-align: center; margin-bottom: 6.5pt; }
.name {
  font-size: 22pt;
  font-weight: bold;
  font-variant: small-caps;
  letter-spacing: 0.6px;
  margin-bottom: 2pt;
}
.contact { font-size: 9.8pt; color: #111; }
.contact a { color: #002266; text-decoration: none; }
.contact span { color: #111; }

/* Sections */
.section { margin-bottom: 5.5pt; }
.section-title {
  font-size: 11.5pt;
  font-variant: small-caps;
  font-weight: bold;
  border-bottom: 1.2pt solid #000;
  padding-bottom: 1.2pt;
  margin-bottom: 3.5pt;
  letter-spacing: 0.4px;
}

/* Summary */
.summary-text { font-size: 10.2pt; margin-left: 2mm; text-align: justify; line-height: 1.38; }

/* Jobs / Projects */
.job { margin-bottom: 4.5pt; margin-left: 2mm; page-break-inside: avoid; }
.job-header { display: flex; justify-content: space-between; align-items: baseline; }
.job-title  { font-weight: bold; font-size: 11pt; }
.job-date   { font-size: 9.8pt; white-space: nowrap; margin-left: 8pt; }
.proj-tech  { font-size: 9.2pt; font-style: italic; margin: 1.5pt 0; }
.proj-link  { font-weight: normal; font-size: 9.2pt; }
.proj-link a { color: #002266; }

/* Lists */
ul { padding-left: 4.5mm; margin-top: 1.5pt; list-style-type: none; }
ul li {
  font-size: 10.2pt;
  margin-bottom: 1.8pt;
  position: relative;
  padding-left: 2.5mm;
  line-height: 1.38;
}
ul li::before { content: "\u2013"; position: absolute; left: 0; }

/* Skills */
.skills-list { list-style: none; padding-left: 2mm; }
.skills-list li::before { content: ""; }
.skills-list li { font-size: 10.2pt; margin-bottom: 1.8pt; line-height: 1.38; }

/* Education */
.edu-table { margin-left: 2mm; }
.edu-row { display: flex; gap: 8pt; margin-bottom: 3.5pt; font-size: 10.2pt; line-height: 1.38; }
.edu-row span:first-child { min-width: 75pt; font-weight: bold; }

/* Screen preview */
@media screen {
  body { margin: 20px auto; box-shadow: 0 2px 20px rgba(0,0,0,0.20); }
}

/* Print */
@media print {
  body { margin: 0; box-shadow: none; height: 297mm; overflow: hidden; }
  body.two-page { height: auto !important; overflow: visible !important; }
  a { color: #000 !important; }
  .job { page-break-inside: avoid; }
}

/* Two-page support */
body.two-page {
  height: auto !important;
  overflow: visible !important;
}

/* Dense Layout overrides */
body.dense-layout {
  font-size: 10pt !important;
  padding: 6mm 10mm 4mm 10mm !important;
  line-height: 1.25 !important;
}
body.dense-layout .header {
  margin-bottom: 4pt !important;
}
body.dense-layout .name {
  font-size: 18pt !important;
}
body.dense-layout .section {
  margin-bottom: 3.5pt !important;
}
body.dense-layout .section-title {
  font-size: 10.5pt !important;
  padding-bottom: 0.8pt !important;
  margin-bottom: 2pt !important;
}
body.dense-layout .summary-text {
  font-size: 9.2pt !important;
  line-height: 1.25 !important;
}
body.dense-layout .job {
  margin-bottom: 2.5pt !important;
}
body.dense-layout .job-title {
  font-size: 10pt !important;
}
body.dense-layout .job-date {
  font-size: 9pt !important;
}
body.dense-layout ul li {
  font-size: 9.2pt !important;
  margin-bottom: 1.0pt !important;
  line-height: 1.25 !important;
}
body.dense-layout .skills-list li {
  font-size: 9.2pt !important;
  margin-bottom: 1.0pt !important;
  line-height: 1.25 !important;
}
body.dense-layout .edu-row {
  margin-bottom: 2pt !important;
  font-size: 9.2pt !important;
  line-height: 1.25 !important;
}

/* Loose Layout overrides */
body.loose-layout {
  font-size: 11.5pt !important;
  padding: 14mm 16mm 12mm 16mm !important;
  line-height: 1.45 !important;
}
body.loose-layout .header {
  margin-bottom: 10pt !important;
}
body.loose-layout .name {
  font-size: 24pt !important;
}
body.loose-layout .section {
  margin-bottom: 10pt !important;
}
body.loose-layout .section-title {
  font-size: 12.5pt !important;
  padding-bottom: 2.0pt !important;
  margin-bottom: 6.0pt !important;
}
body.loose-layout .summary-text {
  font-size: 10.8pt !important;
  line-height: 1.45 !important;
}
body.loose-layout .job {
  margin-bottom: 8.0pt !important;
}
body.loose-layout .job-title {
  font-size: 11.5pt !important;
}
body.loose-layout .job-date {
  font-size: 10.5pt !important;
}
body.loose-layout .proj-tech {
  font-size: 9.8pt !important;
}
body.loose-layout ul li {
  font-size: 10.8pt !important;
  margin-bottom: 3.0pt !important;
  line-height: 1.45 !important;
}
body.loose-layout .skills-list li {
  font-size: 10.8pt !important;
  margin-bottom: 3.0pt !important;
  line-height: 1.45 !important;
}
body.loose-layout .edu-row {
  margin-bottom: 5pt !important;
  font-size: 10.8pt !important;
  line-height: 1.45 !important;
}

/* Super-Dense Layout overrides */
body.super-dense-layout {
  font-size: 9pt !important;
  padding: 4mm 8mm 3mm 8mm !important;
  line-height: 1.15 !important;
}
body.super-dense-layout .header {
  margin-bottom: 2.5pt !important;
}
body.super-dense-layout .name {
  font-size: 15pt !important;
}
body.super-dense-layout .section {
  margin-bottom: 2.0pt !important;
}
body.super-dense-layout .section-title {
  font-size: 9.5pt !important;
  padding-bottom: 0.5pt !important;
  margin-bottom: 1.0pt !important;
  border-bottom-width: 0.8pt !important;
}
body.super-dense-layout .summary-text {
  font-size: 8.5pt !important;
  line-height: 1.15 !important;
}
body.super-dense-layout .job {
  margin-bottom: 1.5pt !important;
}
body.super-dense-layout .job-title {
  font-size: 9pt !important;
}
body.super-dense-layout .job-date {
  font-size: 8pt !important;
}
body.super-dense-layout .proj-tech {
  font-size: 7.8pt !important;
}
body.super-dense-layout ul li {
  font-size: 8.5pt !important;
  margin-bottom: 0.5pt !important;
  line-height: 1.15 !important;
}
body.super-dense-layout .skills-list li {
  font-size: 8.5pt !important;
  margin-bottom: 0.5pt !important;
  line-height: 1.15 !important;
}
body.super-dense-layout .edu-row {
  margin-bottom: 1pt !important;
  font-size: 8.5pt !important;
  line-height: 1.15 !important;
}
/* Expanded Layout overrides */
body.expanded-layout {
  font-size: 12pt !important;
  padding: 20mm 18mm 18mm 18mm !important;
  line-height: 1.5 !important;
}
body.expanded-layout .header {
  margin-bottom: 14pt !important;
}
body.expanded-layout .name {
  font-size: 28pt !important;
}
body.expanded-layout .section {
  margin-bottom: 14pt !important;
}
body.expanded-layout .section-title {
  font-size: 14pt !important;
  padding-bottom: 2.5pt !important;
  margin-bottom: 8.0pt !important;
}
body.expanded-layout .summary-text {
  font-size: 11.5pt !important;
  line-height: 1.5 !important;
}
body.expanded-layout .job {
  margin-bottom: 11.0pt !important;
}
body.expanded-layout .job-title {
  font-size: 12pt !important;
}
body.expanded-layout .job-date {
  font-size: 11pt !important;
}
body.expanded-layout .proj-tech {
  font-size: 10.5pt !important;
}
body.expanded-layout ul li {
  font-size: 11.2pt !important;
  margin-bottom: 4.5pt !important;
  line-height: 1.5 !important;
}
body.expanded-layout .skills-list li {
  font-size: 11.2pt !important;
  margin-bottom: 4.5pt !important;
  line-height: 1.5 !important;
}
body.expanded-layout .edu-row {
  margin-bottom: 7pt !important;
  font-size: 11.2pt !important;
  line-height: 1.5 !important;
}
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
    
    layout_style = getattr(enhanced, "layout_style", "normal")
    body_classes = []
    if layout_style == "dense":
        body_classes.append("dense-layout")
    elif layout_style == "super_dense":
        body_classes.append("super-dense-layout")
    elif layout_style == "loose":
        body_classes.append("loose-layout")
    elif layout_style == "expanded":
        body_classes.append("expanded-layout")
    elif getattr(enhanced, "use_dense_spacing", False):
        body_classes.append("dense-layout")
        
    if getattr(enhanced, "is_two_page", False):
        body_classes.append("two-page")
    body_class = " ".join(body_classes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_e(enhanced.name)} — Resume</title>
<style>{CSS}</style>
</head>
<body class="{body_class}">
{body}
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION via wkhtmltopdf
# ══════════════════════════════════════════════════════════════════════════════

def html_to_pdf_bytes(html: str) -> bytes:
    """
    Converts HTML string to PDF bytes using wkhtmltopdf.
    Returns PDF bytes or raises RuntimeError if wkhtmltopdf not found.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            f.write(html)
            html_path = f.name

        pdf_path = html_path.replace(".html", ".pdf")

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

        if result.returncode != 0:
            raise RuntimeError(
                f"wkhtmltopdf failed:\n{result.stderr.decode()}"
            )

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        return pdf_bytes

    except FileNotFoundError:
        raise RuntimeError(
            "wkhtmltopdf not found.\n"
            "Install: sudo apt-get install wkhtmltopdf\n"
            "Or on Windows: https://wkhtmltopdf.org/downloads.html"
        )
    finally:
        for p in [html_path, pdf_path]:
            try:
                os.unlink(p)
            except Exception:
                pass
