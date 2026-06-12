"""
core/compression_agent.py
────────────────────────────────────────────────────────
One-Page Compression Agent.
Intelligently ranks and compresses resume content to enforce a strict
one-page (A4) limit, prioritizing direct relevance to the target role.
Adjusts document layout density when content size is borderline.
"""

import re
import logging
from copy import deepcopy
from typing import List, Dict
from core.enhancement_engine import EnhancedResume
from core.jd_engine import ParsedJD
from core.scoring_engine import METRIC_RE
from config import canonical_skill_name

def check_pdf_page_count(e: EnhancedResume) -> int:
    """Renders the HTML resume and returns the actual PDF page count using Chromium."""
    from core.resume_generator import generate_html_resume, html_to_pdf_bytes
    html = generate_html_resume(e)
    # Temporarily force two-page class on the body so printing does not clip overflow,
    # allowing us to check the actual number of pages the content spans.
    if 'class="' in html:
        html_for_counting = html.replace('<body class="', '<body class="two-page ')
    else:
        html_for_counting = html.replace('<body>', '<body class="two-page">')
        
    try:
        pdf_bytes = html_to_pdf_bytes(html_for_counting)
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return doc.page_count
    except Exception as err:
        logger.error(f"Error in check_pdf_page_count: {err}")
        return 1


logger = logging.getLogger(__name__)

def estimate_word_count(e: EnhancedResume) -> int:
    """Estimates the printable word count of the resume."""
    count = 0
    if e.name: count += len(e.name.split())
    if e.enhanced_summary: count += len(e.enhanced_summary.split())
    
    for exp in e.enhanced_experience:
        if exp.company: count += len(exp.company.split())
        if exp.title: count += len(exp.title.split())
        for b in exp.bullets:
            count += len(b.split())
            
    for proj in e.enhanced_projects:
        if proj.name: count += len(proj.name.split())
        if proj.description: count += len(proj.description.split())
        if proj.outcome: count += len(proj.outcome.split())
        count += len(proj.tech_used)
        
    count += len(e.enhanced_skills)
    
    for edu in e.education:
        if edu.institution: count += len(edu.institution.split())
        if edu.degree: count += len(edu.degree.split())
        
    count += sum(len(a.split()) for a in e.accomplishments)
    count += sum(len(c.split()) for c in e.certifications)
    count += sum(len(p.split()) for p in e.publications)
    
    return count

def rank_item_relevance(text: str, jd_keywords: List[str]) -> float:
    """Scores how relevant a text block is based on JD keyword matching and metrics."""
    if not text:
        return 0.0
    text_lower = text.lower()
    score = 0.0
    
    # JD keyword matches
    for kw in jd_keywords:
        kw_lower = kw.lower()
        if re.search(r"\b" + re.escape(kw_lower) + r"\b", text_lower):
            score += 2.0  # basic keyword weight
            
    # Quantified metric boost
    if METRIC_RE.search(text):
        score += 3.0  # metrics are highly valued by recruiters
        
    # Sentence length sanity check (penalize overly wordy bullets)
    words = len(text.split())
    if words > 25:
        score -= 1.0
        
    return score

LAYOUTS = {
    "expanded": {
        "body_font_size": 12,
        "body_line_height": 1.5,
        "body_padding_top": 20, # mm
        "body_padding_bottom": 18, # mm
        "body_padding_horizontal": 18, # mm
        "header_margin_bottom": 14, # pt
        "name_font_size": 28, # pt
        "section_margin_bottom": 14, # pt
        "section_title_font_size": 14, # pt
        "section_title_padding_bottom": 2.5, # pt
        "section_title_margin_bottom": 8.0, # pt
        "section_title_border_width": 1.2, # pt
        "summary_font_size": 11.5, # pt
        "summary_line_height": 1.5,
        "job_margin_bottom": 11.0, # pt
        "job_title_font_size": 12, # pt
        "job_date_font_size": 11, # pt
        "proj_tech_font_size": 10.5, # pt
        "list_item_font_size": 11.2, # pt
        "list_item_margin_bottom": 4.5, # pt
        "edu_row_margin_bottom": 7.0, # pt
        "edu_row_font_size": 11.2, # pt
    },
    "loose": {
        "body_font_size": 11.5,
        "body_line_height": 1.45,
        "body_padding_top": 14, # mm
        "body_padding_bottom": 12, # mm
        "body_padding_horizontal": 16, # mm
        "header_margin_bottom": 10, # pt
        "name_font_size": 24, # pt
        "section_margin_bottom": 10, # pt
        "section_title_font_size": 12.5, # pt
        "section_title_padding_bottom": 2.0, # pt
        "section_title_margin_bottom": 6.0, # pt
        "section_title_border_width": 1.2, # pt
        "summary_font_size": 10.8, # pt
        "summary_line_height": 1.45,
        "job_margin_bottom": 8.0, # pt
        "job_title_font_size": 11.5, # pt
        "job_date_font_size": 10.5, # pt
        "proj_tech_font_size": 9.8, # pt
        "list_item_font_size": 10.8, # pt
        "list_item_margin_bottom": 3.0, # pt
        "edu_row_margin_bottom": 5.0, # pt
        "edu_row_font_size": 10.8, # pt
    },
    "normal": {
        "body_font_size": 11,
        "body_line_height": 1.38,
        "body_padding_top": 10, # mm
        "body_padding_bottom": 8, # mm
        "body_padding_horizontal": 14, # mm
        "header_margin_bottom": 6.5, # pt
        "name_font_size": 22, # pt
        "section_margin_bottom": 5.5, # pt
        "section_title_font_size": 11.5, # pt
        "section_title_padding_bottom": 1.2, # pt
        "section_title_margin_bottom": 3.5, # pt
        "section_title_border_width": 1.2, # pt
        "summary_font_size": 10.2, # pt
        "summary_line_height": 1.38,
        "job_margin_bottom": 4.5, # pt
        "job_title_font_size": 11, # pt
        "job_date_font_size": 9.8, # pt
        "proj_tech_font_size": 9.2, # pt
        "list_item_font_size": 10.2, # pt
        "list_item_margin_bottom": 1.8, # pt
        "edu_row_margin_bottom": 3.5, # pt
        "edu_row_font_size": 10.2, # pt
    },
    "dense": {
        "body_font_size": 10,
        "body_line_height": 1.25,
        "body_padding_top": 6, # mm
        "body_padding_bottom": 4, # mm
        "body_padding_horizontal": 10, # mm
        "header_margin_bottom": 4.0, # pt
        "name_font_size": 18, # pt
        "section_margin_bottom": 3.5, # pt
        "section_title_font_size": 10.5, # pt
        "section_title_padding_bottom": 0.8, # pt
        "section_title_margin_bottom": 2.0, # pt
        "section_title_border_width": 1.2, # pt
        "summary_font_size": 9.2, # pt
        "summary_line_height": 1.25,
        "job_margin_bottom": 2.5, # pt
        "job_title_font_size": 10, # pt
        "job_date_font_size": 9, # pt
        "proj_tech_font_size": 8.5, # pt
        "list_item_font_size": 9.2, # pt
        "list_item_margin_bottom": 1.0, # pt
        "edu_row_margin_bottom": 2.0, # pt
        "edu_row_font_size": 9.2, # pt
    },
    "super_dense": {
        "body_font_size": 9,
        "body_line_height": 1.15,
        "body_padding_top": 4, # mm
        "body_padding_bottom": 3, # mm
        "body_padding_horizontal": 8, # mm
        "header_margin_bottom": 2.5, # pt
        "name_font_size": 15, # pt
        "section_margin_bottom": 2.0, # pt
        "section_title_font_size": 9.5, # pt
        "section_title_padding_bottom": 0.5, # pt
        "section_title_margin_bottom": 1.0, # pt
        "section_title_border_width": 0.8, # pt
        "summary_font_size": 8.5, # pt
        "summary_line_height": 1.15,
        "job_margin_bottom": 1.5, # pt
        "job_title_font_size": 9, # pt
        "job_date_font_size": 8, # pt
        "proj_tech_font_size": 7.8, # pt
        "list_item_font_size": 8.5, # pt
        "list_item_margin_bottom": 0.5, # pt
        "edu_row_margin_bottom": 1.0, # pt
        "edu_row_font_size": 8.5, # pt
    }
}

def estimate_resume_height(e: EnhancedResume, layout_name_or_dict) -> float:
    """Estimates the total height of the generated resume in points (pt)."""
    if isinstance(layout_name_or_dict, str):
        p = LAYOUTS[layout_name_or_dict]
    else:
        p = layout_name_or_dict
    MM_TO_PT = 2.83465
    
    # Padding top + bottom
    total_height = (p["body_padding_top"] + p["body_padding_bottom"]) * MM_TO_PT
    
    # Left/right padding in pt
    lr_padding = p["body_padding_horizontal"]
    
    content_width = 595.27 - (2 * lr_padding * MM_TO_PT)
    # Different widths based on indentation structure
    job_bullet_width = content_width - (9.0 * MM_TO_PT)      # bullet indent (4.5mm + 2.5mm) + .job margin (2mm)
    general_bullet_width = content_width - (7.0 * MM_TO_PT)  # bullet indent only
    skills_content_width = content_width - (2.0 * MM_TO_PT)  # .skills-list padding-left

    
    # Helper to calculate lines of wrapped text with word wrap simulation
    def get_lines(text: str, font_size: float, width: float, is_bold: bool = False) -> int:
        if not text:
            return 0
        char_width = (0.44 if is_bold else 0.38) * font_size
        chars_per_line = max(12, int(width / char_width))
        total_lines = 0
        for part in text.split("\n"):
            part_clean = part.strip()
            if not part_clean:
                continue
            # Simple word-wrap simulation
            words = part_clean.split()
            if not words:
                continue
            current_line_len = 0
            lines_in_part = 1
            for word in words:
                word_len = len(word)
                if current_line_len + word_len + (1 if current_line_len > 0 else 0) > chars_per_line:
                    lines_in_part += 1
                    current_line_len = word_len
                else:
                    current_line_len += word_len + (1 if current_line_len > 0 else 0)
            total_lines += lines_in_part
        return total_lines

    # --- 1. Header height ---
    header_h = 0
    header_h += p["name_font_size"] * 1.2
    header_h += 2.0  # margin between name and contact
    
    contact_parts = []
    import re
    if e.phone:
        disp_phone = re.sub(r"[^\d+\s()-]", "", e.phone).strip()
        if disp_phone: contact_parts.append("  " + disp_phone)
    if e.email:
        disp_email = re.sub(r"[^\w@.+-]", "", e.email).strip()
        if disp_email: contact_parts.append("  " + disp_email)
    if e.linkedin:
        li = e.linkedin.replace("https://","").replace("http://","").replace("www.","")
        li = re.sub(r"^[^a-zA-Z0-9]+", "", li).strip()
        if li.lower().startswith("in "):
            li = li[3:].strip()
        if li: contact_parts.append("  " + li)
    if e.github:
        gh = e.github.replace("https://","").replace("http://","").replace("www.","")
        gh = re.sub(r"^[^a-zA-Z0-9]+", "", gh).strip()
        if gh: contact_parts.append("  " + gh)
    if getattr(e, "portfolio", None):
        pf = e.portfolio.replace("https://","").replace("http://","").replace("www.","")
        pf = re.sub(r"^[^a-zA-Z0-9]+", "", pf).strip()
        if pf: contact_parts.append("  " + pf)
        
    contact_text = " | ".join(contact_parts)
    contact_lines = get_lines(contact_text, 9.8, content_width, is_bold=False)
    header_h += contact_lines * 9.8 * 1.38
    header_h += p["header_margin_bottom"]
    total_height += header_h
    
    # Title height calculation
    border_w = p.get("section_title_border_width", 1.2)
    title_h = (p["section_title_font_size"] * 1.2) + p["section_title_padding_bottom"] + border_w + p["section_title_margin_bottom"]
    
    # Determine section rendering order
    from config import SECTION_ORDER_BY_DOMAIN
    order = getattr(e, "section_order", None)
    if not order:
        order = SECTION_ORDER_BY_DOMAIN.get(
            e.domain or "",
            SECTION_ORDER_BY_DOMAIN["default"]
        )
        
    for sec in order:
        if sec == "summary" and e.enhanced_summary:
            sec_h = title_h
            lines = get_lines(e.enhanced_summary, p["summary_font_size"], content_width - 2 * MM_TO_PT, is_bold=False)
            sec_h += lines * p["summary_font_size"] * p["summary_line_height"]
            sec_h += p["section_margin_bottom"]
            total_height += sec_h
            
        elif sec == "experience" and e.enhanced_experience:
            sec_h = title_h
            for exp in e.enhanced_experience:
                title = exp.title or ""
                company = exp.company or ""
                if not title and not company:
                    continue
                heading = f"{title} — {company}" if title and company else (title or company)
                header_lines = get_lines(heading, p["job_title_font_size"], content_width - 120.0 - (2.0 * MM_TO_PT), is_bold=True)
                job_h = header_lines * p["job_title_font_size"] * p["body_line_height"]
                
                # ul margin-top is 1.5pt in CSS
                bullets_h = 1.5
                for b in exp.bullets:
                    if b and len(b) > 5:
                        b_lines = get_lines(b, p["list_item_font_size"], job_bullet_width, is_bold=False)
                        bullets_h += b_lines * p["list_item_font_size"] * p["body_line_height"] + p["list_item_margin_bottom"]
                
                job_h += bullets_h
                job_h += p["job_margin_bottom"]
                sec_h += job_h
            sec_h += p["section_margin_bottom"]
            total_height += sec_h
            
        elif sec == "projects" and e.enhanced_projects:
            sec_h = title_h
            for proj in e.enhanced_projects:
                name = proj.name or ""
                header_lines = get_lines(name, p["job_title_font_size"], content_width - 120.0 - (2.0 * MM_TO_PT), is_bold=True)
                proj_h = header_lines * p["job_title_font_size"] * p["body_line_height"]
                
                if proj.tech_used:
                    tech_str = "Tech Stack: " + ", ".join(canonical_skill_name(t) for t in proj.tech_used if t)
                    # proj-tech has margin: 1.5pt 0; in CSS (adds 3.0pt total margin)
                    tech_lines = get_lines(tech_str, p["proj_tech_font_size"], content_width - 2 * MM_TO_PT, is_bold=False)
                    proj_h += tech_lines * p["proj_tech_font_size"] * p["body_line_height"] + 3.0
                    
                # ul margin-top is 1.5pt in CSS
                bullets_h = 1.5
                if proj.description:
                    d_lines = get_lines(proj.description, p["list_item_font_size"], job_bullet_width, is_bold=False)
                    bullets_h += d_lines * p["list_item_font_size"] * p["body_line_height"] + p["list_item_margin_bottom"]
                if proj.outcome and proj.outcome != proj.description:
                    o_lines = get_lines(proj.outcome, p["list_item_font_size"], job_bullet_width, is_bold=False)
                    bullets_h += o_lines * p["list_item_font_size"] * p["body_line_height"] + p["list_item_margin_bottom"]
                    
                proj_h += bullets_h
                proj_h += p["job_margin_bottom"]
                sec_h += proj_h
            sec_h += p["section_margin_bottom"]
            total_height += sec_h
            
        elif sec == "skills" and e.enhanced_skills:
            sec_h = title_h
            seen = set()
            skills = []
            for s in e.enhanced_skills:
                key = s.lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    skills.append(s)
                    
            LANGUAGES_KW = {"python", "r", "java", "javascript", "typescript", "c++", "c", "c#", "go", "rust", "kotlin", "swift", "scala", "bash", "matlab", "ruby", "php", "perl", "dart", "html", "css", "solidity", "haskell", "assembly"}
            FRAMEWORKS_KW = {"react", "next.js", "nextjs", "angular", "vue", "node.js", "nodejs", "express", "flask", "django", "fastapi", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "opencv", "nltk", "spacy", "huggingface", "transformers", "langchain", "spring boot", "spring", "dotnet", ".net", "laravel", "ruby on rails", "rails", "flutter", "react native", "svelte", "jquery", "bootstrap", "tailwind", "tailwindcss", "graphql", "redux", "jest", "pytest", "junit"}
            DATABASES_KW = {"sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra", "dynamodb", "oracle", "ms sql", "sql server", "mariadb", "neo4j", "couchdb", "elasticsearch", "firebase", "bigquery", "snowflake", "redshift", "hive", "impala", "db2"}
            CLOUD_KW = {"aws", "gcp", "azure", "google cloud", "amazon web services", "microsoft azure", "heroku", "digitalocean", "cloudflare", "lambda", "ec2", "s3", "rds", "route 53", "iam", "ecs", "eks"}
            
            languages = []
            frameworks = []
            databases = []
            clouds = []
            tools = []
            for s in skills:
                s_low = s.lower().strip()
                if s_low in LANGUAGES_KW: languages.append(s)
                elif s_low in FRAMEWORKS_KW: frameworks.append(s)
                elif s_low in DATABASES_KW: databases.append(s)
                elif s_low in CLOUD_KW: clouds.append(s)
                else: tools.append(s)
                
            rows = []
            if languages: rows.append(("Programming Languages", ", ".join(languages)))
            if frameworks: rows.append(("Frameworks & Libraries", ", ".join(frameworks)))
            if databases: rows.append(("Databases", ", ".join(databases)))
            if tools: rows.append(("Tools & Technologies", ", ".join(tools)))
            if clouds: rows.append(("Cloud Platforms", ", ".join(clouds)))
            
            # ul margin-top is 1.5pt in CSS
            bullets_h = 1.5
            if not rows:
                row_text = ", ".join(skills)
                r_lines = get_lines(row_text, p["list_item_font_size"], skills_content_width, is_bold=False)
                bullets_h += r_lines * p["list_item_font_size"] * p["body_line_height"]
            else:
                for label, vals in rows:
                    row_text = f"{label}: {vals}"
                    r_lines = get_lines(row_text, p["list_item_font_size"], skills_content_width, is_bold=False)
                    bullets_h += r_lines * p["list_item_font_size"] * p["body_line_height"] + p["list_item_margin_bottom"]
            sec_h += bullets_h
            sec_h += p["section_margin_bottom"]
            total_height += sec_h
            
        elif sec == "education" and e.education:
            sec_h = title_h
            for edu in e.education:
                deg = (edu.degree or "").strip()
                inst = (edu.institution or "").strip()
                if not deg and not inst:
                    continue
                yr = (edu.year or "").strip()
                gpa_str = f" — {edu.gpa}" if edu.gpa else ""
                content = f"{deg}{gpa_str} — {inst}"
                edu_width = content_width - 75.0 - 8.0 - (2.0 * MM_TO_PT)
                lines = get_lines(content, p["edu_row_font_size"], edu_width, is_bold=True)
                yr_lines = get_lines(yr, p["edu_row_font_size"], 75.0, is_bold=True)
                max_lines = max(lines, yr_lines, 1)
                sec_h += max_lines * p["edu_row_font_size"] * p["body_line_height"] + p["edu_row_margin_bottom"]
            sec_h += p["section_margin_bottom"]
            total_height += sec_h
            
        elif sec == "certifications" and e.certifications:
            sec_h = title_h
            bullets_h = 1.5
            for cert in e.certifications:
                if cert and len(cert) > 3:
                    c_lines = get_lines(cert, p["list_item_font_size"], general_bullet_width, is_bold=False)
                    bullets_h += c_lines * p["list_item_font_size"] * p["body_line_height"] + p["list_item_margin_bottom"]
            sec_h += bullets_h
            sec_h += p["section_margin_bottom"]
            total_height += sec_h
            
        elif sec == "accomplishments" and e.accomplishments:
            sec_h = title_h
            bullets_h = 1.5
            for acc in e.accomplishments:
                if acc and len(acc) > 5:
                    a_lines = get_lines(acc, p["list_item_font_size"], general_bullet_width, is_bold=False)
                    bullets_h += a_lines * p["list_item_font_size"] * p["body_line_height"] + p["list_item_margin_bottom"]
            sec_h += bullets_h
            sec_h += p["section_margin_bottom"]
            total_height += sec_h
            
        elif sec == "publications" and e.publications:
            sec_h = title_h
            bullets_h = 1.5
            for pub in e.publications:
                if pub and len(pub) > 5:
                    p_lines = get_lines(pub, p["list_item_font_size"], general_bullet_width, is_bold=False)
                    bullets_h += p_lines * p["list_item_font_size"] * p["body_line_height"] + p["list_item_margin_bottom"]
            sec_h += bullets_h
            sec_h += p["section_margin_bottom"]
            total_height += sec_h


    # Add 35pt safety cushion to prevent page overflows due to minor rendering variations
    return total_height + 35.0


def compress_resume_internal(enhanced: EnhancedResume, jd: ParsedJD, is_two_page: bool) -> EnhancedResume:
    compressed = deepcopy(enhanced)
    jd_keywords = jd.all_keywords if jd else []
    target_height = 1684.0 if is_two_page else 842.0
    setattr(compressed, "is_two_page", is_two_page)
    
    def finalize(e: EnhancedResume) -> EnhancedResume:
        # Determine the layout names in order
        layout_order = ["expanded", "loose", "normal", "dense", "super_dense"]
        target_pages = 2 if is_two_page else 1
        
        # Calculate heights for all layouts
        heights = {name: estimate_resume_height(e, name) for name in layout_order}
        
        # Find the best predefined layout B that fits
        layout_B = None
        idx_B = -1
        for idx, name in enumerate(layout_order):
            if heights[name] <= target_height:
                layout_B = name
                idx_B = idx
                break
                
        if not layout_B:
            # If nothing fits, use super_dense
            layout_B = "super_dense"
            idx_B = len(layout_order) - 1
            
        # If layout_B is expanded (the largest), no need to interpolate
        if idx_B == 0:
            layout_style = "expanded"
            layout_params = LAYOUTS["expanded"]
            estimated_h = heights["expanded"]
        else:
            # We can interpolate between layout_A = layout_order[idx_B - 1] and layout_B
            layout_A = layout_order[idx_B - 1]
            h_A = heights[layout_A]
            h_B = heights[layout_B]
            
            p_A = LAYOUTS[layout_A]
            p_B = LAYOUTS[layout_B]
            
            # Calculate interpolation factor t (weight for layout_A, which is the larger one)
            # h_interp = t * h_A + (1 - t) * h_B = target_height
            # t = (target_height - h_B) / (h_A - h_B)
            if h_A > h_B:
                t = (target_height - h_B) / (h_A - h_B)
            else:
                t = 0.0
            t = max(0.0, min(1.0, t))
            
            # Interpolate and verify it fits (in case line wrap step-changes cause minor differences)
            interpolated_params = {}
            while t >= 0.0:
                interpolated_params = {}
                for k, val_B in p_B.items():
                    val_A = p_A[k]
                    # Linearly interpolate numeric keys
                    interpolated_params[k] = round(t * val_A + (1 - t) * val_B, 3)
                
                # Verify height
                h_est = estimate_resume_height(e, interpolated_params)
                if h_est <= target_height:
                    # Verify using the actual browser PDF rendering
                    e.layout_params = interpolated_params
                    if check_pdf_page_count(e) <= target_pages:
                        break
                t -= 0.05
                
            if t < 0.0:
                layout_params = p_B
                layout_style = layout_B
                estimated_h = h_B
            else:
                layout_params = interpolated_params
                layout_style = f"dynamic_{layout_A}_{layout_B}_{t:.2f}"
                estimated_h = estimate_resume_height(e, layout_params)
                
        setattr(e, "layout_style", layout_B)  # keep base style name for metadata
        setattr(e, "layout_params", layout_params)
        setattr(e, "use_dense_spacing", layout_B == "dense" or layout_B == "super_dense")
        logger.info(f"Compression Agent: Final layout = {layout_style} (estimated height = {estimated_h:.1f} pt)")
        return e

    def check_fit(e: EnhancedResume) -> bool:
        est_height = estimate_resume_height(e, "super_dense")
        if est_height > 860.0:
            return False
        return check_pdf_page_count(e) <= (2 if is_two_page else 1)


    # If it fits without trimming:
    if check_fit(compressed):
        return finalize(compressed)

    # --- Step 1: Trim auxiliary sections ---
    limit_aux = 5 if is_two_page else 2
    if len(compressed.accomplishments) > limit_aux:
        compressed.accomplishments = compressed.accomplishments[:limit_aux]
        if check_fit(compressed): return finalize(compressed)
        
    if len(compressed.certifications) > limit_aux:
        compressed.certifications = compressed.certifications[:limit_aux]
        if check_fit(compressed): return finalize(compressed)
        
    if len(compressed.publications) > limit_aux:
        compressed.publications = compressed.publications[:limit_aux]
        if check_fit(compressed): return finalize(compressed)

    # --- Step 2: Compaction of Skills ---
    limit_skills = 40 if is_two_page else 24
    if len(compressed.enhanced_skills) > limit_skills:
        core_skills_matches = [s for s in compressed.enhanced_skills if s.lower() in {k.lower() for k in jd_keywords}]
        other_skills = [s for s in compressed.enhanced_skills if s.lower() not in {k.lower() for k in jd_keywords}]
        compressed.enhanced_skills = (core_skills_matches + other_skills)[:limit_skills]
        if check_fit(compressed): return finalize(compressed)

    # --- Step 3: Trim project details ---
    limit_proj_words = 25 if is_two_page else 18
    for proj in compressed.enhanced_projects:
        if proj.description and len(proj.description.split()) > limit_proj_words:
            sentences = proj.description.split(". ")
            if sentences:
                proj.description = sentences[0].strip()
                if not proj.description.endswith("."):
                    proj.description += "."
        if proj.outcome and len(proj.outcome.split()) > limit_proj_words:
            sentences = proj.outcome.split(". ")
            if sentences:
                proj.outcome = sentences[0].strip()
                if not proj.outcome.endswith("."):
                    proj.outcome += "."
    if check_fit(compressed): return finalize(compressed)

    # --- Step 4: Experience Bullets ---
    if is_two_page:
        for exp in compressed.enhanced_experience:
            if len(exp.bullets) > 4:
                bullet_scores = [(j, rank_item_relevance(b, jd_keywords)) for j, b in enumerate(exp.bullets)]
                bullet_scores.sort(key=lambda x: x[1], reverse=True)
                top_b_indices = {idx for idx, _ in bullet_scores[:4]}
                exp.bullets = [b for idx, b in enumerate(exp.bullets) if idx in top_b_indices]
    else:
        if len(compressed.enhanced_experience) >= 3:
            for idx, exp in enumerate(compressed.enhanced_experience):
                limit_bullets = 2 if idx < 2 else 1
                if len(exp.bullets) > limit_bullets:
                    bullet_scores = [(j, rank_item_relevance(b, jd_keywords)) for j, b in enumerate(exp.bullets)]
                    bullet_scores.sort(key=lambda x: x[1], reverse=True)
                    top_b_indices = {i for i, _ in bullet_scores[:limit_bullets]}
                    exp.bullets = [b for i, b in enumerate(exp.bullets) if i in top_b_indices]
        else:
            for exp in compressed.enhanced_experience:
                if len(exp.bullets) > 3:
                    bullet_scores = [(j, rank_item_relevance(b, jd_keywords)) for j, b in enumerate(exp.bullets)]
                    bullet_scores.sort(key=lambda x: x[1], reverse=True)
                    top_b_indices = {idx for idx, _ in bullet_scores[:3]}
                    exp.bullets = [b for idx, b in enumerate(exp.bullets) if idx in top_b_indices]
    if check_fit(compressed): return finalize(compressed)

    # --- Step 5: Keep top projects ---
    limit_projects = 4 if is_two_page else 2
    if len(compressed.enhanced_projects) > limit_projects:
        proj_scores = []
        for i, proj in enumerate(compressed.enhanced_projects):
            proj_text = f"{proj.name} {proj.description} {proj.outcome} {' '.join(proj.tech_used)}"
            score = rank_item_relevance(proj_text, jd_keywords)
            proj_scores.append((i, score))
        proj_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = {idx for idx, _ in proj_scores[:limit_projects]}
        compressed.enhanced_projects = [
            p for i, p in enumerate(compressed.enhanced_projects) if i in top_indices
        ]
    if check_fit(compressed): return finalize(compressed)

    # --- Step 6: Prune summary as a final measure ---
    if compressed.enhanced_summary and len(compressed.enhanced_summary.split()) > 50:
        sentences = re.split(r'(?<=[.!?])\s+', compressed.enhanced_summary)
        summary_words = []
        truncated_sentences = []
        for s in sentences:
            words = s.split()
            if len(summary_words) + len(words) <= 45 or not truncated_sentences:
                summary_words.extend(words)
                truncated_sentences.append(s)
            else:
                break
        compressed.enhanced_summary = " ".join(truncated_sentences)
    if check_fit(compressed): return finalize(compressed)

    # --- Step 7: Clear achievements, certifications, publications if still not fitting ---
    if not check_fit(compressed):
        compressed.accomplishments = []
        if check_fit(compressed): return finalize(compressed)
    if not check_fit(compressed):
        compressed.certifications = []
        if check_fit(compressed): return finalize(compressed)
    if not check_fit(compressed):
        compressed.publications = []
        if check_fit(compressed): return finalize(compressed)

    # --- Step 8: Reduce projects to 1 ---
    if not check_fit(compressed) and len(compressed.enhanced_projects) > 1:
        compressed.enhanced_projects = compressed.enhanced_projects[:1]
        if check_fit(compressed): return finalize(compressed)

    # --- Step 9: Reduce experience bullets to 1 for all roles ---
    if not check_fit(compressed):
        for exp in compressed.enhanced_experience:
            if len(exp.bullets) > 1:
                exp.bullets = exp.bullets[:1]
        if check_fit(compressed): return finalize(compressed)

    # --- Step 10: Keep only top 2 experience entries ---
    if not check_fit(compressed) and len(compressed.enhanced_experience) > 2:
        compressed.enhanced_experience = compressed.enhanced_experience[:2]
        if check_fit(compressed): return finalize(compressed)
    
    return finalize(compressed)

def compress_resume(enhanced: EnhancedResume, jd: ParsedJD, max_words: int = 540, force_one_page: bool = True) -> EnhancedResume:
    """
    Intelligently fits the resume content to target single-page or two-page limits.
    Calculates estimated heights and dynamically selects font size and padding constraints.
    Trims content step-by-step only if needed.
    """
    # 1. Determine if 2-page CV is allowed initially
    num_projects = len(enhanced.enhanced_projects)
    num_experience = len(enhanced.enhanced_experience)
    is_two_page_allowed = (num_projects > 3 and num_experience > 2) if not force_one_page else False
    
    # 2. Check if the full untrimmed content can fit on 1 page (842 pt) under super_dense layout
    if force_one_page:
        logger.info("Compression Agent: Strictly forcing 1 page target.")
        is_two_page = False
    elif estimate_resume_height(enhanced, "super_dense") <= 842.0:
        logger.info("Compression Agent: Full content fits on 1 page under super_dense layout. Targeting 1 page.")
        is_two_page = False
    else:
        is_two_page = is_two_page_allowed
        
    # 3. Run the compression logic
    compressed = compress_resume_internal(enhanced, jd, is_two_page=is_two_page)
    
    # 4. If we targeted 1 page but it still doesn't fit even under super_dense layout after max trimming,
    # fallback to 2 pages (unless force_one_page is active)
    if not is_two_page and check_pdf_page_count(compressed) > 1:
        if force_one_page:
            logger.warning("Compression Agent: Could not fit content on 1 page under super_dense, but force_one_page is active. Staying on 1 page (will use super_dense and might clip).")
        else:
            logger.warning("Compression Agent: Could not fit content on 1 page even with maximum trimming. Falling back to 2 pages.")
            compressed = compress_resume_internal(enhanced, jd, is_two_page=True)
        
    return compressed

