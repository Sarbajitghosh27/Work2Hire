"""
core/role_intelligence.py
────────────────────────────────────────────────────────
Role Intelligence & Domain Compatibility Engine.
Uses structured profiles across tech and non-tech domains
to identify profile mismatches and suggest crossover opportunities.
"""

import re
import logging
from typing import List, Dict, Set
from core.resume_parser import ParsedResume
from core.jd_engine import ParsedJD

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURED ROLE PROFILES
# ──────────────────────────────────────────────────────────────────────────────
ROLE_PROFILES: Dict[str, Dict] = {
    "Artificial Intelligence / ML": {
        "core_skills": {"pytorch", "tensorflow", "keras", "scikit-learn", "numpy", "pandas", "python", "r", "transformers", "huggingface", "llm", "rag", "fine-tuning", "vector database", "opencv", "spacy", "nltk", "mlflow", "triton", "onnx", "langchain"},
        "buzzwords": {"machine learning", "deep learning", "neural network", "natural language processing", "computer vision", "reinforcement learning", "supervised learning", "generative ai", "large language models", "prompt engineering", "feature engineering", "mlops"},
        "synonyms": {"ml engineer", "ai engineer", "machine learning engineer", "data scientist", "research scientist", "nlp engineer", "computer vision engineer", "deep learning engineer"}
    },
    "Data Science / Analytics": {
        "core_skills": {"python", "sql", "postgresql", "r", "pandas", "numpy", "tableau", "power bi", "excel", "sas", "matplotlib", "seaborn", "scikit-learn", "statsmodels", "spark", "hadoop", "airflow", "redshift", "bigquery", "snowflake"},
        "buzzwords": {"data analysis", "data visualization", "business intelligence", "statistical modeling", "predictive analytics", "regression", "hypothesis testing", "ab testing", "metrics", "dashboard", "data pipeline", "etl", "reporting"},
        "synonyms": {"data analyst", "business analyst", "data scientist", "bi analyst", "bi developer", "quantitative analyst", "reporting analyst"}
    },
    "Software Engineering (Backend)": {
        "core_skills": {"python", "java", "go", "rust", "c++", "c#", "node.js", "django", "fastapi", "flask", "spring boot", "ruby on rails", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "docker", "kubernetes", "rest api", "graphql", "grpc", "microservices"},
        "buzzwords": {"backend developer", "server-side", "system design", "database administration", "concurrency", "caching", "scalability", "distributed systems", "asynchronous", "message queue", "orm", "api design"},
        "synonyms": {"backend engineer", "software developer", "systems engineer", "backend developer", "server engineer"}
    },
    "Software Engineering (Frontend)": {
        "core_skills": {"javascript", "typescript", "html", "css", "react", "vue", "angular", "next.js", "nuxt", "svelte", "sass", "tailwind", "webpack", "vite", "npm", "yarn", "jest", "cypress", "redux", "graphql", "rest api", "figma"},
        "buzzwords": {"frontend developer", "user interface", "user experience", "responsive design", "single page application", "css-in-js", "web performance", "dom manipulation", "state management", "cross-browser"},
        "synonyms": {"frontend engineer", "frontend developer", "ui engineer", "web developer", "react developer"}
    },
    "Full Stack Development": {
        "core_skills": {"javascript", "typescript", "python", "html", "css", "react", "next.js", "node.js", "django", "fastapi", "postgresql", "mongodb", "sql", "rest api", "docker", "express", "git", "aws", "tailwind"},
        "buzzwords": {"full stack", "client-server", "mvc architecture", "frontend and backend", "web application", "monorepo", "cloud hosting", "deployment", "end-to-end development"},
        "synonyms": {"fullstack engineer", "full stack developer", "web engineer", "full-stack engineer"}
    },
    "DevOps / Cloud Engineering": {
        "core_skills": {"aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions", "gitlab ci", "bash", "python", "linux", "prometheus", "grafana", "elk stack", "nginx", "dns", "iam", "cloudformation"},
        "buzzwords": {"infrastructure as code", "ci/cd", "site reliability", "monitoring", "logging", "alerting", "auto-scaling", "load balancing", "cloud migration", "containerization", "kubernetes administration", "sre"},
        "synonyms": {"devops engineer", "cloud architect", "site reliability engineer", "sre", "platform engineer", "systems administrator"}
    },
    "Embedded Systems / IoT": {
        "core_skills": {"c", "c++", "python", "rtos", "free-rtos", "arm", "avr", "stm32", "arduino", "raspberry pi", "uart", "spi", "i2c", "can bus", "gpio", "ble", "wi-fi", "zigbee", "pcb design", "multimeter", "oscilloscope", "embedded linux"},
        "buzzwords": {"embedded hardware", "firmware developer", "microcontroller", "low-level programming", "hardware-software integration", "register-level", "device drivers", "bare-metal", "internet of things", "sensor integration"},
        "synonyms": {"embedded engineer", "firmware engineer", "embedded systems developer", "iot developer", "hardware engineer"}
    },
    "VLSI / Hardware Engineering": {
        "core_skills": {"verilog", "vhdl", "systemverilog", "fpga", "asic", "rtl", "synthesis", "timing analysis", "sta", "physical design", "cadence", "synopsys", "mentor graphics", "matlab", "ltspice", "circuit design", "schematic capture"},
        "buzzwords": {"hardware engineering", "chip design", "semiconductor", "microarchitecture", "logic synthesis", "rtl design", "functional verification", "timing closure", "printed circuit board", "analog circuit"},
        "synonyms": {"vlsi engineer", "hardware engineer", "asic verification engineer", "rtl design engineer", "fpga developer"}
    },
    "Cybersecurity": {
        "core_skills": {"wireshark", "nmap", "metasploit", "burp suite", "linux", "bash", "python", "snort", "splank", "siem", "firewalls", "active directory", "cryptography", "ssl/tls", "owasp", "nessus", "kali linux"},
        "buzzwords": {"penetration testing", "vulnerability assessment", "incident response", "security auditing", "soc analyst", "threat intelligence", "malware analysis", "identity access management", "network security", "compliance", "iso 27001", "gdpr", "hipaa"},
        "synonyms": {"cybersecurity analyst", "security engineer", "penetration tester", "ethical hacker", "soc analyst", "information security manager"}
    },
    "Product Management": {
        "core_skills": {"jira", "confluence", "slack", "trello", "amplitude", "mixpanel", "figma", "sql", "excel", "wireframing", "agile", "scrum", "product roadmap", "ab testing", "user research"},
        "buzzwords": {"product lifecycle", "go-to-market", "stakeholder management", "prioritization framework", "mvp", "user stories", "market research", "feature prioritization", "product strategy", "customer journey", "key results", "okrs"},
        "synonyms": {"product manager", "associate product manager", "technical product manager", "product owner", "product lead"}
    },
    "HR / Talent Acquisition": {
        "core_skills": {"hris", "workday", "bamboo-hr", "linkedin recruiter", "ats", "lever", "greenhouse", "payroll", "excel", "compensation", "benefits", "employee onboarding"},
        "buzzwords": {"talent acquisition", "sourcing", "screening", "employee relations", "performance evaluation", "talent pipeline", "workforce planning", "headhunting", "hr policy", "compliance", "interviewing"},
        "synonyms": {"hr generalist", "hr manager", "technical recruiter", "talent acquisition specialist", "hr specialist", "people partner"}
    },
    "Finance / Accounting": {
        "core_skills": {"sap", "oracle finance", "quickbooks", "excel", "tally", "cfa", "cpa", "audit", "general ledger", "balance sheet", "p&l", "budgeting", "taxation", "gst", "corporate finance"},
        "buzzwords": {"financial analysis", "forecasting", "variance analysis", "compliance", "bookkeeping", "reconciliation", "accounts payable", "accounts receivable", "cost optimization", "risk assessment", "financial modeling", "portfolio management"},
        "synonyms": {"financial analyst", "accountant", "finance manager", "controller", "bookkeeper", "tax consultant", "investment analyst"}
    },
    "Marketing / Growth": {
        "core_skills": {"seo", "sem", "google analytics", "hubspot", "mailchimp", "canva", "wordpress", "facebook ads", "google adwords", "sql", "tableau", "copywriting"},
        "buzzwords": {"growth hacking", "digital marketing", "social media strategy", "campaign optimization", "content marketing", "lead generation", "conversion rate", "cro", "ppc", "click-through rate", "branding", "audience segmentation"},
        "synonyms": {"marketing specialist", "growth manager", "digital marketer", "seo analyst", "content marketer", "brand manager", "social media manager"}
    },
    "Sales / Business Development": {
        "core_skills": {"crm", "salesforce", "hubspot", "linkedin sales navigator", "cold outreach", "email sequencing", "powerpoint", "negotiation", "contracting", "financial analysis"},
        "buzzwords": {"b2b sales", "lead generation", "sales pipeline", "quota achievement", "cold calling", "relationship building", "account management", "deal closing", "market expansion", "strategic partnership", "customer success"},
        "synonyms": {"sales executive", "business development associate", "account executive", "bde", "sales manager", "client relationship manager"}
    },
    "Healthcare / Biotech": {
        "core_skills": {"ehr", "emr", "epic systems", "clinical database", "medical terminology", "laboratory protocols", "pcr", "gel electrophoresis", "spectrophotometry", "pipetting"},
        "buzzwords": {"patient care", "clinical trials", "diagnostics", "pharmacology", "fda regulation", "hipaa compliance", "gmp", "quality control", "biotechnology", "patient safety", "laboratory safety", "clinical workflow"},
        "synonyms": {"clinical research associate", "biotech researcher", "lab technician", "healthcare administrator", "nurse", "medical assistant"}
    },
    "Education / Academic": {
        "core_skills": {"canvas", "blackboard", "moodle", "lms", "zoom", "google classroom", "lesson planning", "curriculum development", "grading rubrics", "academic advising"},
        "buzzwords": {"pedagogy", "instructional design", "student engagement", "special education", "e-learning", "classroom management", "educational technology", "blended learning", "formative assessment", "summative assessment", "academic research"},
        "synonyms": {"teacher", "instructor", "curriculum designer", "educational consultant", "academic coordinator", "lecturer", "instructional designer"}
    },
    "Operations / Supply Chain": {
        "core_skills": {"erp", "sap", "oracle supply chain", "excel", "sql", "tableau", "inventory tracking software", "procurement portals"},
        "buzzwords": {"supply chain", "logistics planning", "procurement", "inventory control", "vendor management", "six sigma", "process optimization", "lean operations", "warehouse management", "shipping", "distribution", "demand forecasting"},
        "synonyms": {"operations analyst", "supply chain specialist", "logistics coordinator", "procurement manager", "operations supervisor"}
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN COMPATIBILITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def calculate_compatibility(resume: ParsedResume, jd: ParsedJD) -> Dict:
    """
    Evaluates how compatible the candidate's background is with the target JD domain.
    Flags unrealistic transitions (e.g. non-tech to deep tech roles).
    
    Returns a dict containing:
      - compatibility_score (0-100)
      - is_mismatch (bool)
      - reasons (list of str strings explaining evaluation)
      - missing_foundations (list of crucial domain skills lacking)
      - crossover_skills (list of skills that bridge their background to the target)
    """
    target_domain = jd.domain
    profile = ROLE_PROFILES.get(target_domain)
    
    # Fallback to Software Engineering (Backend) if domain isn't in profiles
    if not profile:
        profile = ROLE_PROFILES["Software Engineering (Backend)"]
        target_domain = "Software Engineering (Backend)"

    reasons = []
    missing_foundations = []
    crossover_skills = []
    
    candidate_skills_lower = {s.lower() for s in resume.skills}
    candidate_text_lower = resume.raw_text.lower()
    
    # 1. Skill Overlap (40% weight)
    core_skills = profile["core_skills"]
    skill_overlap = core_skills & candidate_skills_lower
    skill_score = (len(skill_overlap) / max(len(core_skills) * 0.35, 1)) * 100  # relative to 35% target core coverage
    skill_score = min(skill_score, 100)
    
    # Find crucial missing core skills
    crucial_missing = sorted(list(core_skills - candidate_skills_lower))[:5]
    missing_foundations.extend(crucial_missing)
    
    # 2. Domain Keywords/Buzzwords Match (30% weight)
    buzzwords = profile["buzzwords"]
    matched_buzzwords = set()
    for word in buzzwords:
        if re.search(r"\b" + re.escape(word) + r"\b", candidate_text_lower):
            matched_buzzwords.add(word)
    buzz_score = (len(matched_buzzwords) / max(len(buzzwords) * 0.35, 1)) * 100
    buzz_score = min(buzz_score, 100)
    
    # 3. Title/Experience Domain Match (30% weight)
    # Check if candidate has had jobs with titles in the synonyms list
    synonyms = profile["synonyms"]
    matched_jobs_count = 0
    candidate_titles = []
    for exp in resume.experience:
        title = (exp.title or "").lower()
        if title:
            candidate_titles.append(title)
            if any(syn in title or title in syn for syn in synonyms):
                matched_jobs_count += 1
                
    title_score = 0.0
    if matched_jobs_count > 0:
        title_score = 100.0
    elif len(resume.experience) > 0:
        # Check for partial title alignment or related keywords in job descriptions
        job_body = " ".join(" ".join(exp.bullets) for exp in resume.experience).lower()
        partial_matches = sum(1 for syn in synonyms if syn in job_body)
        title_score = min((partial_matches / 2) * 100, 70.0) # max 70% if no exact title match but domain experience
    else:
        title_score = 0.0 # No work history
        
    # Calculate weighted compatibility score
    compatibility_score = (0.40 * skill_score) + (0.30 * buzz_score) + (0.30 * title_score)
    compatibility_score = round(compatibility_score, 1)
    
    # Crossover identification
    # Technical base transferability (e.g. math/SQL from data analyst to ML)
    all_known_tech = set()
    for prof in ROLE_PROFILES.values():
        all_known_tech.update(prof["core_skills"])
        
    # Common cross-domain bridges (e.g., Python, SQL, Excel, Git, Jira)
    universal_bridges = {"python", "sql", "excel", "git", "jira", "docker", "confluence", "scrum", "agile"}
    crossover_skills = list(candidate_skills_lower & core_skills | (candidate_skills_lower & universal_bridges & core_skills))
    
    # Mismatch Threshold Evaluation
    # If compatibility_score is extremely low (<30) and there is zero experience or core skill overlap
    is_mismatch = False
    
    # Identify candidate's primary domain to explain mismatch better
    candidate_domain_scores = {}
    for d, prof in ROLE_PROFILES.items():
        sk_over = len(prof["core_skills"] & candidate_skills_lower)
        candidate_domain_scores[d] = sk_over
        
    best_candidate_domain = max(candidate_domain_scores, key=candidate_domain_scores.get)
    if candidate_domain_scores[best_candidate_domain] == 0:
        best_candidate_domain = "Unknown/General"
        
    # Mismatch conditions
    if compatibility_score < 30.0:
        # High mismatch risk
        is_mismatch = True
        reasons.append(f"Target role is in '{target_domain}', but candidate's profile strongly aligns with '{best_candidate_domain}'.")
        reasons.append(f"No direct work experience matching the target synonyms ({', '.join(list(synonyms)[:3])}).")
        reasons.append("Very low core skill overlap (< 10%). Blind translation would yield high hallucination risk.")
    elif compatibility_score < 50.0:
        # Moderate warning
        reasons.append("Pivoting profile detected: Candidate possesses some transferable skills but lacks direct role experience.")
        if len(skill_overlap) < 2:
            reasons.append(f"Lacks key foundational technologies for '{target_domain}'. Need to add projects using: {', '.join(crucial_missing[:3])}.")
    else:
        reasons.append(f"Strong structural alignment with '{target_domain}' domain.")
        
    return {
        "compatibility_score": compatibility_score,
        "is_mismatch": is_mismatch,
        "reasons": reasons,
        "missing_foundations": missing_foundations,
        "crossover_skills": crossover_skills[:6],
        "candidate_primary_domain": best_candidate_domain
    }
