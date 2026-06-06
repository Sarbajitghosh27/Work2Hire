"""
config.py — Central configuration for the Rozgar24x7 AI/ML pipeline.
Supports three LLM backends — zero external API calls, zero tokenization cost.

  BACKEND 1: Ollama        — runs any GGUF/quantised model locally via Ollama server
  BACKEND 2: HuggingFace   — loads a model directly from HuggingFace Hub or local path
  BACKEND 3: FineTuned     — loads your own fine-tuned model (LoRA / QLoRA / full)

Set LLM_BACKEND below then configure the matching section.
"""

from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# CHOOSE YOUR BACKEND
# ──────────────────────────────────────────────────────────────────────────────
# Options: "ollama"  |  "huggingface"  |  "finetuned"
LLM_BACKEND = "ollama"

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND 1: Ollama settings
# ──────────────────────────────────────────────────────────────────────────────
OLLAMA_MODEL    = "qwen2.5:1.5b"          # swap to llama3.1:8b / phi3:mini / gemma2:9b
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT  = 120

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND 2: HuggingFace Transformers (runs fully locally, no API)
# ──────────────────────────────────────────────────────────────────────────────
# Set HF_MODEL_PATH to a HuggingFace Hub model ID or absolute local folder path.
# Examples:
#   "mistralai/Mistral-7B-Instruct-v0.2"
#   "meta-llama/Llama-3.1-8B-Instruct"    (needs HF_TOKEN env var)
#   "microsoft/phi-3-mini-4k-instruct"
#   "google/gemma-2-9b-it"
#   "/home/user/models/my_local_model"
HF_MODEL_PATH   = "microsoft/phi-3-mini-4k-instruct"
HF_TOKEN        = ""          # HuggingFace token if needed (llama-3 etc.)
HF_DEVICE       = "auto"      # "auto" | "cuda" | "cpu"   — auto picks GPU if available
HF_LOAD_IN_4BIT = True        # 4-bit quantisation via bitsandbytes (saves VRAM)
HF_LOAD_IN_8BIT = False       # 8-bit quantisation (slightly more VRAM, more accurate)
HF_MAX_NEW_TOKENS = 512
HF_TEMPERATURE  = 0.3
HF_DO_SAMPLE    = True

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND 3: Fine-tuned model (LoRA / QLoRA adapter or fully merged model)
# ──────────────────────────────────────────────────────────────────────────────
# FINETUNED_BASE_MODEL: the original base model the adapter was trained on.
# FINETUNED_ADAPTER_PATH: path to your LoRA/QLoRA adapter folder (or "" if merged).
# FINETUNED_MERGED_PATH: if you merged adapter into base, point here directly.
#
# To use a LoRA adapter:
#   FINETUNED_BASE_MODEL   = "mistralai/Mistral-7B-v0.1"
#   FINETUNED_ADAPTER_PATH = "/home/user/lora_adapter_ats_resume_v1"
#   FINETUNED_MERGED_PATH  = ""
#
# To use a fully merged fine-tuned model:
#   FINETUNED_BASE_MODEL   = ""
#   FINETUNED_ADAPTER_PATH = ""
#   FINETUNED_MERGED_PATH  = "/home/user/finetuned_resume_model"
FINETUNED_BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_ADAPTER_PATH = ""          # path to LoRA adapter (leave "" if merged)
FINETUNED_MERGED_PATH  = ""          # path to merged model (leave "" if using adapter)
FINETUNED_DEVICE       = "auto"
FINETUNED_LOAD_IN_4BIT = True
FINETUNED_MAX_NEW_TOKENS = 512
FINETUNED_TEMPERATURE  = 0.3

# ──────────────────────────────────────────────────────────────────────────────
# Shared LLM generation settings
# ──────────────────────────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS  = 512

# ──────────────────────────────────────────────────────────────────────────────
# Embedding model — runs locally via HuggingFace, zero cost
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ──────────────────────────────────────────────────────────────────────────────
# Scoring weights  (must sum to 1.0)
# ──────────────────────────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "ats_keyword"     : 0.25,
    "semantic_match"  : 0.30,
    "technical_depth" : 0.20,
    "readability"     : 0.15,
    "achievement"     : 0.10,
}

# ──────────────────────────────────────────────────────────────────────────────
# Role domains for personalisation
# ──────────────────────────────────────────────────────────────────────────────
ROLE_DOMAINS = [
    "Artificial Intelligence / ML",
    "Data Science / Analytics",
    "Software Engineering (Backend)",
    "Software Engineering (Frontend)",
    "Full Stack Development",
    "DevOps / Cloud Engineering",
    "Embedded Systems / IoT",
    "VLSI / Hardware Engineering",
    "Cybersecurity",
    "Product Management",
    "HR / Talent Acquisition",
    "Finance / Accounting",
    "Marketing / Growth",
    "Sales / Business Development",
    "Healthcare / Biotech",
    "Education / Academic",
    "Operations / Supply Chain",
]

# ──────────────────────────────────────────────────────────────────────────────
# Tech keyword taxonomy (ATS scoring)
# ──────────────────────────────────────────────────────────────────────────────
TECH_TAXONOMY: dict[str, List[str]] = {
    "languages"     : ["python","java","javascript","typescript","c++","c","go","rust","kotlin","swift","scala","r","matlab"],
    "ml_frameworks" : ["pytorch","tensorflow","keras","scikit-learn","xgboost","lightgbm","catboost","huggingface","transformers","langchain"],
    "data"          : ["pandas","numpy","sql","postgresql","mongodb","redis","elasticsearch","spark","airflow","dbt","kafka"],
    "cloud"         : ["aws","gcp","azure","docker","kubernetes","terraform","github actions","ci/cd","lambda","ec2","s3"],
    "web"           : ["fastapi","django","flask","react","next.js","node.js","graphql","rest api","microservices"],
    "tools"         : ["git","linux","bash","jupyter","mlflow","wandb","dvc","onnx","triton","vllm"],
    "nlp"           : ["spacy","nltk","bert","gpt","llm","rag","embeddings","faiss","vector database","fine-tuning"],
    "hr"            : ["recruiting", "talent acquisition", "onboarding", "ats", "compensation", "benefits", "employee relations", "performance management", "sourcing", "hris", "workday", "payroll", "human resources", "talent management", "workforce planning"],
    "finance"       : ["accounting", "financial analysis", "excel", "financial modeling", "valuation", "budgeting", "forecasting", "sap", "auditing", "taxation", "risk management", "cfa", "cpa", "bookkeeping", "general ledger", "balance sheet", "p&l", "corporate finance", "treasury"],
    "marketing"     : ["seo", "sem", "growth marketing", "google analytics", "digital marketing", "content creation", "social media", "email campaigns", "branding", "copywriting", "lead generation", "crm", "hubspot", "product marketing", "marketing strategy", "adwords", "ppc"],
    "sales"         : ["b2b", "salesforce", "lead generation", "crm", "cold calling", "negotiation", "account management", "sales pipeline", "quota", "deal closing", "relationship building", "business development", "key accounts", "sales operations"],
    "healthcare"    : ["clinical", "patient care", "ehr", "emr", "hipaa", "diagnostics", "pharmacology", "nursing", "medical terminology", "telehealth", "patient safety", "healthcare administration", "medical records"],
    "education"     : ["curriculum design", "pedagogy", "lesson planning", "classroom management", "e-learning", "lms", "grading", "student engagement", "instructional design", "educational technology", "pedagogical", "academic advising", "special education"],
    "operations"    : ["supply chain", "logistics", "procurement", "inventory management", "six sigma", "process optimization", "erp", "vendor management", "lean", "operations management", "quality assurance", "logistics planning", "warehouse management"],
}

IMPLICIT_SKILL_MAP: dict[str, List[str]] = {
    "mlops"       : ["docker","kubernetes","mlflow","ci/cd","monitoring","model serving"],
    "llm"         : ["transformers","fine-tuning","rag","vector database","prompt engineering"],
    "backend"     : ["rest api","sql","authentication","caching","microservices"],
    "data engineer": ["airflow","spark","kafka","dbt","sql","etl"],
    "devops"      : ["docker","kubernetes","terraform","ci/cd","linux","bash"],
    "embedded"    : ["c","c++","rtos","uart","spi","i2c","arm","pcb"],
    "hris"        : ["workday", "payroll", "human resources", "talent management"],
    "accounting"  : ["general ledger", "balance sheet", "p&l", "excel"],
    "seo"         : ["google analytics", "digital marketing", "lead generation", "ppc"],
    "b2b"         : ["salesforce", "crm", "negotiation", "business development"],
    "clinical"    : ["patient care", "ehr", "emr", "hipaa"],
    "e-learning"  : ["curriculum design", "lms", "instructional design"],
    "supply chain": ["logistics", "procurement", "inventory management", "erp"],
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA ANALYST SKILL TAXONOMY — used by ATS Architect pipeline
# ──────────────────────────────────────────────────────────────────────────────
DATA_ANALYST_SKILLS: dict = {
    "core": [
        "excel", "sql", "power bi", "tableau", "data visualization",
        "data cleaning", "reporting", "dashboarding", "statistics",
        "business intelligence", "kpi tracking", "pivot tables",
        "data analysis", "spreadsheet modeling",
    ],
    "intermediate": [
        "python", "pandas", "numpy", "matplotlib", "seaborn",
        "hypothesis testing", "a/b testing", "etl", "data wrangling",
        "google analytics", "looker", "data storytelling",
        "funnel analysis", "cohort analysis", "regression analysis",
    ],
    "advanced": [
        "machine learning", "predictive analytics", "scikit-learn",
        "forecasting", "time series analysis", "nlp",
        "bigquery", "snowflake", "spark", "airflow", "dbt",
        "r", "statsmodels", "feature engineering",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# NON-TECHNICAL BACKGROUND SIGNALS — keyword signals per domain
# ──────────────────────────────────────────────────────────────────────────────
NON_TECH_BACKGROUND_SIGNALS: dict = {
    "HR / Talent Acquisition": [
        "recruiting", "talent acquisition", "onboarding", "employee relations",
        "performance management", "hris", "payroll", "sourcing", "headhunting",
        "hr generalist", "hr manager", "people operations", "workforce planning",
        "human resources", "interviewing", "job posting",
    ],
    "Marketing / Growth": [
        "marketing", "digital marketing", "seo", "sem", "social media",
        "content marketing", "email campaigns", "brand", "copywriting",
        "google analytics", "hubspot", "campaign", "advertising", "ppc",
        "lead generation", "influencer", "market research",
    ],
    "Sales / Business Development": [
        "sales", "b2b", "lead generation", "salesforce", "cold calling",
        "negotiation", "account manager", "pipeline", "quota",
        "closing deals", "business development", "crm", "relationship building",
        "revenue", "client acquisition",
    ],
    "Finance / Accounting": [
        "accounting", "financial analysis", "budgeting", "forecasting",
        "sap", "auditing", "taxation", "gst", "bookkeeping",
        "general ledger", "balance sheet", "p&l", "corporate finance",
        "reconciliation", "accounts payable", "accounts receivable",
    ],
    "Operations / Supply Chain": [
        "operations", "supply chain", "logistics", "procurement",
        "inventory", "six sigma", "lean", "process optimization",
        "vendor management", "warehouse", "erp", "quality assurance",
    ],
    "Education / Academic": [
        "teaching", "curriculum", "pedagogy", "classroom", "student",
        "instructional design", "e-learning", "lms", "academic",
        "lesson planning", "assessment", "educational technology",
    ],
    "Healthcare / Biotech": [
        "patient care", "clinical", "nursing", "medical", "ehr", "emr",
        "hipaa", "diagnostics", "pharmacology", "hospital", "healthcare",
        "biotech", "laboratory",
    ],
    "Customer Success / Support": [
        "customer success", "customer support", "client management",
        "ticketing", "zendesk", "freshdesk", "onboarding clients",
        "churn reduction", "nps", "csat", "customer satisfaction",
    ],
    "Administration / Business Support": [
        "administrative", "office management", "coordination",
        "scheduling", "data entry", "ms office", "excel reports",
        "document management", "business support",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# TRANSFERABILITY KNOWLEDGE GRAPH
# Maps non-tech background → analytics skills + analytics language bridge
# ──────────────────────────────────────────────────────────────────────────────
TRANSFERABILITY_MAP: dict = {
    "HR / Talent Acquisition": {
        "analytics_bridge": [
            "Workforce Analytics", "Employee Attrition Analysis",
            "Hiring Funnel Analysis", "Recruitment Metrics",
            "HR Dashboarding", "Headcount Reporting",
        ],
        "recommended_tools": ["power bi", "excel", "sql", "tableau"],
        "bullet_rewrites": {
            "managed employee records": "Maintained and analyzed employee datasets to support workforce reporting and operational decision-making",
            "handled employee records": "Maintained and analyzed employee datasets to support workforce reporting and operational decision-making",
            "prepared monthly reports": "Developed monthly KPI reports and dashboards to track organizational performance trends and support data-driven decision making",
            "recruited candidates": "Analyzed candidate pipeline data to optimize hiring funnel conversion rates and reduce time-to-fill metrics",
            "tracked hiring metrics": "Built hiring funnel dashboards to monitor recruitment KPIs including time-to-hire, offer acceptance rate, and pipeline velocity",
            "conducted performance reviews": "Analyzed employee performance data to identify attrition risk patterns and support workforce planning decisions",
            "managed onboarding": "Designed onboarding data workflows tracking completion rates and new-hire performance metrics",
            "sourced candidates": "Applied data-driven sourcing strategies analyzing channel performance metrics across LinkedIn, job boards, and referral networks",
        },
        "transferable_skills": [
            "Data Collection & Organization", "KPI Reporting",
            "Stakeholder Communication", "Workforce Planning",
            "Process Documentation", "Excel Proficiency",
        ],
    },
    "Marketing / Growth": {
        "analytics_bridge": [
            "Campaign Analytics", "A/B Testing", "Customer Segmentation",
            "Conversion Rate Optimization", "Funnel Analysis",
            "Digital Marketing Analytics",
        ],
        "recommended_tools": ["google analytics", "excel", "tableau", "sql", "python"],
        "bullet_rewrites": {
            "managed social media": "Analyzed social media performance metrics including engagement rate, reach, and conversion to optimize content strategy",
            "ran email campaigns": "Designed and analyzed A/B testing frameworks for email campaigns achieving measurable improvements in open and click-through rates",
            "created content": "Developed data-driven content strategies using Google Analytics insights to improve organic traffic and audience engagement",
            "tracked campaign performance": "Built campaign performance dashboards tracking ROAS, CPC, CTR, and conversion metrics across digital channels",
            "managed seo": "Applied data analysis to SEO performance metrics including keyword rankings, organic traffic trends, and backlink authority",
            "generated leads": "Analyzed lead generation funnel data to identify drop-off points and optimize conversion rates across marketing channels",
            "managed advertising": "Monitored and optimized PPC advertising spend using data analysis to maximize ROI and minimize cost per acquisition",
        },
        "transferable_skills": [
            "Campaign Performance Analysis", "A/B Testing",
            "Data Visualization", "Google Analytics",
            "Customer Segmentation", "KPI Reporting",
        ],
    },
    "Sales / Business Development": {
        "analytics_bridge": [
            "Revenue Analytics", "Sales Pipeline Analysis",
            "KPI Dashboard Creation", "Sales Forecasting",
            "Customer Churn Analysis", "Territory Analysis",
        ],
        "recommended_tools": ["excel", "salesforce", "power bi", "sql", "tableau"],
        "bullet_rewrites": {
            "managed sales pipeline": "Analyzed sales pipeline data to forecast quarterly revenue, identify bottlenecks, and optimize deal progression rates",
            "exceeded sales targets": "Tracked and analyzed individual and team sales KPIs to identify performance trends and strategic growth opportunities",
            "generated revenue": "Built revenue reporting dashboards tracking deal velocity, win rates, and pipeline coverage to inform sales strategy",
            "managed client relationships": "Analyzed client engagement data and purchase patterns to identify upsell opportunities and reduce churn risk",
            "cold called prospects": "Analyzed outreach conversion data across channels to optimize sales development playbooks and improve pipeline quality",
            "prepared sales reports": "Developed monthly and quarterly sales analytics reports for senior leadership using Excel and CRM data",
            "managed accounts": "Monitored account health metrics and revenue trends to prioritize retention efforts and identify expansion opportunities",
        },
        "transferable_skills": [
            "Revenue Reporting", "Sales Dashboard Creation",
            "KPI Monitoring", "Forecasting", "CRM Data Analysis",
            "Business Intelligence",
        ],
    },
    "Finance / Accounting": {
        "analytics_bridge": [
            "Financial Analytics", "Excel Financial Modeling",
            "Budget Variance Analysis", "P&L Analysis",
            "Cost Optimization Reporting", "Financial Dashboard Creation",
        ],
        "recommended_tools": ["excel", "sql", "power bi", "python", "tableau"],
        "bullet_rewrites": {
            "prepared financial statements": "Analyzed financial statements and balance sheet data to identify trends, variances, and business performance insights",
            "managed budget": "Developed budget vs actual dashboards tracking variance analysis and forecasting future financial performance",
            "conducted audits": "Applied data analysis techniques during financial audits to identify anomalies, patterns, and compliance risks",
            "reconciled accounts": "Built automated reconciliation workflows in Excel to streamline account matching and reduce manual errors",
            "prepared tax returns": "Analyzed tax data and financial records to identify optimization opportunities and ensure regulatory compliance",
            "managed accounts payable": "Analyzed payable aging reports and cash flow data to optimize payment scheduling and vendor relationships",
            "forecasted revenue": "Built financial forecasting models using historical data trends to project revenue and cost scenarios",
        },
        "transferable_skills": [
            "Financial Data Analysis", "Excel Modeling",
            "Budget Variance Reporting", "Data-Driven Forecasting",
            "P&L Analysis", "Business Intelligence",
        ],
    },
    "Operations / Supply Chain": {
        "analytics_bridge": [
            "Process Analytics", "Supply Chain Data Analysis",
            "Root Cause Analysis", "Operational KPI Reporting",
            "Demand Forecasting", "Process Optimization Analytics",
        ],
        "recommended_tools": ["excel", "sql", "power bi", "tableau", "erp"],
        "bullet_rewrites": {
            "managed operations": "Analyzed operational data to identify process bottlenecks, reduce cycle times, and improve throughput efficiency",
            "optimized processes": "Applied data-driven process analysis (root cause analysis, pareto charts) to reduce operational waste and improve KPIs",
            "managed inventory": "Built inventory analytics dashboards tracking stock levels, turnover rates, and demand forecast accuracy",
            "managed supply chain": "Analyzed supply chain performance data to reduce lead times, optimize vendor selection, and minimize logistics costs",
            "managed vendors": "Evaluated vendor performance using data analysis to optimize procurement decisions and reduce supply chain risk",
            "prepared reports": "Developed operational KPI dashboards tracking throughput, defect rates, and on-time delivery metrics for leadership",
            "coordinated logistics": "Analyzed logistics performance data including delivery times, costs per shipment, and carrier performance metrics",
        },
        "transferable_skills": [
            "Process Analytics", "Data Cleaning & Reporting",
            "Root Cause Analysis", "KPI Monitoring",
            "Operational Dashboarding", "Business Intelligence",
        ],
    },
    "Education / Academic": {
        "analytics_bridge": [
            "Student Performance Analytics", "Learning Outcome Analysis",
            "Assessment Data Analysis", "Educational KPI Reporting",
            "Curriculum Effectiveness Analysis",
        ],
        "recommended_tools": ["excel", "google analytics", "power bi", "sql"],
        "bullet_rewrites": {
            "taught students": "Analyzed student assessment data to identify performance trends and optimize instructional strategies for improved outcomes",
            "designed curriculum": "Developed data-informed curriculum frameworks measuring learning outcomes and adjusting content based on performance metrics",
            "managed classroom": "Tracked and analyzed student engagement and performance metrics to personalize learning interventions",
            "conducted assessments": "Designed and analyzed student assessment frameworks using statistical methods to measure learning effectiveness",
            "mentored students": "Used performance data to identify at-risk students and implement targeted support interventions improving retention rates",
        },
        "transferable_skills": [
            "Data-Driven Decision Making", "Performance Reporting",
            "Stakeholder Communication", "Excel Proficiency",
            "Analytical Thinking", "Documentation",
        ],
    },
    "Healthcare / Biotech": {
        "analytics_bridge": [
            "Clinical Data Analysis", "Patient Outcome Analytics",
            "Healthcare KPI Reporting", "Diagnostic Data Analysis",
            "Population Health Analytics",
        ],
        "recommended_tools": ["excel", "sql", "tableau", "python", "power bi"],
        "bullet_rewrites": {
            "managed patient care": "Analyzed patient care workflow data to identify efficiency gaps and improve clinical process outcomes",
            "conducted clinical research": "Collected, cleaned, and analyzed clinical trial data to identify statistically significant treatment outcomes",
            "managed medical records": "Maintained and analyzed patient data records ensuring data integrity and compliance for health analytics reporting",
            "monitored patients": "Tracked and analyzed patient vitals and outcome data to support evidence-based care decisions",
        },
        "transferable_skills": [
            "Data Collection & Quality", "Clinical Data Analysis",
            "Reporting & Documentation", "Compliance Awareness",
            "Statistical Thinking", "Excel Proficiency",
        ],
    },
    "Customer Success / Support": {
        "analytics_bridge": [
            "Customer Analytics", "Churn Analysis",
            "NPS / CSAT Data Analysis", "Customer Journey Analysis",
            "Support Ticket Analytics",
        ],
        "recommended_tools": ["excel", "sql", "tableau", "power bi", "google analytics"],
        "bullet_rewrites": {
            "managed customer accounts": "Analyzed customer account health metrics and engagement data to identify churn risk and prioritize retention efforts",
            "resolved customer issues": "Analyzed support ticket data to identify recurring issue patterns and recommend systemic process improvements",
            "improved customer satisfaction": "Tracked NPS and CSAT metrics over time, analyzing feedback data to identify drivers of customer satisfaction",
            "onboarded customers": "Developed onboarding analytics dashboards tracking time-to-value and activation milestones for new customers",
        },
        "transferable_skills": [
            "Customer Data Analysis", "KPI Tracking",
            "Reporting & Dashboarding", "Stakeholder Communication",
            "Data-Driven Problem Solving", "Excel Proficiency",
        ],
    },
    "Administration / Business Support": {
        "analytics_bridge": [
            "Operational Data Reporting", "Business Performance Analytics",
            "Administrative KPI Tracking", "Data Entry & Cleaning",
            "Process Documentation & Analysis",
        ],
        "recommended_tools": ["excel", "power bi", "sql", "tableau"],
        "bullet_rewrites": {
            "managed data entry": "Maintained structured data entry workflows ensuring data accuracy and integrity for organizational reporting",
            "prepared reports": "Developed recurring operational reports and dashboards to track business performance metrics for management review",
            "managed office operations": "Analyzed operational workflows to identify inefficiencies and implement data-driven process improvements",
            "coordinated meetings": "Tracked and analyzed meeting outcomes and action item completion rates to improve organizational efficiency",
            "maintained records": "Organized and analyzed business records to surface operational trends and support data-driven administrative decisions",
        },
        "transferable_skills": [
            "Data Organization & Cleaning", "Reporting",
            "Microsoft Excel", "Process Documentation",
            "Attention to Detail", "Administrative Analytics",
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# RECOMMENDED CERTIFICATIONS — for non-tech → DA transition roadmap
# ──────────────────────────────────────────────────────────────────────────────
RECOMMENDED_CERTIFICATIONS: dict = {
    "priority_1": [
        "Google Data Analytics Professional Certificate (Coursera) — 6 months",
        "Microsoft Power BI Data Analyst Associate (PL-300) — 2 months",
        "IBM Data Analyst Professional Certificate (Coursera) — 3 months",
    ],
    "priority_2": [
        "Tableau Desktop Specialist Certification — 1 month",
        "SQL for Data Science (UC Davis — Coursera) — 4 weeks",
        "Excel Skills for Business Specialization (Macquarie — Coursera) — 2 months",
    ],
    "priority_3": [
        "Python for Everybody Specialization (Michigan — Coursera) — 3 months",
        "Statistics with Python Specialization (Michigan — Coursera) — 3 months",
        "Meta Marketing Analytics Professional Certificate (Coursera) — 5 months",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# PROJECT RECOMMENDATION MAP — Missing skill → recommended project
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_RECOMMENDATION_MAP: dict = {
    "sql": {
        "name": "SQL Sales / HR Analytics Dashboard",
        "description": "Build a relational database from a public dataset (e.g., AdventureWorks, HR data). Write 15+ SQL queries covering JOINs, GROUP BY, window functions, and CTEs.",
        "tools": ["MySQL / PostgreSQL", "SQL Server", "DBeaver"],
        "outcome": "Extract business insights: top revenue products, attrition by department, monthly trends",
        "dataset": "Kaggle HR Analytics Dataset / Northwind Database",
        "difficulty": "Beginner",
    },
    "power bi": {
        "name": "HR Analytics Power BI Dashboard",
        "description": "Connect Power BI to an HR dataset, build an interactive dashboard showing attrition rate, headcount by department, salary distribution, and performance scores.",
        "tools": ["Power BI Desktop", "Excel", "DAX"],
        "outcome": "Interactive dashboard with slicers, KPI cards, and drill-down capabilities",
        "dataset": "IBM HR Analytics Employee Attrition Dataset (Kaggle)",
        "difficulty": "Beginner",
    },
    "tableau": {
        "name": "Marketing Campaign Performance Dashboard",
        "description": "Visualize digital marketing campaign data in Tableau — track CTR, conversion rate, ROAS by channel, and geographic performance.",
        "tools": ["Tableau Public", "Excel / CSV"],
        "outcome": "Published Tableau Public dashboard with story points and executive summary",
        "dataset": "Digital Advertising Dataset (Kaggle) / Google Analytics export",
        "difficulty": "Beginner-Intermediate",
    },
    "python": {
        "name": "Employee Attrition Prediction (Python)",
        "description": "Use pandas and scikit-learn to analyze IBM HR Attrition dataset — EDA, feature engineering, logistic regression / random forest model.",
        "tools": ["Python", "Pandas", "Scikit-learn", "Matplotlib", "Jupyter"],
        "outcome": "Model achieving 85%+ accuracy with feature importance analysis and actionable HR recommendations",
        "dataset": "IBM HR Analytics Employee Attrition (Kaggle)",
        "difficulty": "Intermediate",
    },
    "statistics": {
        "name": "A/B Testing Analysis — Marketing Campaign",
        "description": "Analyze results of a simulated A/B test on email campaigns or landing pages — hypothesis testing, p-values, confidence intervals, effect size.",
        "tools": ["Python / Excel", "Scipy", "Statsmodels"],
        "outcome": "Statistical significance report with business recommendation and visualization",
        "dataset": "E-commerce A/B Test Dataset (Kaggle) / Simulated data",
        "difficulty": "Intermediate",
    },
    "excel": {
        "name": "Sales Performance Excel Dashboard",
        "description": "Build a dynamic sales analytics Excel dashboard with pivot tables, VLOOKUP/XLOOKUP, conditional formatting, and interactive charts.",
        "tools": ["Microsoft Excel", "Google Sheets"],
        "outcome": "One-page executive dashboard tracking revenue by region, product, and sales rep with trend analysis",
        "dataset": "Superstore Sales Dataset (Tableau Public) / Any company sales data",
        "difficulty": "Beginner",
    },
    "pandas": {
        "name": "Customer Churn Analysis (Pandas + EDA)",
        "description": "Perform exploratory data analysis on telecom churn dataset using pandas — data cleaning, missing value treatment, correlation analysis, visualization.",
        "tools": ["Python", "Pandas", "Matplotlib", "Seaborn", "Jupyter"],
        "outcome": "EDA report identifying top 5 churn drivers with visualizations and business recommendations",
        "dataset": "Telco Customer Churn Dataset (Kaggle)",
        "difficulty": "Intermediate",
    },
    "data visualization": {
        "name": "COVID-19 / Public Data Story Dashboard",
        "description": "Create a compelling data story visualizing a public dataset (COVID, elections, climate) across multiple chart types with narrative context.",
        "tools": ["Tableau Public / Power BI / Plotly"],
        "outcome": "Published interactive visualization story with 5+ chart types and key insights narrative",
        "dataset": "Our World in Data / WHO / World Bank Open Data",
        "difficulty": "Beginner",
    },
    "forecasting": {
        "name": "Sales Forecasting — Time Series Analysis",
        "description": "Build a time series forecasting model on retail sales data using ARIMA or Prophet — seasonality decomposition, trend analysis, forecast accuracy evaluation.",
        "tools": ["Python", "Prophet / Statsmodels", "Pandas", "Matplotlib"],
        "outcome": "12-month sales forecast with confidence intervals and MAPE < 15%",
        "dataset": "Rossmann Store Sales / M5 Forecasting Dataset (Kaggle)",
        "difficulty": "Intermediate-Advanced",
    },
    "etl": {
        "name": "ETL Pipeline — Data Warehouse Mini Project",
        "description": "Build a simple ETL pipeline extracting data from CSV/API, transforming with Python, and loading into a local SQLite/PostgreSQL database with scheduled runs.",
        "tools": ["Python", "Pandas", "SQLite / PostgreSQL", "Apache Airflow (optional)"],
        "outcome": "Automated ETL pipeline with data quality checks and summary analytics table",
        "dataset": "Any public API (OpenWeather, Alpha Vantage, REST Countries)",
        "difficulty": "Intermediate",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA ANALYST TARGET ROLES — for transition targeting
# ──────────────────────────────────────────────────────────────────────────────
DA_TARGET_ROLES = [
    "Data Analyst",
    "Business Analyst",
    "Product Analyst",
    "BI Analyst / Developer",
    "Data Scientist (Entry Level)",
    "Data Scientist",
    "Data Engineer",
    "Software Engineer (Backend)",
    "Software Engineer (Frontend)",
    "Full Stack Developer",
    "DevOps / Cloud Engineer",
    "Artificial Intelligence / ML Engineer",
    "Embedded Systems / IoT Engineer",
    "VLSI / Hardware Engineer",
    "Robotics Engineer",
    "Cybersecurity Analyst",
    "Product Manager",
    "HR Specialist / People Analyst",
    "Finance / Quantitative Analyst",
    "Operations / Supply Chain Analyst",
    "Marketing / Growth Analyst",
]

ATS_BAD_PATTERNS = [
    r"(header|footer)",
    r"(table|columns|text box)",
    r"(image|photo|picture)",
    r"\.(jpg|jpeg|png|gif)",
]

# ──────────────────────────────────────────────────────────────────────────────
# Domain-aware resume section ordering
# ──────────────────────────────────────────────────────────────────────────────
SECTION_ORDER_BY_DOMAIN: dict[str, list] = {
    "default": ["summary", "experience", "projects", "skills", "education", "certifications", "accomplishments", "publications"],
    "Artificial Intelligence / ML": ["summary", "skills", "experience", "projects", "education", "certifications", "accomplishments", "publications"],
    "Data Science / Analytics": ["summary", "skills", "experience", "projects", "education", "certifications", "accomplishments", "publications"],
    "Software Engineering (Backend)": ["summary", "experience", "projects", "skills", "education", "certifications", "accomplishments", "publications"],
    "Software Engineering (Frontend)": ["summary", "experience", "projects", "skills", "education", "certifications", "accomplishments", "publications"],
    "Full Stack Development": ["summary", "experience", "projects", "skills", "education", "certifications", "accomplishments", "publications"],
    "DevOps / Cloud Engineering": ["summary", "skills", "experience", "projects", "education", "certifications", "accomplishments", "publications"],
    "Embedded Systems / IoT": ["summary", "experience", "projects", "skills", "education", "certifications", "accomplishments", "publications"],
    "VLSI / Hardware Engineering": ["summary", "experience", "projects", "skills", "education", "certifications", "accomplishments", "publications"],
    "Cybersecurity": ["summary", "skills", "experience", "certifications", "projects", "education", "accomplishments", "publications"],
    "Product Management": ["summary", "experience", "skills", "projects", "education", "certifications", "accomplishments", "publications"],
    "HR / Talent Acquisition": ["summary", "skills", "experience", "education", "certifications", "accomplishments"],
    "Finance / Accounting": ["summary", "skills", "experience", "education", "certifications", "accomplishments"],
    "Marketing / Growth": ["summary", "skills", "experience", "projects", "education", "certifications", "accomplishments"],
    "Sales / Business Development": ["summary", "experience", "skills", "education", "certifications", "accomplishments"],
    "Healthcare / Biotech": ["summary", "skills", "experience", "education", "certifications", "accomplishments"],
    "Education / Academic": ["summary", "experience", "education", "skills", "certifications", "accomplishments"],
    "Operations / Supply Chain": ["summary", "experience", "skills", "education", "certifications", "accomplishments"],
}

SECTION_KEYWORDS = {
    "experience"    : ["experience","work history","employment","professional background"],
    "education"     : ["education","academic","qualification","degree","university"],
    "skills"        : ["skills","technologies","technical","competencies","expertise"],
    "projects"      : ["projects","portfolio","work samples","case studies"],
    "certifications": ["certification","certificate","credential","license","award"],
    "summary"       : ["summary","objective","profile","about","overview"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Realism, Logging, and Confidence Settings (Imported by core engines)
# ──────────────────────────────────────────────────────────────────────────────
REALISM_FILTER = True
REALISM_BANNED_PHRASES = [
    "streamline", "synergy", "paradigm shift", "utilize", "leverage",
    "innovative", "cutting-edge", "dynamic", "proven track record", "results-oriented",
    "spearheaded", "game-changer", "world-class", "out-of-the-box", "robustly",
    "transformative", "revolutionary", "go-to person", "thought leader", "detail-oriented"
]
CONFIDENCE_SCORING = True
LOGGING_ENABLED = True
LOG_DIR = "logs"

# ──────────────────────────────────────────────────────────────────────────────
# Optimization Loop Settings
# ──────────────────────────────────────────────────────────────────────────────
OPTIMIZATION_TARGET_SCORE = 0.90
OPTIMIZATION_MAX_ITERATIONS = 3
OPTIMIZATION_MIN_AUTHENTICITY = 75.0

