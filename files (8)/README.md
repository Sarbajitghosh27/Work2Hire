# Rozgar24x7 — AI Resume Builder

ATS-optimised resume builder powered by **local LLMs only**.
Zero external API. Zero tokenization cost. Runs 100% on your machine.

## Three LLM Backends

| Backend | Model | RAM Required | Setup |
|---------|-------|--------------|-------|
| **Ollama** | mistral:7b / llama3.1:8b / phi3:mini | 4–8 GB | easiest |
| **HuggingFace** | any HF Instruct model (4-bit) | 4–6 GB GPU | medium |
| **Fine-tuned** | your own LoRA/QLoRA or merged model | 4–6 GB GPU | advanced |

Switch by setting `LLM_BACKEND` in `config.py`.

---

## Project Structure

```
rozgar24x7/
├── app.py                    ← Streamlit UI (two-panel, live preview + edit mode)
├── config.py                 ← All LLM backend settings, scoring weights, taxonomy
├── requirements.txt
└── core/
    ├── __init__.py
    ├── llm_backend.py        ← NEW: unified LLM caller (Ollama / HF / FineTuned)
    ├── resume_parser.py      ← PDF/DOCX parser
    ├── jd_engine.py          ← JD intelligence engine
    ├── scoring_engine.py     ← ATS scoring (BM25 + SBERT, 5 sub-scores)
    ├── enhancement_engine.py ← 8-agent pipeline (now uses llm_backend.py)
    └── resume_generator.py   ← HTML/PDF resume generator
```

---

## Setup

### Step 1 — Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

### Backend 1 — Ollama (Recommended)

```bash
# 1. Install Ollama
#    https://ollama.com/download  (Linux / macOS / Windows)

# 2. Start the server (keep this terminal open)
ollama serve

# 3. Pull a model
ollama pull mistral:7b          # default — good balance (~4 GB)
# ollama pull llama3.1:8b       # best quality (~5 GB)
# ollama pull phi3:mini         # fastest, lowest RAM (~2 GB)
# ollama pull gemma2:9b         # Google, very capable (~5 GB)
# ollama pull codellama:7b      # coding-focused

# 4. config.py settings
#    LLM_BACKEND = "ollama"
#    OLLAMA_MODEL = "mistral:7b"
```

---

### Backend 2 — HuggingFace Transformers (local, no API)

```bash
# 1. Install extra dependencies (already in requirements.txt)
pip install transformers torch accelerate bitsandbytes

# 2. config.py settings
#    LLM_BACKEND  = "huggingface"
#    HF_MODEL_PATH = "microsoft/phi-3-mini-4k-instruct"  # ~2 GB, fast
#    # or: "mistralai/Mistral-7B-Instruct-v0.2"          # ~4 GB
#    # or: "meta-llama/Llama-3.1-8B-Instruct"            # needs HF_TOKEN
#    # or: "google/gemma-2-9b-it"                         # needs HF_TOKEN
#    # or: "/absolute/path/to/local/model/folder"         # fully offline
#    HF_LOAD_IN_4BIT = True   # saves GPU memory, slightly lower quality
#    HF_DEVICE = "auto"       # auto-detects GPU; set "cpu" to force CPU

# 3. For gated models (llama-3, gemma) — set your HF token
#    HF_TOKEN = "hf_your_token_here"
#    # or: export HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

---

### Backend 3 — Your Fine-tuned Model (LoRA / QLoRA / Merged)

```bash
# Option A: LoRA / QLoRA adapter
#    LLM_BACKEND           = "finetuned"
#    FINETUNED_BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
#    FINETUNED_ADAPTER_PATH = "/home/you/my_lora_adapter"
#    FINETUNED_MERGED_PATH  = ""

# Option B: Fully merged fine-tuned model
#    LLM_BACKEND           = "finetuned"
#    FINETUNED_BASE_MODEL   = ""
#    FINETUNED_ADAPTER_PATH = ""
#    FINETUNED_MERGED_PATH  = "/home/you/my_merged_resume_model"

# Install peft for adapter loading
pip install peft
```

Training tip — to fine-tune your own resume rewriter model, create a dataset of
(weak bullet / weak summary → strong ATS-optimised version) pairs and fine-tune
using QLoRA with `transformers` + `trl` (SFTTrainer). The `SYSTEM_PROMPT` in
`core/llm_backend.py` defines the persona — include it in your training data.

---

## Run the App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Multi-Agent Pipeline

| Agent | Role | Uses LLM? |
|-------|------|-----------|
| Agent 1 — Intake | Structured form data | No |
| Agent 2 — Validator | Merge form + parsed resume | No |
| Agent 3 — Gap Analyst | Identify missing JD keywords | No |
| Agent 4a — Summariser | Rewrite professional summary | **Yes** |
| Agent 4b — Bullet Rewriter | Rewrite weak bullets | **Yes** |
| Agent 4c — Project Enhancer | Expand project descriptions to match JD | **Yes** |
| Agent 5 — Skill Booster | Infer + inject JD keywords → ATS 90+ | **Yes** |
| Agent 6 — QA Guard | Strip contact bleed, validate output | No |

LLM calls go through `core/llm_backend.py` → active backend.
Contact info is **never** passed to the LLM.

---

## ATS Scoring System

Five weighted sub-scores (edit weights in `config.py`):

| Score | Weight | What it measures |
|-------|--------|-----------------|
| ATS Keyword | 25% | BM25 keyword overlap with JD |
| Semantic Match | 30% | Sentence-BERT cosine similarity |
| Technical Depth | 20% | JD tech stack coverage + project density |
| Readability | 15% | Action verbs, section completeness, word count |
| Achievement Impact | 10% | Quantified metrics in bullets |

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (local, ~80 MB).

---

## Optional: PDF Export

```bash
# Ubuntu/Debian
sudo apt-get install wkhtmltopdf
# macOS
brew install wkhtmltopdf
# Windows: https://wkhtmltopdf.org/downloads.html
```

---

## Tips for ATS 90+

1. Paste the full JD — more text = better keyword extraction
2. Every bullet must start with a past-tense action verb
3. At least 40% of bullets should have a metric (%, ×, K users, ms)
4. Include all JD tech even if basic — Agent 5 handles inference
5. Keep resume to 1 page (550–800 words)
6. Use the Edit mode (✏️ button) to fine-tune any section after generation
