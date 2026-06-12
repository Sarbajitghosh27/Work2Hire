"""
core/llm_backend.py
────────────────────────────────────────────────────────────────────────────────
Unified LLM backend for Rozgar24x7.
Zero external API calls. Zero tokenization cost.

Three backends — configured in config.py:
  "ollama"      — Ollama local server (mistral, llama3, phi3, gemma2, etc.)
  "huggingface" — HuggingFace Transformers, runs model locally (4-bit / 8-bit)
  "finetuned"   — Your own LoRA/QLoRA adapter or fully merged fine-tuned model

Usage:
    from core.llm_backend import get_llm, call_llm

    llm = get_llm()                              # cached singleton
    response = call_llm("Rewrite this bullet...") # call active backend
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy singletons ──────────────────────────────────────────────────────────
_ollama_client   = None
_hf_pipeline     = None
_ft_pipeline     = None

SYSTEM_PROMPT = (
    "You are an expert ATS-optimised resume writer for the Indian job market. "
    "Write in third-person-implied past tense. Use strong action verbs. "
    "Be technically specific. Never invent facts, metrics, or contact details. "
    "Output ONLY what is requested — no preamble, no labels, no explanation."
)


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 1: Ollama
# ══════════════════════════════════════════════════════════════════════════════

def _get_ollama():
    global _ollama_client
    if _ollama_client is None:
        import ollama
        _ollama_client = ollama
    return _ollama_client


def _call_ollama(prompt: str, max_tokens: int) -> str:
    from config import OLLAMA_MODEL, LLM_TEMPERATURE
    client = _get_ollama()
    try:
        resp = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": LLM_TEMPERATURE, "num_predict": max_tokens},
        )
        # Handle both old dict API and new object API from ollama-python
        if isinstance(resp, dict):
            return resp["message"]["content"].strip()
        else:
            return resp.message.content.strip()
    except Exception as e:
        raise RuntimeError(
            f"Ollama error: {e}\n\n"
            "Make sure Ollama is running:\n"
            "  1. ollama serve\n"
            f"  2. ollama pull {OLLAMA_MODEL}"
        )


def check_ollama() -> dict:
    """Returns {'ok': bool, 'models': [...], 'error': str}"""
    try:
        from config import OLLAMA_MODEL
        client = _get_ollama()
        data = client.list()
        # Handle both old dict API (data["models"][i]["name"])
        # and new object API (data.models[i].model)
        raw_models = []
        if isinstance(data, dict):
            raw_models = data.get("models", [])
        else:
            raw_models = getattr(data, "models", [])

        available = []
        for m in raw_models:
            if isinstance(m, dict):
                available.append(m.get("name") or m.get("model", ""))
            else:
                available.append(getattr(m, "name", None) or getattr(m, "model", ""))

        model_ok = any(
            OLLAMA_MODEL == m or OLLAMA_MODEL.split(":")[0] in m
            for m in available
        )
        return {"ok": True, "models": available, "model_ready": model_ok, "error": ""}
    except Exception as e:
        return {"ok": False, "models": [], "model_ready": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 2: HuggingFace Transformers (local)
# ══════════════════════════════════════════════════════════════════════════════

def _get_hf_pipeline():
    global _hf_pipeline
    if _hf_pipeline is not None:
        return _hf_pipeline

    from config import (
        HF_MODEL_PATH, HF_TOKEN, HF_DEVICE,
        HF_LOAD_IN_4BIT, HF_LOAD_IN_8BIT,
        HF_MAX_NEW_TOKENS, HF_TEMPERATURE,
    )
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

        logger.info(f"Loading HuggingFace model: {HF_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_PATH,
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )

        quant_config = None
        if HF_LOAD_IN_4BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif HF_LOAD_IN_8BIT:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_PATH,
            token=HF_TOKEN or None,
            quantization_config=quant_config,
            device_map=HF_DEVICE,
            trust_remote_code=True,
        )
        model.eval()

        _hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            temperature=HF_TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.info(f"HuggingFace model loaded: {HF_MODEL_PATH}")
        return _hf_pipeline

    except ImportError as e:
        raise RuntimeError(
            f"Missing package: {e}\n"
            "Install with:\n"
            "  pip install transformers torch accelerate bitsandbytes"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace model '{HF_MODEL_PATH}': {e}")


def _call_hf(prompt: str, max_tokens: int) -> str:
    from config import HF_MAX_NEW_TOKENS
    pipe = _get_hf_pipeline()
    full_prompt = f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"
    outputs = pipe(
        full_prompt,
        max_new_tokens=min(max_tokens, HF_MAX_NEW_TOKENS),
        return_full_text=False,
    )
    text = outputs[0]["generated_text"].strip()
    # Strip any residual instruction tokens
    text = re.sub(r"\[/?INST\]|<</?SYS>>", "", text).strip()
    return text


def check_hf_model() -> dict:
    """Check if HuggingFace model can be loaded."""
    from config import HF_MODEL_PATH
    try:
        pipe = _get_hf_pipeline()
        return {"ok": True, "model": HF_MODEL_PATH, "error": ""}
    except Exception as e:
        return {"ok": False, "model": HF_MODEL_PATH, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 3: Fine-tuned model (LoRA adapter or merged)
# ══════════════════════════════════════════════════════════════════════════════

def _get_ft_pipeline():
    global _ft_pipeline
    if _ft_pipeline is not None:
        return _ft_pipeline

    from config import (
        FINETUNED_BASE_MODEL, FINETUNED_ADAPTER_PATH, FINETUNED_MERGED_PATH,
        FINETUNED_DEVICE, FINETUNED_LOAD_IN_4BIT, FINETUNED_MAX_NEW_TOKENS,
        FINETUNED_TEMPERATURE,
    )
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

        quant_config = None
        if FINETUNED_LOAD_IN_4BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        if FINETUNED_MERGED_PATH:
            # ── Fully merged fine-tuned model ─────────────────────────────
            model_path = FINETUNED_MERGED_PATH
            logger.info(f"Loading merged fine-tuned model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map=FINETUNED_DEVICE,
                trust_remote_code=True,
            )

        elif FINETUNED_ADAPTER_PATH:
            # ── LoRA / QLoRA adapter on top of base model ─────────────────
            # pyrefly: ignore [missing-import]
            from peft import PeftModel
            logger.info(f"Loading base: {FINETUNED_BASE_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(
                FINETUNED_BASE_MODEL, trust_remote_code=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                FINETUNED_BASE_MODEL,
                quantization_config=quant_config,
                device_map=FINETUNED_DEVICE,
                trust_remote_code=True,
            )
            logger.info(f"Loading LoRA adapter: {FINETUNED_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_PATH)
            model = model.merge_and_unload()  # merge for faster inference

        else:
            raise ValueError(
                "Fine-tuned backend selected but neither FINETUNED_MERGED_PATH "
                "nor FINETUNED_ADAPTER_PATH is set in config.py"
            )

        model.eval()
        _ft_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=FINETUNED_MAX_NEW_TOKENS,
            temperature=FINETUNED_TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.info("Fine-tuned model loaded successfully")
        return _ft_pipeline

    except ImportError as e:
        raise RuntimeError(
            f"Missing package: {e}\n"
            "Install with:\n"
            "  pip install transformers torch accelerate bitsandbytes peft"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load fine-tuned model: {e}")


def _call_finetuned(prompt: str, max_tokens: int) -> str:
    from config import FINETUNED_MAX_NEW_TOKENS
    pipe = _get_ft_pipeline()
    full_prompt = f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"
    outputs = pipe(
        full_prompt,
        max_new_tokens=min(max_tokens, FINETUNED_MAX_NEW_TOKENS),
        return_full_text=False,
    )
    text = outputs[0]["generated_text"].strip()
    text = re.sub(r"\[/?INST\]|<</?SYS>>", "", text).strip()
    return text


def check_finetuned_model() -> dict:
    """Check if fine-tuned model can be loaded."""
    from config import FINETUNED_ADAPTER_PATH, FINETUNED_MERGED_PATH, FINETUNED_BASE_MODEL
    try:
        pipe = _get_ft_pipeline()
        label = FINETUNED_MERGED_PATH or f"{FINETUNED_BASE_MODEL} + LoRA ({FINETUNED_ADAPTER_PATH})"
        return {"ok": True, "model": label, "error": ""}
    except Exception as e:
        return {"ok": False, "model": "", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE — used by enhancement_engine.py
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """
    Route to the configured backend and return the generated text.
    Never calls any external API — 100% local inference.
    """
    from config import LLM_BACKEND
    backend = LLM_BACKEND.lower().strip()

    if backend == "ollama":
        return _call_ollama(prompt, max_tokens)
    elif backend == "huggingface":
        return _call_hf(prompt, max_tokens)
    elif backend == "finetuned":
        return _call_finetuned(prompt, max_tokens)
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND: '{backend}'. "
            "Set LLM_BACKEND to 'ollama', 'huggingface', or 'finetuned' in config.py"
        )


def get_backend_status() -> dict:
    """Returns a status dict for the currently configured backend."""
    from config import LLM_BACKEND, OLLAMA_MODEL, HF_MODEL_PATH, FINETUNED_ADAPTER_PATH, FINETUNED_MERGED_PATH
    backend = LLM_BACKEND.lower().strip()

    if backend == "ollama":
        status = check_ollama()
        status["backend"] = "ollama"
        status["model_label"] = OLLAMA_MODEL
    elif backend == "huggingface":
        status = {"ok": True, "backend": "huggingface", "model_label": HF_MODEL_PATH,
                  "model_ready": True, "error": ""}
    elif backend == "finetuned":
        label = FINETUNED_MERGED_PATH or f"LoRA: {FINETUNED_ADAPTER_PATH}"
        status = {"ok": True, "backend": "finetuned", "model_label": label,
                  "model_ready": True, "error": ""}
    else:
        status = {"ok": False, "backend": backend, "model_label": "", "error": "Unknown backend"}

    return status


def preload_model():
    """
    Call this at startup to load the model into memory.
    For Ollama this is a no-op (Ollama manages its own memory).
    For HuggingFace/FineTuned this warms up the pipeline.
    """
    from config import LLM_BACKEND
    backend = LLM_BACKEND.lower().strip()
    if backend == "huggingface":
        logger.info("Preloading HuggingFace model...")
        _get_hf_pipeline()
    elif backend == "finetuned":
        logger.info("Preloading fine-tuned model...")
        _get_ft_pipeline()
    # Ollama: no preload needed
