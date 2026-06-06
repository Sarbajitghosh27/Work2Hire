"""
core/finetune.py
────────────────────────────────────────────────────────
LoRA Fine-Tuning Script for Qwen 2.5 1.5B Instruct.
Extracts transformation examples (weak -> strong) from logs/generation_log.jsonl
and runs supervised fine-tuning using PEFT, Accelerate, and TRL.

Includes an out-of-the-box synthetic dataset generator if the log file is empty.
"""

import os
import json
import logging
import argparse
from typing import List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA FALLBACK
# ──────────────────────────────────────────────────────────────────────────────
SYNTHETIC_TRANSFORMS = [
    {
        "prompt": "Rewrite the following resume bullet point for a Software Engineer using a strong action verb and quantified metric: 'I was responsible for fixing bugs and worked on the main website backend.'",
        "completion": "Optimized backend system architecture, resolving 50+ critical performance bugs and improving page load speeds by 28%."
    },
    {
        "prompt": "Rewrite the following resume bullet point for a Machine Learning Engineer using a strong action verb and quantified metric: 'I helped train the product recommendation model using PyTorch.'",
        "completion": "Developed and trained a PyTorch-based product recommendation algorithm, increasing user click-through rate (CTR) by 14%."
    },
    {
        "prompt": "Rewrite the following resume bullet point for a DevOps Engineer using a strong action verb and quantified metric: 'Used Jenkins and Docker to make builds faster.'",
        "completion": "Automated CI/CD pipelines via Docker and Jenkins, reducing overall deployment cycle times by 45%."
    },
    {
        "prompt": "Rewrite the following resume bullet point for an HR Specialist using a strong action verb and quantified metric: 'Handled candidate interviewing and onboarding.'",
        "completion": "Overhauled end-to-end recruitment and onboarding workflows, reducing average time-to-hire by 12 days and increasing retention by 20%."
    },
    {
        "prompt": "Rewrite the following resume bullet point for a Financial Analyst using a strong action verb and quantified metric: 'Did the budget forecasting in Excel.'",
        "completion": "Constructed complex financial models and forecasting templates in Excel, trimming budget discrepancies from 8% to under 1.5%."
    },
    {
        "prompt": "Rewrite the following resume bullet point for a Marketing Specialist using a strong action verb and quantified metric: 'I ran the email campaigns and did social media posting.'",
        "completion": "Designed and launched targeted email and social media campaigns, boosting inbound leads by 35% and sales conversions by 12%."
    },
    {
        "prompt": "Rewrite the following resume bullet point for an Operations Manager using a strong action verb and quantified metric: 'Responsible for inventory tracking and shipping logistics.'",
        "completion": "Redesigned warehouse inventory tracking protocols, reducing shipping delays by 40% and cutting overhead costs by 15%."
    }
]

def load_dataset_from_logs(log_path: str) -> List[Dict[str, str]]:
    """Loads dataset from JSONL generation logs, or generates synthetic fallback data."""
    dataset = []
    
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    # Convert record fields into an instruct prompt-response format
                    if record.get("ats_delta", 0) > 5.0: # only use runs with notable score improvement
                        prompt = (
                            f"Optimize this resume candidate's profile for the domain '{record.get('jd_domain')}' "
                            f"and seniority '{record.get('jd_seniority')}'. Enhance skills and keywords."
                        )
                        completion = (
                            f"Enhanced ATS score from {record.get('original_ats_score')}% to {record.get('enhanced_ats_score')}% "
                            f"with {record.get('enhanced_skill_count')} optimized skills."
                        )
                        dataset.append({"prompt": prompt, "completion": completion})
        except Exception as e:
            logger.warning(f"Error reading log file: {e}. Falling back to synthetic dataset.")
            
    if len(dataset) < 10:
        logger.info(f"Log file has insufficient data ({len(dataset)} examples). Generating synthetic training examples...")
        dataset = []
        # Replicate transforms to fill a baseline dataset size (e.g. 50 examples)
        for i in range(7):
            for t in SYNTHETIC_TRANSFORMS:
                dataset.append(t)
                
    return dataset

# ──────────────────────────────────────────────────────────────────────────────
# LORA TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_training(
    base_model_id: str,
    output_dir: str,
    log_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(
            f"Missing dependencies for training: {e}\n"
            "Please install the required packages:\n"
            "  pip install transformers torch accelerate peft datasets trl bitsandbytes"
        )
        return

    logger.info(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Load Data
    raw_data = load_dataset_from_logs(log_path)
    logger.info(f"Dataset loaded: {len(raw_data)} training examples.")
    
    # Map to Chat/Instruct format suitable for Qwen/Instruct models
    def format_chats(example):
        return {
            "text": f"<|im_start|>system\nYou are a professional ATS resume engineer.<|im_end|>\n"
                    f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{example['completion']}<|im_end|>"
        }
        
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_chats)
    
    # 2. Configure QLoRA (4-bit quantization to fit in standard GPU VRAM)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    logger.info("Loading quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. Configure LoRA adapter
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        warmup_ratio=0.03,
        report_to="none"
    )
    
    # 5. Launch SFTTrainer
    logger.info("Starting training loop...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        peft_config=peft_config,
        args=training_args,
    )
    
    trainer.train()
    
    logger.info(f"Training completed. Saving LoRA adapter to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Success! LoRA adapters saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 1.5B Instruct using LoRA on resume changes.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face ID or path of the base model.")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_lora", help="Output directory for LoRA adapters.")
    parser.add_argument("--log_path", type=str, default="logs/generation_log.jsonl", help="Path to jsonl log file.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    
    args = parser.parse_args()
    
    run_training(
        base_model_id=args.base_model,
        output_dir=args.output_dir,
        log_path=args.log_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
