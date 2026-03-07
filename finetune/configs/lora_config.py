import os
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

# Cấu hình LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

training_config = {
    "model": "pnnbao-ump/VieNeu-TTS-0.3B",
    "run_name": "VieNeu-TTS-0.3B-LoRA",
    "output_dir": os.path.join("finetune", "output"),
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "max_steps": 5000,
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
    "warmup_ratio": 0.05,
    "bf16": True,
    "use_4bit": False,
}


def get_training_args(config):
    return TrainingArguments(
        output_dir=os.path.join(config["output_dir"], config["run_name"]),
        do_train=True,
        do_eval=False,
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )
