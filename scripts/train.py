import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, TrainerCallback,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model

class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

# Custom Trainer
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None or not loss.requires_grad:
            raise ValueError("Loss is missing or not requiring grad! Check data/model config.")
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # System message to guide the model's behavior
    system_message = "You are an AI assistant trained to provide accurate, concise, and helpful responses to user queries."

    # 1) Load dataset
    dataset = load_dataset("json", data_files={"train": "../data/data.jsonl"})

    # 2) Tokenizer & model
    model_name_or_path = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Tokenize function
    def tokenize_function(examples):
        # Apply system message to each example in the batch
        input_texts = [f"System: {system_message}\n{text}" for text in examples["text"]]
        encodings = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=1024
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings


    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format("torch")

    # 4) Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=5,
        save_steps=200,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=50,
        fp16=True,
        overwrite_output_dir=True,
    )

    # 5) 4-bit quantization config (optional)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 6) Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    # 7) LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 8) Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 9) Trainer
    latest_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        callbacks=[ClearCacheCallback()],
    )

    if latest_checkpoint is not None:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    # 10) Save
    save_path = "../flask_app/models/blackgpt"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
