import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from trl import SFTConfig, SFTTrainer

from config import CHECKPOINT_DIR
from load_model import load_base_model
from prepare_data import load_and_prepare_dataset

def main():
    print("[INFO] Loading model and tokenizer...")
    model, tokenizer = load_base_model()

    print("[INFO] Loading and preparing dataset...")
    dataset = load_and_prepare_dataset()

    BATCH_SIZE = 4  # small, since we're on CPU
    LEARNING_RATE = 5e-5

    sft_config = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        max_length=512,
        packing=False,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        completion_only_loss=True,
        gradient_checkpointing=False,
        optim="adamw_torch",  # note: adamw_torch_fused requires CUDA
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=False,
        load_best_model_at_end=True,
        metric_for_best_model="mean_token_accuracy",
        greater_is_better=True,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer
    )

    print("[INFO] Starting training...")
    train_output = trainer.train()
    print(train_output)

    return trainer

if __name__ == "__main__":
    trainer = main()