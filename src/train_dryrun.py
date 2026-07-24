import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trl import SFTConfig, SFTTrainer

from config import CHECKPOINT_DIR
from load_model import load_base_model
from prepare_data import load_and_prepare_dataset

def main():
    print("[INFO] Loading model and tokenizer...")
    model, tokenizer = load_base_model()

    print("[INFO] Loading and preparing dataset...")
    dataset = load_and_prepare_dataset()

    # Shrink to a tiny subset just to test the pipeline runs end-to-end
    small_train = dataset["train"].select(range(20))
    small_eval = dataset["test"].select(range(10))

    sft_config = SFTConfig(
        output_dir="./checkpoint_models_dryrun",
        max_length=512,
        packing=False,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        completion_only_loss=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=5e-5,
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
        train_dataset=small_train,
        eval_dataset=small_eval,
        processing_class=tokenizer
    )

    print("[INFO] Starting dry-run training (20 samples, 1 epoch)...")
    train_output = trainer.train()
    print(train_output)

if __name__ == "__main__":
    main()