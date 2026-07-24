from datasets import load_dataset
import ast
import random
from config import DATASET_NAME

def get_random_idx(dataset):
    return random.randint(0, len(dataset) - 1)

if __name__ == "__main__":
    dataset = load_dataset(DATASET_NAME)
    print(f"[INFO] Number of samples: {len(dataset['train'])}")

    random_idx = get_random_idx(dataset["train"])
    sample = dataset["train"][random_idx]

    print(f"\n[INFO] Input:\n{sample['sequence']}\n")
    print(f"[INFO] Structured JSON output:\n{ast.literal_eval(sample['gpt-oss-120b-label'])}\n")
    print(f"[INFO] Condensed output (what we'll train on):\n{sample['gpt-oss-120b-label-condensed']}")