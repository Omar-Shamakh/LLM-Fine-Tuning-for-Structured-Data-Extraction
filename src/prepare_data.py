from datasets import load_dataset
from config import DATASET_NAME

# Tags our model will learn to assign
tags_dict = {
    'np': 'nutrition_panel',
    'il': 'ingredient_list',
    'me': 'menu',
    're': 'recipe',
    'fi': 'food_items',
    'di': 'drink_items',
    'fa': 'food_advertistment',
    'fp': 'food_packaging'
}

def sample_to_conversation(sample):
    """Convert a raw dataset sample into prompt/completion message format."""
    return {
        "prompt": [
            {"role": "user", "content": sample["sequence"]}
        ],
        "completion": [
            {"role": "assistant", "content": sample["gpt-oss-120b-label-condensed"]}
        ]
    }

def load_and_prepare_dataset():
    dataset = load_dataset(DATASET_NAME)

    # Map raw samples into prompt/completion format
    dataset = dataset.map(sample_to_conversation, batched=False)

    # Create an 80/20 train/test split
    dataset = dataset["train"].train_test_split(
        test_size=0.2,
        shuffle=False,
        seed=42
    )
    return dataset

if __name__ == "__main__":
    dataset = load_and_prepare_dataset()

    print(f"[INFO] Train samples: {len(dataset['train'])}")
    print(f"[INFO] Test samples: {len(dataset['test'])}")

    print(f"\n[INFO] Example formatted sample:")
    print(dataset["train"][0])