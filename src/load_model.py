from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_base_model()
    print(f"[INFO] Model loaded on device: {model.device}")
    print(f"[INFO] Model dtype: {model.dtype}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable params: {trainable:,}")
    print(f"[INFO] Total params: {total:,}")