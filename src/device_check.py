import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    DEVICE = get_device()
    print(f"[INFO] Using device: {DEVICE}")

    if DEVICE == "cuda":
        idx = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory
        print(f"[INFO] GPU: {torch.cuda.get_device_name(idx)}")
        print(f"[INFO] Total VRAM: {total / 1e9:.2f} GB")
    else:
        print("[WARNING] No CUDA GPU detected. Training will be slow/CPU-bound.")
        print("[INFO] If so, we'll run the training step on Google Colab instead.")