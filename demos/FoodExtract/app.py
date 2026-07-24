import time
import torch
import gradio as gr
import spaces

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_PATH = "OmarShamakh/FoodExtract-gemma-3-270m-fine-tune-v1"

print(f"[INFO] Loading model: {MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


@spaces.GPU
def pred_on_text(input_text):
    start_time = time.time()

    raw_output = pipe(
        text_inputs=[{"role": "user", "content": input_text}],
        max_new_tokens=256,
        disable_compile=True
    )

    total_time = round(time.time() - start_time, 4)
    generated_text = raw_output[0]["generated_text"][1]["content"]

    return generated_text, raw_output, total_time


description = """Extract food and drink items from text with a fine-tuned Small Language Model
([Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it)).

Fine-tuned on the [FoodExtract-1k dataset](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k).

Example:
- Input: "For breakfast I had eggs, bacon and toast and a glass of orange juice"
- Output:
    food_or_drink: 1
    tags: fi, di
    foods: eggs, bacon, toast
    drinks: orange juice
"""

demo = gr.Interface(
    fn=pred_on_text,
    inputs=gr.TextArea(lines=4, label="Input Text"),
    outputs=[
        gr.TextArea(lines=4, label="Generated Text"),
        gr.TextArea(lines=7, label="Raw Output"),
        gr.Number(label="Generation Time (s)")
    ],
    title="🍳 Structured FoodExtract with a Fine-Tuned Gemma 3 270M",
    description=description,
    examples=[
        ["Hello world! This is my first fine-tuned LLM!"],
        ["A plate of food with grilled barramundi, salad with avocado, olives, tomatoes and Italian dressing"],
        ["British Breakfast with baked beans, fried eggs, black pudding, sausages, bacon, mushrooms, a cup of tea and toast and fried tomatoes"],
        ["Steak tacos"],
        ["A photo of a dog sitting on a beach"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)