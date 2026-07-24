---
title: FoodExtract Fine-tuned LLM Structured Data Extractor v1
emoji: 📝➡️🍟
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# FoodExtract

Fine-tuned [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it) to extract food and drink items from raw text.

**Input:** any short text or image caption, e.g.

    A truly eclectic feast is laid out on the table, featuring crispy fried chicken,
    a seared steak, and loaded tacos with mayonnaise. A vibrant assortment of fresh
    fruit sits nearby, including a red apple, a pineapple, and cherries. To drink,
    there's a classic iced latte, an earthy matcha latte, and a glass of milk.

**Output:** a structured extraction, e.g.

    food_or_drink: 1
    tags: fi, re
    foods: tacos, red apple, pineapple, cherries, fried chicken, steak, mayonnaise
    drinks: iced latte, matcha latte, milk

## Tags dictionary

| Tag | Meaning |
|-----|---------|
| np | nutrition_panel |
| il | ingredient_list |
| me | menu |
| re | recipe |
| fi | food_items |
| di | drink_items |
| fa | food_advertisement |
| fp | food_packaging |

- Model: https://huggingface.co/OmarShamakh/FoodExtract-gemma-3-270m-fine-tune-v1
- Dataset: https://huggingface.co/datasets/mrdbourke/FoodExtract-1k