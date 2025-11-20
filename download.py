import os
from transformers import MarianMTModel, MarianTokenizer

model_name = "facebook/nllb-200-distilled-600M"
local_model_dir = "ml/models"
local_path = os.path.join(local_model_dir, "nllb-200-distilled-600M")

# Nếu chưa có thì tải từ Hugging Face
if not os.path.exists(local_path):
    print(f"Downloading {model_name} ...")
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Save vào local
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
else:
    print(f"Loading from local {local_path} ...")
    model = MarianMTModel.from_pretrained(local_path)
    tokenizer = MarianTokenizer.from_pretrained(local_path)
