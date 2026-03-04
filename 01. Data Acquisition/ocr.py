import requests
import os
import csv
import torch

from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from PIL import Image
from others.config import Config


class VisualCaptioning:
    def generate_caption(image_url : str, processor, model, prompt : str) -> str:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        text = processor.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=80, temperature=0.2)

        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if "assistant" in output:
            output = output.split("assistant")[-1].strip()

        return output

def generate_caption_from_apps_screenshots(apps : list, filename = Config.APPS_OCR_CAPTIONS_FILENAME, model_name="Qwen/Qwen2VL-7B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name).to(device)
    prompt = "Generate a short, high-signal caption describing only the main functional features visible in this mobile app screenshot. Focus on core educational tools (e.g., quiz types, exercises, progress indicators, learning modules, audio practice, streak tracking). Ignore decorative UI elements, colors, and layout details unless essential. Be precise and under 25 words"

    processed_ids = set()

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_ids.add(row["caption_id"])

    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["caption_id", "app_id", "screenshot", "caption"]
        )

        if not file_exists:
            writer.writeheader()

        for app in tqdm(apps):
            for i, screenshot in enumerate(app["screenshots"]):

                if i > 2: # limit 3
                    break

                caption_id = f"{app['appId']}_screenshot_{i}"

                if caption_id in processed_ids:
                    continue

                caption = VisualCaptioning.generate_caption(screenshot, processor, model, prompt)

                if caption is None:
                    continue

                row = {
                    "caption_id": caption_id,
                    "app_id": app["appId"],
                    "screenshot": screenshot,
                    "caption": caption
                } 

                writer.writerow(row)
                csvfile.flush() 

                print(f"Generated caption for {caption_id}: {caption}")