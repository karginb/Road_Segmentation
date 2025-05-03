import os 
import torch
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

model_path = "segformer_output/checkpoint-940"  
image_path = "images/test/0006R0_f03780.png" 
output_mask_dir = "segformer_output/inference/"

os.makedirs(output_mask_dir, exist_ok=True)

target_class_id = 1
overlay_color = (0, 255, 255)  


processor = SegformerImageProcessor.from_pretrained(model_path)
model = SegformerForSemanticSegmentation.from_pretrained(model_path).to("cuda" if torch.cuda.is_available else "cpu")
model.eval()

image = Image.open(image_path).convert("RGB").resize((640, 640))
original_image = np.array(image)

inputs = processor(images = image, return_tensors = "pt").to(model.device)


with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits


preds = torch.argmax(logits, dim = 1)[0].cpu().numpy()


preds = Image.fromarray(preds.astype(np.uint8)).resize((640, 640), resample = Image.NEAREST)
preds = np.array(preds)

road_mask = (preds == target_class_id)

overlay_image = original_image.copy()
overlay_image[road_mask] = overlay_color

Image.fromarray(overlay_image).save(os.path.join(output_mask_dir + "segformer_mask_output3.png"))

