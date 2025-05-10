import os 
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import matplotlib.pyplot as plt

model_path = "segformer_output/checkpoint-940"
image_dir = "images/test"
mask_dir = "images/test_labels"
batch_size = 4
resize_dim = (640, 640)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

road_rgb = (128, 64, 128)

processor = SegformerImageProcessor.from_pretrained(model_path)
model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
model.eval()

def resize_image(image, size):
    return image.resize(size, resample=Image.NEAREST)

def process_batch(batch_images, batch_masks, model, processor):
    y_true, y_pred = [], []

    batch_images = [resize_image(img, resize_dim) for img in batch_images]
    batch_masks = [resize_image(mask, resize_dim) for mask in batch_masks]

    inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    for pred_mask, mask in zip(logits, batch_masks):
        pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()
        pred_mask_resized = resize_image(Image.fromarray(pred_mask.astype(np.uint8)), resize_dim)
        pred_mask_resized = np.array(pred_mask_resized)

        mask_array = np.array(mask.convert("RGB"))
        road_mask = np.all(mask_array == road_rgb, axis=-1).astype(np.uint8)

        y_true.extend(road_mask.flatten())
        y_pred.extend(pred_mask_resized.flatten())

    torch.cuda.empty_cache()
    return y_true, y_pred

def evaluate_model(image_dir, mask_dir, model, processor, batch_size):
    y_true, y_pred = [], []

    images = sorted(os.listdir(image_dir))
    num_samples = len(images)

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_images = []
        batch_masks = []

        for j in range(i, min(i + batch_size, num_samples)):
            image_path = os.path.join(image_dir, images[j])
            mask_name = (images[j].split(".png")[0] + "_L.png")
            mask_path = os.path.join(mask_dir, mask_name)

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")

            batch_images.append(image)
            batch_masks.append(mask)

        if batch_images:
            batch_true, batch_pred = process_batch(batch_images, batch_masks, model, processor)
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)

    print(f"y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")

    precision = precision_score(y_true, y_pred, pos_label=1, average="binary")
    recall = recall_score(y_true, y_pred, pos_label=1, average="binary")
    f1 = f1_score(y_true, y_pred, pos_label=1, average="binary")
    acc = accuracy_score(y_true, y_pred)

    intersection = np.logical_and(np.array(y_true) == 1, np.array(y_pred) == 1).sum()
    union = np.logical_or(np.array(y_true) == 1, np.array(y_pred) == 1).sum()
    iou = intersection / union if union != 0 else 0

    return {
        "F1 Score (Road)": f1,
        "Accuracy": acc,
        "Precision (Road)": precision,
        "Recall (Road)": recall,
        "IoU (Road)": iou
    }

def plot_metrics(results):
    metrics = list(results.keys())
    values = list(results.values())
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color="skyblue")
    plt.ylim(0, 1)
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title("Model Evaluation Metrics")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

results = evaluate_model(image_dir, mask_dir, model, processor, batch_size)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

plot_metrics(results)