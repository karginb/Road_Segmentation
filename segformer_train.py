import os 
import numpy as np 
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"The device is: {device}")

csv_path = "images/class_dict.csv"
class_map = pd.read_csv(csv_path)

road_class = class_map[class_map["name"] == "Road"].iloc[0]
road_rgb = (road_class.r, road_class.g, road_class.b)

color2id = {
    road_rgb: 1
}

id2label = {
    0: "background",
    1: "road"
}

label2id = {
    "background": 0,
    "road": 1
}

class CamVidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_dir, self.images[index])).convert("RGB").resize((480, 640))
        mask = Image.open(os.path.join(self.mask_dir, self.masks[index])).convert("RGB").resize((480, 640), resample=Image.NEAREST)
        
        inputs = self.feature_extractor(image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        
        mask_array = np.array(mask)
        label_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        
        matches = np.all(mask_array == road_rgb, axis=-1)
        label_mask[matches] = 1
        inputs["labels"] = torch.tensor(label_mask, dtype=torch.long)
        return inputs

data_path = "images"

feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
train_dataset = CamVidDataset(os.path.join(data_path, "train"), os.path.join(data_path, "train_labels"), feature_extractor)
val_dataset = CamVidDataset(os.path.join(data_path, "val"), os.path.join(data_path, "val_labels"), feature_extractor)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

args = TrainingArguments(
    output_dir="~/Computer_Vision/Road_Segmentation/segformer_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    logging_dir="~/Computer_Vision/Road_Segmentation/segformer_output",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    resized_preds = []
    for pred in preds:
        pred_pil = Image.fromarray(pred.astype(np.uint8)).resize((480, 640), resample=Image.NEAREST)
        resized_preds.append(np.array(pred_pil))

    resized_preds = np.stack(resized_preds)
    acc = (resized_preds == labels).mean()
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.train()

last_checkpoint_dir = os.path.join(
    os.path.expanduser(args.output_dir),
    f"checkpoint-{trainer.state.global_step}"
)
feature_extractor.save_pretrained(last_checkpoint_dir)

