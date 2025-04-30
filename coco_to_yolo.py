import os
import json
from tqdm import tqdm

# Yollar
directories = ["train", "test", "validation"]

for i in directories:
    coco_json_path = i + '.json'
    images_dir = os.path.join('CamVid/' + i)
    output_labels_dir = os.path.join('labels/' + i)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    
    id2name = {img['id']: img['file_name'] for img in coco['images']}
    
    for ann in tqdm(coco['annotations']):
        image_id = ann['image_id']
        segmentation = ann['segmentation'][0]

        image_width = [img['width'] for img in coco['images'] if img['id'] == image_id][0]
        image_height = [img['height'] for img in coco['images'] if img['id'] == image_id][0]
        
        norm_seg = []
        for i in range(0, len(segmentation), 2):
            x = segmentation[i] / image_width
            y = segmentation[i+1] / image_height
            norm_seg.extend([x, y])
            
        label_path = os.path.join(output_labels_dir, id2name[image_id].replace('.png', '.txt').replace('.jpg', '.txt'))
        with open(label_path, 'a') as f:
            f.write(f"0 " + " ".join(map(str, norm_seg)) + "\n")
