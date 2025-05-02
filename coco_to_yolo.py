import os
import json
from tqdm import tqdm

# Clamp fonksiyonu
def clamp(val):
    return min(max(val, 0.0), 1.0)

directories = ["train", "test", "val"]

for i in directories:
    coco_json_path = os.path.join("Json_Files/" + i + '.json')
    images_dir = os.path.join('CamVid/' + i)
    output_labels_dir = os.path.join('labels/' + i)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    
    id2name = {img['id']: img['file_name'] for img in coco['images']}
    image_dims = {img['id']: (img['width'], img['height']) for img in coco['images']}

    for ann in tqdm(coco['annotations'], desc=f"Converting {i}.json"):
        image_id = ann['image_id']
        segmentation = ann['segmentation'][0]

        width, height = image_dims[image_id]
        
        norm_seg = []
        for j in range(0, len(segmentation), 2):
            x = clamp(segmentation[j] / width)
            y = clamp(segmentation[j+1] / height)
            norm_seg.extend([x, y])
        
        label_path = os.path.join(output_labels_dir, id2name[image_id].rsplit('.', 1)[0] + ".txt")
        with open(label_path, 'a') as f:
            f.write(f"0 " + " ".join(map(str, norm_seg)) + "\n")
