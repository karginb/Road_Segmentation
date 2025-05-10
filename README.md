# Road Segmentation Project

This repository contains the implementation of a road segmentation project using two state-of-the-art deep learning models: YOLOv8-Seg and SegFormer. The objective of this project is to accurately identify and segment road surfaces in urban driving scenes using the CamVid dataset.

## 📦 Dataset

The CamVid dataset, consisting of 701 images captured in various road scenarios, was used for training, validation, and testing. Each image is paired with a pixel-wise labeled mask, and the dataset is divided as follows:
- Training set: 367 images
- Validation set: 101 images
- Test set: 233 images

Dataset Link: [CamVid Dataset - Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid)

## 🚀 Models Implemented

1. **YOLOv8-Seg:**
   - Utilizes a bounding box-based approach for segmentation.
   - Lightweight and suitable for real-time applications.
   - Achieved high accuracy with lower computational cost.

2. **SegFormer:**
   - Transformer-based architecture designed for semantic segmentation.
   - Capable of capturing fine-grained details and sharper boundaries.
   - Requires higher computational power but produces more precise segmentation maps.

## 📊 Evaluation Metrics
The following evaluation metrics were used to assess model performance:
- **Accuracy**: Measures overall pixel-wise classification accuracy.
- **F1-Score**: Evaluates the balance between precision and recall.
- **Precision**: Indicates the proportion of correctly classified road pixels.
- **Recall**: Measures how well the model identifies road pixels among actual road pixels.
- **IoU (Intersection over Union)**: Provides a comprehensive measure of overlap between predicted and ground truth masks.

| Model    | Accuracy | F1-Score | Precision | Recall | IoU  |
|----------|-----------|----------|-----------|--------|------|
| YOLOv8-Seg | 99%      | 98%      | 96%       | 98%    | 94%  |
| SegFormer | 97%      | 95%      | 95%       | 94%    | 92%  |

## 🛠️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/karginb/Road_Segmentation.git
cd Road_Segmentation
```
2. Download the CamVid dataset and place it in the `images/` directory.
```bash
https://www.kaggle.com/datasets/carlolepelaars/camvid
```
3. Train the models:
```bash
python yolo_segment_train.py
python segformer_train.py
```
4. Inference:
```bash
python yolo_segment_inference.py
python segformer_inference.py
```

## 📈 Results and Analysis
- YOLOv8-Seg achieved faster inference times and demonstrated strong accuracy with minimal computational resources.
- SegFormer provided more detailed and refined segmentation maps, particularly in complex road patterns and occluded areas.
- The trade-off between processing speed and segmentation quality was evident, suggesting potential for hybrid models in future work.

![YOLOv8-Seg Result](/home/berat/Computer_Vision/Road_Segmentation/runs/road_seg_img_result/Seq05VD_f04110.png)

![SegFormer Result](/home/berat/Computer_Vision/Road_Segmentation/segformer_output/inference/Seq05VD_f04110.png)



## 🔗 References
- [CamVid Dataset](https://www.kaggle.com/datasets/carlolepelaars/camvid)
- [YOLOv8 Documentation](https://github.com/ultralytics/yolov8)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)

