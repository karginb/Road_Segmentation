from ultralytics import YOLO
import os
import cv2
import numpy as np 

SEG_MODEL_PATH = "/home/berat/Computer_Vision/Road_Segmentation/runs/segment/road_seg_yolov8/weights/best.pt"
DETECT_MODEL_PATH = "/home/berat/Computer_Vision/Road_Segmentation/yolov8n.pt"
DATA_PATH = "/home/berat/Computer_Vision/Road_Segmentation/road2.mp4"
OUTPUT_DIR = "runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

seg_model = YOLO(SEG_MODEL_PATH)
detect_model = YOLO(DETECT_MODEL_PATH)


CONFIDENCE_SCORE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(DATA_PATH)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
out_path = os.path.join(OUTPUT_DIR, "combined_output.mp4")
video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    detection_frame = frame.copy()

    seg_result = seg_model.predict(frame, imgsz=640, show_boxes=False, verbose=False)[0]
    seg_masked = seg_result.plot(conf=False, line_width=0)

    detect_results = detect_model.predict(detection_frame, imgsz=640, verbose=False)[0]
    boxes = np.array(detect_results.boxes.data.tolist())

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

        if score > CONFIDENCE_SCORE:
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            score *= 100
            class_name = detect_results.names[class_id]
            text = f"{class_name}: {score:.2f}"
            text_loc = (x1, y1 -10)
            
            label_size, baseline = cv2.getTextSize(text, FONT, 1, 1)
            cv2.rectangle(detection_frame, (x1, y1 - 10 - label_size[1]), (x1 + label_size[0], int(y1 + baseline - 10)), (0, 255, 0), cv2.FILLED)

            cv2.putText(detection_frame, text, (x1, y1 - 10), FONT, 1, (0, 0, 0), thickness = 3)


    combined = cv2.addWeighted(seg_masked, 0.7, detection_frame, 0.3, 0)

    cv2.imshow("Inference", combined)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

    video.write(combined)



cap.release()
video.release()
cv2.destroyAllWindows()
