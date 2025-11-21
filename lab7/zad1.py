import cv2
import random
from ultralytics import YOLO
import json
import numpy as np
from collections import defaultdict

yolo = YOLO("yolo11x.pt")

def getColors(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "resources/office_yolo.mp4"
pic_path = "resources/office_yolo.png"
videoCap = cv2.VideoCapture(video_path)

CONFIDENCE_THRESHOLDS = [0.1, 0.3, 0.5, 0.7]
pic = cv2.imread(pic_path)


def yolo_pic(model, image_path, thresholds):
    pic = cv2.imread(image_path)

    image_results = {}
    class_names = model.names
    
    print("\n--- Procesowanie obrazu ---")

    for conf_thresh in thresholds:
        conf_str = f"{conf_thresh:.1f}".replace('.', '_')
        pic_annotated = pic.copy()
        
        results = model.predict(pic, conf=conf_thresh, verbose=False, boxes=True)

        detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes.cpu() 
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                cls = int(box.cls[0].numpy())
                conf = float(box.conf[0].numpy())
                
                class_name = class_names.get(cls, f"Class_{cls}")
                color = getColors(cls)

                detection_data = {
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2] 
                }
                detections.append(detection_data)
                
                cv2.rectangle(pic_annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(pic_annotated, (x1, max(y1 - 25, 0)), (x1 + w, y1), color, -1)
                cv2.putText(pic_annotated, label,
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
        
        json_filename = f"image_detections_{conf_str}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({"confidence_threshold": conf_thresh, "detections": detections}, f, indent=4)
        
        image_filename = f"image{conf_str}.png"
        cv2.imwrite(image_filename, pic_annotated)
        
        image_results[conf_thresh] = {
            "json_file": json_filename,
            "image_file": image_filename,
            "num_detections": len(detections)
        }
        
    return image_results, class_names
                


def yolo_video(model, video_path, thresholds):
    video_cap = cv2.VideoCapture(video_path)

    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n--- Procesowanie wideo ---")

    representative_conf = 0.5
    conf_str_rep = f"{representative_conf:.1f}".replace('.', '_')
    video_output_path = f"video{conf_str_rep}.mp4"
    video_writer = cv2.VideoWriter(video_output_path, 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   fps, (width, height))

    all_threshold_detections = {thresh: [] for thresh in thresholds}
    total_class_counts = defaultdict(int)
    
    frame_count = 0

    min_conf = min(thresholds)

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
            
        frame_time_ms = video_cap.get(cv2.CAP_PROP_POS_MSEC)
        
        if frame_count >= 100: 
            break

        results_raw = model.predict(frame, conf=min_conf, verbose=False, boxes=True)
        
        frame_annotated = frame.copy()
        
        if results_raw and results_raw[0].boxes:
            boxes_raw = results_raw[0].boxes.cpu() 
            
            for conf_thresh in thresholds:
                current_frame_detections = []
                is_representative = (conf_thresh == representative_conf)

                for box in boxes_raw:
                    conf = float(box.conf[0].numpy())
                    
                    if conf >= conf_thresh:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                        cls = int(box.cls[0].numpy())
                        
                        class_name = model.names.get(cls, f"Class_{cls}")
                        
                        detection_data = {
                            "frame_number": frame_count,
                            "time_ms": frame_time_ms,
                            "class_id": cls,
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox_xyxy": [x1, y1, x2, y2]
                        }
                        current_frame_detections.append(detection_data)
                    
                        if is_representative:
                            total_class_counts[class_name] += 1
                            
                            color = getColors(cls)
                            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
                            
                            label = f"{class_name} {conf:.2f}"
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame_annotated, (x1, max(y1 - 25, 0)), (x1 + w, y1), color, -1)
                            cv2.putText(frame_annotated, label,
                                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (255, 255, 255), 2)
                
                all_threshold_detections[conf_thresh].extend(current_frame_detections)
            
            video_writer.write(frame_annotated)
            
        frame_count += 1
        
    video_cap.release()
    video_writer.release()
    
    video_results = {}
    for conf_thresh, detections_list in all_threshold_detections.items():
        conf_str = f"{conf_thresh:.1f}".replace('.', '_')
        json_filename = f"video_detections_{conf_str}.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({"confidence_threshold": conf_thresh, "detections": detections_list}, f, indent=4)
            
        video_results[conf_thresh] = {
            "json_file": json_filename,
            "num_detections": len(detections_list)
        }

    stats_filename = f"video_stats_{conf_str_rep}.json"
    with open(stats_filename, 'w', encoding='utf-8') as f:
        sorted_counts = dict(sorted(total_class_counts.items(), key=lambda item: item[1], reverse=True))
        json.dump({"confidence_threshold": representative_conf, "class_counts": sorted_counts}, f, indent=4)
        
    video_results["stats"] = {
        "stats_file": stats_filename,
        "video_file": video_output_path,
        "class_counts": sorted_counts
    }

    return video_results


if __name__ == "__main__":
    image_summary, class_names = yolo_pic(yolo, pic_path, CONFIDENCE_THRESHOLDS)
    
    video_summary = yolo_video(yolo, video_path, CONFIDENCE_THRESHOLDS)

    
    for conf, res in image_summary.items():
        print(f"* **Conf={conf:.1f}** ({res['num_detections']} detekcji):")
        print(f"  - Plik JSON: {res['json_file']}")
        print(f"  - Plik obrazu: {res['image_file']}")
    
    print("\n" + "---")

    representative_conf = 0.5
    for conf, res in video_summary.items():
        if conf != "stats":
            print(f"* **Conf={conf:.1f}** ({res['num_detections']} detekcji łącznie): Plik JSON: {res['json_file']}")
        else:
            print(f"\n* **Wideo z bounding boxami** (Conf={representative_conf:.1f}): {res['video_file']}")
            print(f"* **Statystyki** (Conf={representative_conf:.1f}): {res['stats_file']}")
            
            print("### Statystyki wideo (Conf=0.5):")
            top_n = 5
            print(f"Top {top_n} najczęściej wykrywanych klas:")
            for i, (cls, count) in enumerate(res['class_counts'].items()):
                if i < top_n:
                    print(f"- **{cls}**: {count} detekcji")
