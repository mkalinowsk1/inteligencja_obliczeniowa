import cv2
import random
from ultralytics import YOLO

yolo = YOLO("yolo11x.pt")

def getColors(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "resources/office_yolo.mp4"
pic_path = "resources/office_yolo.png"
videoCap = cv2.VideoCapture(video_path)
pic = cv2.imread(pic_path)


def yolo_pic(picture):
    results = yolo.predict(picture)
    for result in results:
        class_names = result.names
        for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls = int(box.cls[0])
                    class_name = class_names[cls]

                    conf = float(box.conf[0])

                    color = getColors(cls)

                    cv2.rectangle(picture, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(picture, f"{class_name} {conf:.2f}",
                                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2)
                
    cv2.imshow("yolo", picture)
    return cv2.waitKey(0)
                


def yolo_video(videoCap):
    frame_count = 0

    while True:
        ret, frame = videoCap.read()
        if not ret:
            break
        results = yolo.track(frame, stream=True)

        for result in results:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls = int(box.cls[0])
                    class_name = class_names[cls]

                    conf = float(box.conf[0])

                    color = getColors(cls)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

        if frame_count < 20:
            cv2.imshow("YOLOOOOOOOOO", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break

        frame_count += 1

    videoCap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #yolo_video(videoCap)
    yolo_pic(pic)