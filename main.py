from ultralytics import YOLO
import cv2
import time

# Models
general_model = YOLO("yolov8m.pt")
helmet_model = YOLO("runs/detect/train/weights/best.pt")

# Webcam
cap = cv2.VideoCapture(0)

frame_width = 640
frame_height = 480

cap.set(3, frame_width)
cap.set(4, frame_height)

# Video Writers
out = cv2.VideoWriter(
    "output.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    20,
    (frame_width, frame_height)
)

# Counting Line
line_x = 320

crossed_ids = set()
total_count = 0

prev_time = 0
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    frame_count += 1

    if frame_count % 2 != 0:
        continue

    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

    prev_time = current_time

    # =========================
    # COCO OBJECT DETECTION
    # =========================
    general_results = general_model.track(
        frame,
        persist=True,
        imgsz=640,
        conf=0.4,          # Increased from 0.5 to reduce general false positives
        half=True          # Added half=True (FP16) to optimize for faster inference if using a GPU
    )

    # =========================
    # HELMET DETECTION
    # =========================
    helmet_results = helmet_model.predict(
        frame,
        imgsz=320,
        conf=0.65,          # Increased from 0.5 to 0.8 to filter out the false helmet detections like your hair
        half=True          # Added half=True (FP16) to speed up inference
    )

    # Draw counting line
    cv2.line(
        frame,
        (line_x, 0),
        (line_x, 480),
        (0, 0, 255),
        3
    )

    object_counts = {}

    # =========================
    # PROCESS COCO OBJECTS
    # =========================
    for result in general_results:

        boxes = result.boxes

        for box in boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            class_id = int(box.cls[0])

            class_name = general_model.names[class_id]

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if class_name not in object_counts:
                object_counts[class_name] = 0

            object_counts[class_name] += 1

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Person crossing logic
            if class_name == "person":

                if center_x > line_x:

                    if track_id not in crossed_ids:

                        crossed_ids.add(track_id)

                        total_count += 1

                        print(
                            f"Person {track_id} crossed line"
                        )

    # =========================
    # PROCESS HELMET MODEL
    # =========================
    for result in helmet_results:

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = helmet_model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (255, 0, 0)

            if "no" in class_name.lower():
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # =========================
    # DISPLAY COUNTS
    # =========================
    y_position = 30

    for obj_name, count in sorted(object_counts.items()):

        cv2.putText(
            frame,
            f"{obj_name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        y_position += 22

        if y_position > 400:
            break

    cv2.putText(
        frame,
        f"Cross Count: {total_count}",
        (20, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    out.write(frame)

    cv2.imshow(
        "Multi Object + Helmet Detection System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()