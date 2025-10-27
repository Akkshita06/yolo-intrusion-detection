from ultralytics import YOLO
import cv2, os, sys, time, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "video5.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output_file.mp4")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "yolov8n.pt")
INTRUSION_SAVE_DIR = os.path.join(BASE_DIR, "intrusions")
CONF_THRESH = 0.3

def beep():
    try:
        import winsound
        winsound.Beep(1000, 300)
    except:
        sys.stdout.write("\a")
        sys.stdout.flush()

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: video not found at: {VIDEO_PATH}")
    sys.exit(1)
if not os.path.exists(MODEL_WEIGHTS):
    print(f"ERROR: model weights not found at: {MODEL_WEIGHTS}")
    sys.exit(1)

os.makedirs(INTRUSION_SAVE_DIR, exist_ok=True)

print("Loading YOLO model...")
model = YOLO(MODEL_WEIGHTS)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video:", VIDEO_PATH)
    sys.exit(1)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

restricted_zone = np.array([(0,0),(width,0),(width,height),(0,height)], np.int32)

def bbox_center_inside(x1, y1, x2, y2, polygon):
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    return cv2.pointPolygonTest(polygon, (cx,cy), False) >= 0

intrusion_count = 0
person_inside = False  # Tracks if a person is already inside

print("Processing video...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    alert_triggered = False

    results = model(frame, conf=CONF_THRESH, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else []
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else []
        cls   = r.boxes.cls.cpu().numpy() if len(r.boxes) else []

        for box, conf, c in zip(boxes, confs, cls):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(c)]} {conf:.2f}"
            color = (0,255,0)

            if model.names[int(c)].lower() == "person":
                if bbox_center_inside(x1, y1, x2, y2, restricted_zone):
                    color = (0,0,255)
                    alert_triggered = True  # Mark intrusion detected

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Only increment count when person enters
    if alert_triggered and not person_inside:
        intrusion_count += 1
        person_inside = True
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.putText(frame, f"INTRUSION #{intrusion_count} - {ts}",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
        beep()
        fname = os.path.join(INTRUSION_SAVE_DIR, f"intrusion_{intrusion_count}_{ts}.jpg")
        cv2.imwrite(fname, frame)

    # Reset flag only if no person is detected in this frame
    if not alert_triggered:
        person_inside = False

    cv2.polylines(frame, [restricted_zone], isClosed=True, color=(0,0,255), thickness=2)
    out.write(frame)
    cv2.imshow("Restricted Area Alert", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Finished. Output video:", OUTPUT_VIDEO)
cap.release()
out.release()
cv2.destroyAllWindows()
