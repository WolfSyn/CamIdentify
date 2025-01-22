import cv2
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('yolov5s.pt')  # Using the small pretrained model

# Define the camera feed (use 0 for the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect objects using YOLO
    results = model(frame)

    # Annotate the frame with bounding boxes and labels
    for r in results:
        for box in r.boxes:
            # Extract box coordinates, label, and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()
            cls = box.cls.cpu().numpy()
            label = model.names[int(cls)]

            # Ensure label is a string and conf is a float
            label = str(label)
            conf = float(conf)

            # Format the label text
            text = "{} {:.2f}".format(label, conf)

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
