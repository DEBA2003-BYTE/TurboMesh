from ultralytics import YOLO
import torch


# -----------------------------------------
# Detect device
# -----------------------------------------

if torch.cuda.is_available():

    device = "cuda"

elif torch.backends.mps.is_available():

    device = "mps"

else:

    device = "cpu"


print("Using device:", device)


# -----------------------------------------
# Load YOLO model
# -----------------------------------------

model = YOLO("yolo11n.pt")


# -----------------------------------------
# Run object detection
# -----------------------------------------

results = model.predict(
    source="test.jpg",
    device=device,
    save=True
)


# -----------------------------------------
# Display detections
# -----------------------------------------

for result in results:

    print()
    print("Detected objects:")

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = result.names[
            class_id
        ]

        print(
            class_name,
            "confidence:",
            round(confidence, 2)
        )