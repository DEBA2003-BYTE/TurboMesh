from ultralytics import YOLO
import torch


# Detect available GPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


print("Using device:", device)


# Load license plate model
print("Loading license plate model...")

model = YOLO("models/license_plate.pt")

print("License plate model loaded successfully!")


# Run detection
results = model.predict(
    source="Traffic.jpeg",
    device=device,
    conf=0.25,
    save=True
)


# Display detections
for result in results:

    print()
    print("Detected license plates:")

    if result.boxes is None or len(result.boxes) == 0:
        print("No license plates detected.")
        continue

    for box in result.boxes:

        confidence = float(box.conf[0])

        coordinates = box.xyxy[0].tolist()

        print(
            "License plate:",
            "confidence =", round(confidence, 2),
            "box =", [round(x, 1) for x in coordinates]
        )