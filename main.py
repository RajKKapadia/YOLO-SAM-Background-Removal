import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

# Load YOLOv8 model (for detecting people)
yolo_model = YOLO("yolo12x.pt")

# Load image
image_path = "input.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect person with YOLO
results = yolo_model(image_rgb)
person_boxes = []
for box in results[0].boxes:
    cls_id = int(box.cls[0].item())
    if cls_id == 0:  # class 0 = person in COCO dataset
        print(box)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_boxes.append([x1, y1, x2, y2])

if not person_boxes:
    raise ValueError("No person detected in the image!")

# Load SAM
sam_checkpoint = "sam_vit_h.pth"  # Download from Meta's GitHub
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Get mask for the first detected person
box = np.array(person_boxes[0])
masks, _, _ = predictor.predict(box=box, multimask_output=False)
mask = masks[0]

# Apply mask with transparency
rgba_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
rgba_image[~mask] = (0, 0, 0, 0)  # Transparent background

# Save PNG
Image.fromarray(rgba_image).save("output.png")
