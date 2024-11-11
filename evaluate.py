import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import YOLOv1Pretrained  # Import your model class

# Load the trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLOv1Pretrained().to(device)
model.load_state_dict(torch.load("checkpoints/yolov1_epoch_100.pth", map_location=device))
model.eval()  # Set the model to evaluation mode

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Postprocess the model's output to get bounding boxes and labels
def postprocess_output(output, S=7, B=2, C=20, conf_threshold=0.5):
    output = output[0]  # Remove batch dimension
    boxes = []
    for i in range(S):
        for j in range(S):
            for b in range(B):
                confidence = output[i, j, b * 5 + 4]
                if confidence > conf_threshold:
                    x, y, w, h = output[i, j, b * 5:b * 5 + 4]
                    x = (j + x) / S  # Convert to image coordinates
                    y = (i + y) / S
                    w = w ** 2  # YOLOv1 uses the square root of width and height
                    h = h ** 2
                    boxes.append((x, y, w, h, confidence))
    return boxes

# Visualize the bounding boxes on the image
def visualize_image(image_path, boxes):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    image_width, image_height = image.size
    for (x, y, w, h, confidence) in boxes:
        x1 = (x - w / 2) * image_width
        y1 = (y - h / 2) * image_height
        x2 = (x + w / 2) * image_width
        y2 = (y + h / 2) * image_height
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{confidence:.2f}", color="red", fontsize=12)

    plt.show()

# Use the model to make predictions on a new image
image_path = "path_to_your_image.jpg"
image = preprocess_image(image_path)
with torch.no_grad():  # No need to compute gradients for inference
    output = model(image)

boxes = postprocess_output(output)
visualize_image(image_path, boxes)
