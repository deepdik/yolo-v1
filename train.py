import torch
import torch.optim as optim
from model import YOLOv1
from loss import YoloLoss
from dataset import get_data_loader

def preprocess_targets(targets, S, B, C):
    # Existing code for preprocessing targets
    batch_size = len(targets)
    target_tensor = torch.zeros((batch_size, S, S, B * 5 + C))

    # Define class mapping
    class_mapping = {
        "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
        "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9,
        "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14,
        "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19
    }

    for i, target in enumerate(targets):
        # Extract image width and height from the annotation
        image_width = int(target['annotation']['size']['width'])
        image_height = int(target['annotation']['size']['height'])

        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]  # Ensure we have a list, even if there's only one object

        for obj in objects:
            # Extract class label and bounding box
            class_name = obj['name']
            bbox = obj['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])

            # Convert class name to class index using the mapping
            class_index = class_mapping[class_name]

            # Calculate the center (x, y), width, and height of the bounding box
            x_center = (xmin + xmax) / 2 / image_width  # Normalize by image width
            y_center = (ymin + ymax) / 2 / image_height  # Normalize by image height
            width = (xmax - xmin) / image_width  # Normalize by image width
            height = (ymax - ymin) / image_height  # Normalize by image height

            # Map the center (x, y) to the S x S grid cell
            grid_x = int(x_center * S)
            grid_y = int(y_center * S)

            # Check if the grid cell already has an object
            if target_tensor[i, grid_y, grid_x, 4] == 0:  # If confidence is 0, assign
                # Fill in the target tensor
                target_tensor[i, grid_y, grid_x, 0:4] = torch.tensor([x_center, y_center, width, height])
                target_tensor[i, grid_y, grid_x, 4] = 1  # Confidence score
                target_tensor[i, grid_y, grid_x, 5 + class_index] = 1  # One-hot class label
            # Else, handle cases where multiple objects fall in the same grid cell (optional)

    return target_tensor

def train_yolo(num_epochs=100, batch_size=16, learning_rate=0.001):
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    model = YOLOv1().to(device)
    criterion = YoloLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    train_loader = get_data_loader(batch_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = preprocess_targets(targets, S=7, B=2, C=20).to(device)  # Convert targets to tensor format

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")

        # Save model checkpoint after each epoch
        checkpoint_path = f"checkpoints/yolov1_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    # Save the final model
    final_model_path = "checkpoints/yolov1_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

if __name__ == "__main__":
    train_yolo()
