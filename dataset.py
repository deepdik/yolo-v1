from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch

# Move collate_fn outside of get_data_loader
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, 0)  # Stack images into a batch
    return images, targets

def get_data_loader(batch_size=16, num_workers=4):
    transform = T.Compose([
        T.Resize((448, 448)),  # Resize images to 448x448
        T.ToTensor()           # Convert images to tensors
    ])

    train_dataset = VOCDetection(root='./data', year='2007', image_set='train', download=True, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn  # Use the global collate_fn
    )

    return train_loader
