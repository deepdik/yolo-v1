
import torch.nn as nn
import torch
import torchvision.models as models

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )

        # Calculate the flattened size after the convolutional layers
        self._initialize_fc_input_size()

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C))
        )

    def _initialize_fc_input_size(self):
        # Use a dummy input to calculate the size of the output from conv_layers
        dummy_input = torch.zeros(1, 3, 448, 448)
        output = self.conv_layers(dummy_input)
        self.fc_input_size = output.view(-1).size(0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)





# class YOLOv1(nn.Module):
#     def __init__(self, S=7, B=2, C=20):
#         super(YOLOv1, self).__init__()
#         self.S = S
#         self.B = B
#         self.C = C

#         # Load a pre-trained ResNet model
#         resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the fully connected layer

#         # Freeze the backbone layers if desired
#         for param in self.backbone.parameters():
#             param.requires_grad = False

#         # Additional convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.1)
#         )

#         # Calculate the flattened size after the convolutional layers
#         self.fc_input_size = self._get_flattened_size()

#         # Fully connected layers
#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.fc_input_size, 4096),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.5),
#             nn.Linear(4096, S * S * (B * 5 + C))
#         )

#     def _get_flattened_size(self):
#         # Use a dummy input to calculate the size of the output from the conv_layers
#         dummy_input = torch.zeros(1, 3, 448, 448)
#         x = self.backbone(dummy_input)
#         x = self.conv_layers(x)
#         return x.view(1, -1).size(1)  # Get the flattened size

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.conv_layers(x)
#         x = self.fc_layers(x.view(x.size(0), -1))  # Flatten the output
#         return x.view(-1, self.S, self.S, self.B * 5 + self.C)


