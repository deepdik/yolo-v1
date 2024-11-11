import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        batch_size = predictions.size(0)

        predictions = predictions.view(batch_size, self.S, self.S, self.B * 5 + self.C)
        target = target.view(batch_size, self.S, self.S, self.B * 5 + self.C)

        # Extract coordinates, confidence, and class probabilities
        pred_boxes = predictions[..., :self.B * 4].view(batch_size, self.S, self.S, self.B, 4)
        target_boxes = target[..., :self.B * 4].view(batch_size, self.S, self.S, self.B, 4)
        conf_pred = predictions[..., self.B * 4:self.B * 5].view(batch_size, self.S, self.S, self.B)
        conf_target = target[..., self.B * 4:self.B * 5].view(batch_size, self.S, self.S, self.B)
        class_pred = predictions[..., self.B * 5:].view(batch_size, self.S, self.S, self.C)
        class_target = target[..., self.B * 5:].view(batch_size, self.S, self.S, self.C)

        # Coordinate loss
        coord_mask = target[..., 4] > 0
        coord_mask = coord_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_boxes)
        coord_loss = self.lambda_coord * torch.sum(((pred_boxes - target_boxes) ** 2) * coord_mask)

        # Object and no-object confidence loss
        object_mask = target[..., 4] > 0
        no_object_mask = target[..., 4] == 0
        object_loss = torch.sum((conf_pred - conf_target) ** 2 * object_mask.unsqueeze(-1))
        no_object_loss = self.lambda_noobj * torch.sum((conf_pred - conf_target) ** 2 * no_object_mask.unsqueeze(-1))

        # Classification loss
        class_loss = torch.sum((class_pred - class_target) ** 2 * object_mask.unsqueeze(-1))

        # Total loss
        total_loss = coord_loss + object_loss + no_object_loss + class_loss
        return total_loss
