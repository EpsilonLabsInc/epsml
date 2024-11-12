import torch.nn as nn


class SampleBalancedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(SampleBalancedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, labels):
        # Calculate sigmoid probabilities.
        probs = torch.sigmoid(logits)

        # Count number of 1s and 0s in each sample.
        num_ones = labels.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
        num_zeros = labels.size(1) - num_ones       # Shape: (batch_size, 1)

        # Avoid division by zero.
        num_ones = torch.where(num_ones == 0, torch.ones_like(num_ones), num_ones)
        num_zeros = torch.where(num_zeros == 0, torch.ones_like(num_zeros), num_zeros)

        # Calculate sample-specific weights.
        pos_weight = len(labels) / num_ones  # Shape: (batch_size, 1)
        neg_weight = len(labels) / num_zeros  # Shape: (batch_size, 1)
        pos_weight = pos_weight / (pos_weight + neg_weight)
        neg_weight = neg_weight / (pos_weight + neg_weight)

        # Apply weights to positive and negative parts of the loss.
        loss_pos = -pos_weight * labels * torch.log(probs + 1e-12)  # Positive label loss.
        loss_neg = -neg_weight * (1 - labels) * torch.log(1 - probs + 1e-12) # Negative label loss.

        # Combine positive and negative losses.
        loss = loss_pos + loss_neg

        # Apply reduction method
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # No reduction, returns a per-sample loss.
