import torch
from torch import nn
from torchvision.models.video import swin3d_t, swin3d_s, swin3d_b, Swin3D_T_Weights, Swin3D_S_Weights, Swin3D_B_Weights

class CustomSwin3D(nn.Module):
    def __init__(self, model_size: str, num_classes, use_pretrained_weights):
        super(CustomSwin3D, self).__init__()

        self.model_size = model_size.lower()

        # Load the Swin3D model.
        if self.model_size == "tiny":
            weights = Swin3D_T_Weights.KINETICS400_V1 if use_pretrained_weights else None
            self.transform = weights.transforms()
            self.model = swin3d_t(weights=weights)
        elif self.model_size == "small":
            weights = Swin3D_S_Weights.KINETICS400_V1 if use_pretrained_weights else None
            self.transform = weights.transforms()
            self.model = swin3d_s(weights=weights)
        elif self.model_size == "base":
            weights = Swin3D_B_Weights.KINETICS400_V1 if use_pretrained_weights else None
            self.transform = weights.transforms()
            self.model = swin3d_b(weights=weights)
        else:
            raise TypeError("Argument model_size should be any of (tiny, small, base)")

        # Replace the head with a new linear layer.
        if num_classes != self.model.head.out_features:
            print(f"Replacing Swin3D head with a new linear layer with {num_classes} out features")
            self.model.head = nn.Linear(in_features=self.model.head.in_features,
                                        out_features=num_classes,
                                        bias=self.model.head.bias is not None)

    def forward(self, x):
        return self.model(x)

    def get_transform(self):
        return self.transform
