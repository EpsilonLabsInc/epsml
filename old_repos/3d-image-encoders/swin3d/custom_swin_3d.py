import copy

import torch
from torch import nn
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
from torchvision.models.video import swin3d_t, swin3d_s, swin3d_b

class CustomSwin3D(nn.Module):
    def __init__(self, model_size: str, num_classes, use_pretrained_weights, use_swin_v2=True):
        super(CustomSwin3D, self).__init__()

        self.model_size = model_size.lower()
        weights = "DEFAULT" if use_pretrained_weights else None

        # Load the Swin3D model.
        if self.model_size == "tiny":
            self.model = swin3d_t(weights=weights)
            self.__swin_2d_v2 = swin_v2_t(weights=weights) if use_swin_v2 else None
        elif self.model_size == "small":
            self.model = swin3d_s(weights=weights)
            self.__swin_2d_v2 = swin_v2_s(weights=weights) if use_swin_v2 else None
        elif self.model_size == "base":
            self.model = swin3d_b(weights=weights)
            self.__swin_2d_v2 = swin_v2_b(weights=weights) if use_swin_v2 else None
        else:
            raise TypeError("Argument model_size should be any of (tiny, small, base)")

        # Replace the head with a new linear layer.
        if num_classes != self.model.head.out_features:
            print(f"Replacing Swin3D head with a new linear layer with {num_classes} out features")
            self.model.head = nn.Linear(in_features=self.model.head.in_features,
                                        out_features=num_classes,
                                        bias=self.model.head.bias is not None)

        # Replace with SwinV2 if necessary.
        if use_swin_v2:
            print("Replacing SwinV1 with SwinV2 in the Swin3D model")
            self.model.features = copy.deepcopy(self.__swin_2d_v2.features)
            self.__swin_2d_v2 = None
            del self.__swin_2d_v2

    def forward(self, x):
        return self.model(x)
