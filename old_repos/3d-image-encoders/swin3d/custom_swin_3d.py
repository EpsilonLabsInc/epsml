import copy

import torch
from torch import nn
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
from torchvision.models.video import swin3d_t, swin3d_s, swin3d_b

class CustomSwin3D(nn.Module):
    def __init__(self, model_size: str, num_classes, use_pretrained_weights, use_single_channel_input=False, use_swin_v2=True):
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
            self.__replace_swin_v1_with_swin_v2()

        # Accept single channel inputs instead of 3 channel ones.
        if use_single_channel_input:
            self.model.patch_embed.proj = nn.Conv3d(1, 128, kernel_size=(2, 4, 4), stride=(2, 4, 4))

    def forward(self, x):
        return self.model(x)

    def __replace_swin_v1_with_swin_v2(self):
        org_model = copy.deepcopy(self.model)

        # Replace SwinV1 'features' layer with SwinV2 'features' layer but
        # skip the first child layer since it's for 2-D inputs.
        self.model.features = copy.deepcopy(nn.Sequential(*list(self.__swin_2d_v2.features.children())[1:]))

        # Find all 'SwinTransformerBlockV2' layers and replace their 'attn' layer with the corresponding one from the Swin3D model.
        # The 'attn' layers in SwinV2 use 'ShiftedWindowAttentionV2' which is for 2-D.
        for index1, (name1, module1) in enumerate(self.model.features.named_children()):
            if module1.__class__.__name__ != "Sequential":
                continue

            for index2, (name2, module2) in enumerate(module1.named_children()):
                if module2.__class__.__name__ != "SwinTransformerBlockV2":
                    continue

                org_layer = list(list(org_model.features.children())[index1].children())[index2]
                module2.attn = org_layer.attn

        # Clean-up
        org_model = None
        del org_model
        self.__swin_2d_v2 = None
        del self.__swin_2d_v2
