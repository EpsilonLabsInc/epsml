from enum import Enum

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DinoVitType(Enum):
    SMALL = 1
    BASE = 2
    LARGE = 3
    GIANT = 4


class DinoVit(nn.Module):
    def __init__(self, dino_vit_type: DinoVitType, dino_vit_checkpoint=None):
        super().__init__()

        if dino_vit_type == DinoVitType.SMALL:
            uri = "facebook/dinov2-small"
        elif dino_vit_type == DinoVitType.BASE:
            uri = "facebook/dinov2-base"
        elif dino_vit_type == DinoVitType.LARGE:
            uri = "facebook/dinov2-large"
        elif dino_vit_type == DinoVitType.GIANT:
            uri = "facebook/dinov2-giant"
        else:
            raise ValueError(f"Unsupported Dino ViT type: {dino_vit_type}")

        self.__model = AutoModel.from_pretrained(uri)
        self.__image_processor = AutoImageProcessor.from_pretrained(uri)

        if dino_vit_checkpoint:
            state_dict = torch.load(dino_vit_checkpoint)
            self.__model.load_state_dict(state_dict)

    def forward(self, x):
        return self.__model(x)

    def get_image_processor(self):
        return self.__image_processor
