from enum import Enum

import torch
import torch.nn as nn
from transformers import AutoImageProcessor

from dinov2.models import vision_transformer as vit


class DinoVitType(Enum):
    SMALL = 1
    BASE = 2
    LARGE = 3
    GIANT = 4


class DinoVit(nn.Module):
    def __init__(self, dino_vit_type: DinoVitType, dino_vit_checkpoint=None, img_size=512, init_values=1.0e-05, block_chunks=4):
        super().__init__()

        if dino_vit_type == DinoVitType.SMALL:
            self.__model = vit.vit_small(img_size=img_size, init_values=init_values, block_chunks=block_chunks)
            self.__image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        elif dino_vit_type == DinoVitType.BASE:
            self.__model = vit.vit_base(img_size=img_size, init_values=init_values, block_chunks=block_chunks)
            self.__image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        elif dino_vit_type == DinoVitType.LARGE:
            self.__model = vit.vit_large(img_size=img_size, init_values=init_values, block_chunks=block_chunks)
            self.__image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        elif dino_vit_type == DinoVitType.GIANT:
            self.__model = vit.vit_giant2(img_size=img_size, init_values=init_values, block_chunks=block_chunks)
            self.__image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
        else:
            raise ValueError(f"Unsupported Dino ViT type: {dino_vit_type}")

        self.__image_processor.size["shortest_edge"] = img_size
        self.__image_processor.crop_size["height"] = img_size
        self.__image_processor.crop_size["width"] = img_size

        if dino_vit_checkpoint:
            checkpoint = torch.load(dino_vit_checkpoint)
            state_dict = checkpoint["model"]
            prefix = "student.backbone."
            student_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
            self.__model.load_state_dict(student_state_dict)

    def forward(self, x):
        return self.__model(x)

    def get_image_processor(self):
        return self.__image_processor
