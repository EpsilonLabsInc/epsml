import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class RadDinoVit(nn.Module):
    def __init__(self):
        super().__init__()

        self.__repo = "microsoft/rad-dino"
        self.__model = AutoModel.from_pretrained(self.__repo)
        self.__image_processor = AutoImageProcessor.from_pretrained(self.__repo)

    def forward(self, x):
        return self.__model(x)

    def get_image_processor(self):
        return self.__image_processor
