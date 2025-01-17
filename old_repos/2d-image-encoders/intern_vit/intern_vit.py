import torch
import torch.nn as nn
from transformers import CLIPImageProcessor

from internvl.model.internvl_chat import InternVLChatModel


class InternVit(nn.Module):
    def __init__(self, intern_vl_checkpoint_dir):
        super().__init__()

        vlm_model = InternVLChatModel.from_pretrained(
            intern_vl_checkpoint_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True
        )

        self.__model = vlm_model.vision_model
        self.__image_processor = CLIPImageProcessor.from_pretrained("OpenGVLab/InternViT-300M-448px-V2_5")

    def forward(self, x):
        return self.__model(x)

    def get_image_processor(self):
        return self.__image_processor
