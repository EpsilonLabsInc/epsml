from enum import Enum
import sys
import os

import torch
import torch.nn as nn
from types import SimpleNamespace
from torchvision.transforms import v2

from dinov3.models import vision_transformer


class DinoV3Type(Enum):
    SMALL = 1
    BASE = 2
    LARGE = 3
    GIANT = 4


class DinoV3(nn.Module):
    def __init__(self, dino_v3_type: DinoV3Type, dino_v3_checkpoint=None, img_size=1024, use_attention_pooling=False):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.dino_v3_type = dino_v3_type

        if dino_v3_type == DinoV3Type.SMALL:
            # ViT-S/14 configuration
            embed_dim = 384
            self.__model = vision_transformer.DinoVisionTransformer(
                img_size=img_size,
                patch_size=14,
                embed_dim=embed_dim,
                depth=12,
                num_heads=6,
                ffn_ratio=4.0,
                norm_layer="rmsnorm",
                ffn_layer="swiglu"
            )
            # Build torchvision-based eval transform (no Hugging Face processor)
            self.__image_processor = self.__build_torchvision_processor(img_size)

        elif dino_v3_type == DinoV3Type.BASE:
            # ViT-B/14 configuration
            embed_dim = 768
            self.__model = vision_transformer.DinoVisionTransformer(
                img_size=img_size,
                patch_size=14,
                embed_dim=embed_dim,
                depth=12,
                num_heads=12,
                ffn_ratio=4.0,
                norm_layer="rmsnorm",
                ffn_layer="swiglu"
            )
            self.__image_processor = self.__build_torchvision_processor(img_size)

        elif dino_v3_type == DinoV3Type.LARGE:
            embed_dim = 1024
            self.__model = vision_transformer.DinoVisionTransformer(
                img_size=img_size,
                patch_size=16,
                embed_dim=embed_dim,
                depth=24,
                num_heads=16,
                ffn_ratio=4.0,
                norm_layer="layernorm",
                ffn_layer="mlp",
                layerscale_init=1e-4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True
            )
            self.__image_processor = self.__build_torchvision_processor(img_size)

        elif dino_v3_type == DinoV3Type.GIANT:
            embed_dim = 4096
            self.__model = vision_transformer.DinoVisionTransformer(
                img_size=img_size,
                patch_size=16,
                embed_dim=embed_dim,
                depth=40,
                num_heads=32,
                ffn_ratio=3.0,
                qkv_bias=False,
                proj_bias=True,
                ffn_bias=True,
                layerscale_init=1e-5,
                norm_layer="layernorm",
                ffn_layer="swiglu",
                n_storage_tokens=4,
                untie_global_and_local_cls_norm=True,
            )
            self.__image_processor = self.__build_torchvision_processor(img_size)
        else:
            raise ValueError(f"Unsupported Dino V3 type: {dino_v3_type}")

        # torchvision processor already parameterized with target img_size

        if dino_v3_checkpoint:
            print(f"Loading DINOv3 checkpoint from {dino_v3_checkpoint}")
            checkpoint = torch.load(dino_v3_checkpoint, map_location='cpu')

            # Handle different checkpoint formats
            if "teacher" in checkpoint:
                state_dict = checkpoint["teacher"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            cleaned_state_dict = {}
            for key, value in state_dict.items():
                # Remove common prefixes
                if key.startswith("module."):
                    key = key[7:]
                if key.startswith("student."):
                    key = key[8:]
                if key.startswith("backbone."):
                    key = key[9:]
                cleaned_state_dict[key] = value

            missing, unexpected = self.__model.load_state_dict(cleaned_state_dict, strict=False)
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")

    def forward(self, x):
        features = self.__model.forward_features(x)
        # Always return CLS token by default
        return features["x_norm_clstoken"]

    def get_features(self, x):
        # Return full feature dictionary for attention pooling
        return self.__model.forward_features(x)

    # -------------------------
    # Internal helpers
    # -------------------------
    def __build_torchvision_processor(self, img_size: int):
        """Create a lightweight processor that mimics DINOv3 eval preprocessing using torchvision v2.

        Pipeline (per image):
          - ToImage() -> converts PIL/numpy to tensor in CHW, uint8 range
          - Resize(shorter_side -> img_size) preserving aspect ratio
          - CenterCrop(img_size)
          - ToDtype(float32, scale=True)
          - Normalize(ImageNet mean/std)

        Returns an object with a __call__(images=[PIL...], return_tensors="pt").pixel_values Tensor[N,3,H,W].
        """
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        # Resize: match Dinov3's make_classification_eval_transform default behavior: resize so shorter side = resize_size, then center crop
        # In our use-case we set both resize_size=crop_size=img_size
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(img_size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

        class TorchVisionProcessor:
            def __init__(self, tfm):
                self._tfm = tfm

            def __call__(self, images, return_tensors="pt"):
                if not isinstance(images, (list, tuple)):
                    images_list = [images]
                else:
                    images_list = images
                processed = [self._tfm(img) for img in images_list]
                pixel_values = torch.stack(processed) if len(processed) > 1 else processed[0].unsqueeze(0)
                return SimpleNamespace(pixel_values=pixel_values)

        return TorchVisionProcessor(transform)

    def get_image_processor(self):
        return self.__image_processor
