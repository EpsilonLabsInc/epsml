from enum import Enum
import sys
import os

import torch
import torch.nn as nn
from transformers import AutoImageProcessor

# Add dinov3 to path
sys.path.insert(0, '/home/yan/work/dinov3')
from dinov3.models import vision_transformer


class AttentionalPooling(nn.Module):
    """Attention pooling module that uses multi-head attention with a learnable query."""
    def __init__(self, hidden_size, dims_per_head=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.dims_per_head = dims_per_head

        # Learnable parameters (query vector + MHA block)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=hidden_size // dims_per_head, batch_first=True
        )

    def forward(self, last_hidden_state, key_padding_mask=None):
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        # key_padding_mask: (batch_size, sequence_length) - True for positions to ignore
        batch_size = last_hidden_state.size(0)

        # Expand query to match batch size
        query = self.query.expand(batch_size, 1, self.hidden_size)

        # Apply multi-head attention and squeeze to get (batch_size, hidden_size)
        pooled_output, attention_weights = self.multihead_attention(
            query=query, key=last_hidden_state, value=last_hidden_state, 
            key_padding_mask=key_padding_mask
        )
        return pooled_output.squeeze(1), attention_weights


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
            # Use DINOv2 image processor as fallback
            self.__image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            
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
            self.__image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            
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
            self.__image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
            
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
            self.__image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
        else:
            raise ValueError(f"Unsupported Dino V3 type: {dino_v3_type}")
        
        self.__image_processor.size["shortest_edge"] = img_size
        self.__image_processor.crop_size["height"] = img_size
        self.__image_processor.crop_size["width"] = img_size
        
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
        # Always return CLS token
        # Attention pooling across multiple images is handled in DinoV3Classifier
        return features["x_norm_clstoken"]
    
    def get_image_processor(self):
        return self.__image_processor
