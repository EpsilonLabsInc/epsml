import torch
import torch.nn as nn

from dino_v3 import DinoV3, DinoV3Type, AttentionalPooling


class DinoV3Classifier(nn.Module):
    def __init__(self, num_classes, dino_v3_checkpoint=None, dino_v3_output_dim=4096, hidden_dim=1024, dropout_rate=0.2, dino_v3_type=DinoV3Type.GIANT, img_size=1024, use_attention_pooling=False):
        super().__init__()
        
        self.use_attention_pooling = use_attention_pooling
        self.dino_v3 = DinoV3(dino_v3_type=dino_v3_type, dino_v3_checkpoint=dino_v3_checkpoint, img_size=img_size, use_attention_pooling=False)
        
        if use_attention_pooling:
            self.attentional_pooling = AttentionalPooling(hidden_size=dino_v3_output_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(dino_v3_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Set the dtype of the classifier to match the dtype of the DinoV3
        dtype = next(self.dino_v3.parameters()).dtype
        self.classifier = self.classifier.to(dtype)
    
    def forward(self, x=None, images=None, **kwargs):
        # Handle both direct tensor input and dictionary input from training
        if x is None and images is not None:
            x = images
        elif x is None:
            raise ValueError("Either x or images must be provided")
        
        # Handle multi-image input with attention pooling
        if isinstance(x, list):
            # Variable number of images per study
            flat_tensor_list = []
            group_sizes = []
            for tensors in x:
                if len(tensors.shape) == 4:  # (num_images, C, H, W)
                    flat_tensor_list.extend(tensors)
                    group_sizes.append(len(tensors))
                else:  # Single image
                    flat_tensor_list.append(tensors)
                    group_sizes.append(1)
            
            images_stacked = torch.stack(flat_tensor_list)
            features = self.dino_v3(images_stacked)  # Get CLS tokens for all images
            
            if self.use_attention_pooling:
                # Pool across all images in each study
                import torch.nn.functional as F
                from itertools import accumulate
                
                ends = list(accumulate(group_sizes))
                starts = [0] + ends[:-1]
                
                pooled_features = []
                for s, e in zip(starts, ends):
                    study_features = features[s:e].unsqueeze(0)  # (1, num_images, hidden_dim)
                    pooled, _ = self.attentional_pooling(study_features)
                    pooled_features.append(pooled)
                
                output = torch.cat(pooled_features, dim=0)
            else:
                # Max pooling across images in each study
                ends = list(accumulate(group_sizes))
                starts = [0] + ends[:-1]
                output = torch.stack([features[s:e].max(dim=0).values for s, e in zip(starts, ends)])
                
        elif len(x.shape) == 5:
            # Fixed number of images: (batch, num_images, C, H, W)
            batch, num_images, C, H, W = x.shape
            x_reshaped = x.view(batch * num_images, C, H, W)
            features = self.dino_v3(x_reshaped)  # (batch*num_images, hidden_dim)
            features = features.view(batch, num_images, -1)  # (batch, num_images, hidden_dim)
            
            if self.use_attention_pooling:
                output, _ = self.attentional_pooling(features)
            else:
                output = features.mean(dim=1)  # Average pooling across images
                
        else:
            # Single image per sample
            output = self.dino_v3(x)
        
        return self.classifier(output)
    
    def get_image_processor(self):
        return self.dino_v3.get_image_processor()
