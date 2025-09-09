import torch
import torch.nn as nn

from dino_v3 import DinoV3, DinoV3Type


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
        
        # We need to get patch tokens, not CLS tokens for attention pooling
        # So we need to modify DinoV3 to return the full feature dict
        
        # Handle multi-image input
        if isinstance(x, list):
            # Variable number of images per study (like InternViT)
            from itertools import accumulate
            
            # Flatten all images into a single batch
            flat_tensor_list = [tensor for tensors in x for tensor in tensors]
            images_stacked = torch.stack(flat_tensor_list)
            
            # Get features from all images
            features = self.dino_v3.get_features(images_stacked)
            
            # Track group sizes for reconstruction
            group_sizes = [len(tensors) for tensors in x]
            ends = list(accumulate(group_sizes))
            starts = [0] + ends[:-1]
            
            if self.use_attention_pooling:
                # Get patch tokens (CLS already excluded)
                patch_tokens = features['x_norm_patchtokens']
                seq_len = patch_tokens.shape[1]  # Number of patches per image
                hidden_dim = patch_tokens.shape[2]
                max_images = max(group_sizes)
                
                # Create padded tensor for all groups
                padded_patch_tokens = torch.zeros(
                    len(group_sizes), max_images * seq_len, hidden_dim,
                    dtype=patch_tokens.dtype, device=patch_tokens.device
                )
                
                # Create padding mask (True = ignore this position)
                padding_mask = torch.ones(
                    len(group_sizes), max_images * seq_len,
                    dtype=torch.bool, device=patch_tokens.device
                )
                
                # Fill in actual patch tokens and update mask
                for i, (s, e, size) in enumerate(zip(starts, ends, group_sizes)):
                    actual_patches = size * seq_len
                    # Reshape patches from this group into a flat sequence
                    group_patches = patch_tokens[s:e].reshape(-1, hidden_dim)
                    padded_patch_tokens[i, :actual_patches] = group_patches
                    padding_mask[i, :actual_patches] = False
                
                # Apply attention pooling with padding mask
                output, _ = self.attentional_pooling(padded_patch_tokens, key_padding_mask=padding_mask)
            else:
                # Max pooling over CLS tokens for each group
                cls_tokens = features['x_norm_clstoken']
                output = torch.stack([cls_tokens[s:e].max(dim=0).values for s, e in zip(starts, ends)])
                
        elif len(x.shape) == 5:
            # Fixed number of images: (batch, num_images, C, H, W)
            batch, num_images, C, H, W = x.shape
            x_reshaped = x.view(batch * num_images, C, H, W)
            
            # Get features from vision transformer
            features = self.dino_v3.get_features(x_reshaped) if hasattr(self.dino_v3, 'get_features') else self.dino_v3(x_reshaped)
            
            if self.use_attention_pooling:
                # Like InternViT, we need to strip CLS tokens and use only patch tokens
                if isinstance(features, dict) and 'x_norm_patchtokens' in features:
                    # Get patch tokens (already excludes CLS token)
                    patch_tokens = features['x_norm_patchtokens']  # (batch*num_images, seq_len, hidden_dim)
                    seq_len = patch_tokens.shape[1]
                    hidden_dim = patch_tokens.shape[2]
                    
                    # Reshape to (batch, num_images*seq_len, hidden_dim) to pool over all patches from all images
                    patch_tokens_reshaped = patch_tokens.view(batch, num_images * seq_len, hidden_dim)
                    output, _ = self.attentional_pooling(patch_tokens_reshaped)
                else:
                    raise ValueError("Attention pooling requires patch tokens from vision transformer")
            else:
                # Without attention pooling, use CLS tokens and average
                cls_tokens = features if not isinstance(features, dict) else features.get('x_norm_clstoken', features)
                cls_tokens = cls_tokens.view(batch, num_images, -1)
                output = cls_tokens.mean(dim=1)
                
        else:
            # Single image per sample
            output = self.dino_v3(x)
        
        return self.classifier(output)
    
    def get_image_processor(self):
        return self.dino_v3.get_image_processor()
