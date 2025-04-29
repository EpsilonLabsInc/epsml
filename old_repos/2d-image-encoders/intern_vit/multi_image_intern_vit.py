import torch
import torch.nn as nn
from transformers import CLIPImageProcessor

from internvl.model.internvl_chat import InternVLChatModel


class MultiImageInternVit(nn.Module):
    def __init__(self, intern_vl_checkpoint_dir, encoder_layer_split_number=44):
        super().__init__()

        # Create VLM.
        vlm_model = InternVLChatModel.from_pretrained(
            intern_vl_checkpoint_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True
        )

        # Extract ViT.
        vit = vlm_model.vision_model

        # Make sure encoder layer split number is valid.
        num_encoder_layers = len(vit.encoder.layers)
        assert(encoder_layer_split_number > 0 and encoder_layer_split_number < num_encoder_layers)

        # Split the model.
        self.__embeddings = vit.embeddings
        self.__encoder_part_1 = nn.Sequential(*vit.encoder.layers[:encoder_layer_split_number])  # First half of encoder layers.
        self.__encoder_part_2 = nn.Sequential(*vit.encoder.layers[encoder_layer_split_number:])  # Second half of encoder layers.

        # Create fusion convolution layer.
        self.__num_features = self.__encoder_part_1[-1].mlp.fc2.out_features
        self.__fusion_convolution = nn.Conv2d(in_channels=3 * self.__num_features, out_channels=self.__num_features, kernel_size=(1, 1)).to(vlm_model.dtype)

        # Get image processor.
        self.__image_processor = CLIPImageProcessor.from_pretrained("OpenGVLab/InternViT-300M-448px-V2_5")

    def forward(self, samples):
        all_outputs = []

        for sample in samples:
            combined_features = []

            for image in sample:
                embeddings = self.__embeddings(image)
                features = self.__encoder_part_1(embeddings)  # Shape: [1, 1025, 3200]
                combined_features.append(features)

            combined_features = torch.cat(combined_features, dim=0)  # Shape: [number of images in sample, 1025, 3200]

            mean_feat = combined_features.mean(dim=0, keepdim=True)  # Shape: [1, 1025, 3200]
            var_feat = combined_features.var(dim=0, keepdim=True) if combined_features.size(0) > 1 else torch.zeros_like(mean_feat)  # Shape: [1, 1025, 3200]
            epsilon = 1e-6
            var_feat = var_feat + epsilon
            max_feat, _ = combined_features.max(dim=0, keepdim=True)  # Shape: [1, 1025, 3200]

            fused_features = torch.cat([mean_feat, var_feat, max_feat], dim=2)  # Shape: [1, 1025, 9600]
            fused_features = fused_features.unsqueeze(3)  # Shape: [1, 1025, 9600, 1]
            fused_features = fused_features.permute(0, 2, 1, 3)  # Shape: [1, 9600, 1025, 1]

            fused_features = self.__fusion_convolution(fused_features)  # Shape: [1, 3200, 1025, 1]

            fused_features = fused_features.squeeze(3)  # Shape: [1, 3200, 1025]
            fused_features = fused_features.permute(0, 2, 1)  # Shape: [1, 1025, 3200]

            output = self.__encoder_part_2(fused_features)
            output = output[:, 0, :]  # Use the embedding corresponding to the first token (CLS).

            all_outputs.append(output)

        all_outputs = torch.cat(all_outputs, dim=0)

        return all_outputs

    def get_image_processor(self):
        return self.__image_processor
