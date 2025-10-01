import torch
import torch.nn as nn

from intern_vit import MultiImageInternVit


class MultiImageInternVitClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 intern_vl_checkpoint_dir,
                 intern_vit_output_dim=1024,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                 hidden_dim=1024,
                 dropout_rate=0.2,
                 encoder_layer_split_number=44):
        super().__init__()

        print("WARNING: Because of BatchNorm1d that doesn't work on single element batches, MultiImageInternVitClassifier currently supports only batch sizes >= 2")

        # MultiImageInternVit model.
        self.multi_image_intern_vit = MultiImageInternVit(intern_vl_checkpoint_dir=intern_vl_checkpoint_dir, encoder_layer_split_number=encoder_layer_split_number)

        # Get image processor.
        self.__image_processor = self.multi_image_intern_vit.get_image_processor()

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(intern_vit_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Set the dtype of the classifier to match the dtype of the MultiImageInternVit.
        dtype = next(self.multi_image_intern_vit.parameters()).dtype
        self.classifier = self.classifier.to(dtype)

    def forward(self, images, **kwargs):
        if len(images) < 2:
            raise ValueError("Because of BatchNorm1d that doesn't work on single element batches, MultiImageInternVitClassifier currently supports only batch sizes >= 2")

        embeddings = self.multi_image_intern_vit(images)
        output = self.classifier(embeddings)

        return {"output": output, "embeddings": embeddings}

    def get_image_processor(self):
        return self.__image_processor

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_fusion_convolution(self):
        for param in self.multi_image_intern_vit.fusion_convolution.parameters():
            param.requires_grad = True
