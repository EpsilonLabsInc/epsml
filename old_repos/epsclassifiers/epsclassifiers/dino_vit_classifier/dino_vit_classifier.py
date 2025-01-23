import torch.nn as nn

from dino_vit import DinoVit, DinoVitType


class DinoVitClassifier(nn.Module):
    def __init__(self, num_classes, dino_vit_checkpoint=None, dino_vit_output_dim=1024, hidden_dim=1024, dropout_rate=0.2):
        super().__init__()

        print("WARNING: Because of BatchNorm1d that doesn't work on single element batches, DinoVitClassifier currently supports only batch sizes >= 2")

        # DinoViT model.
        self.dino_vit = DinoVit(dino_vit_type=DinoVitType.LARGE, dino_vit_checkpoint=dino_vit_checkpoint)

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(dino_vit_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Set the dtype of the classifier to match the dtype of the DinoViT.
        dtype = next(self.dino_vit.parameters()).dtype
        self.classifier = self.classifier.to(dtype)

    def forward(self, x):
        if x.shape[0] < 2:
            raise ValueError("Because of BatchNorm1d that doesn't work on single element batches, DinoVitClassifier currently supports only batch sizes >= 2")

        output = self.dino_vit(x)
        return self.classifier(output)

    def get_image_processor(self):
        return self.dino_vit.get_image_processor()
