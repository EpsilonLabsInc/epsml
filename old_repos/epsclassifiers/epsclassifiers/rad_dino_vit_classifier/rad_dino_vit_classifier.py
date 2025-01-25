import torch.nn as nn

from rad_dino_vit import RadDinoVit


class RadDinoVitClassifier(nn.Module):
    def __init__(self, num_classes, rad_dino_vit_output_dim=768, hidden_dim=1024, dropout_rate=0.2):
        super().__init__()

        print("WARNING: Because of BatchNorm1d that doesn't work on single element batches, RadDinoVitClassifier currently supports only batch sizes >= 2")

        # DinoViT model.
        self.rad_dino_vit = RadDinoVit()

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(rad_dino_vit_output_dim, hidden_dim),
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
        dtype = next(self.rad_dino_vit.parameters()).dtype
        self.classifier = self.classifier.to(dtype)

    def forward(self, x):
        if x.shape[0] < 2:
            raise ValueError("Because of BatchNorm1d that doesn't work on single element batches, RadDinoVitClassifier currently supports only batch sizes >= 2")

        output = self.rad_dino_vit(x)
        return self.classifier(output.pooler_output)

    def get_image_processor(self):
        return self.rad_dino_vit.get_image_processor()
