import torch.nn as nn

from intern_vit import InternVit


class InternVitClassifier(nn.Module):
    def __init__(self, num_classes, intern_vl_checkpoint_dir, intern_vit_output_dim=1024, hidden_dim=1024, dropout_rate=0.2):
        super().__init__()

        print("WARNING: Because of BatchNorm1d that doesn't work on single element batches, InternVitClassifier currently supports only batch sizes >= 2")

        # InternViT model.
        self.intern_vit = InternVit(intern_vl_checkpoint_dir=intern_vl_checkpoint_dir)

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

        # Set the dtype of the classifier to match the dtype of the InternViT.
        dtype = next(self.intern_vit.parameters()).dtype
        self.classifier = self.classifier.to(dtype)

    def forward(self, x):
        if x.shape[0] < 2:
            raise ValueError("Because of BatchNorm1d that doesn't work on single element batches, InternVitClassifier currently supports only batch sizes >= 2")

        output = self.intern_vit(x)
        return self.classifier(output.pooler_output)

    def get_image_processor(self):
        return self.intern_vit.get_image_processor()
