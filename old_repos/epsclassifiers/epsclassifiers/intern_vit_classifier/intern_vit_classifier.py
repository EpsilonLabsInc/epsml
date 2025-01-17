import torch.nn as nn

from intern_vit import InternVit


class InternVitClassifier(nn.Module):
    def __init__(self, num_classes, intern_vl_checkpoint_dir, intern_vit_output_dim=1024, hidden_dim=1024, dropout_rate=0.2):
        super().__init__()

        # InternViT model.
        self.__intern_vit = InternVit(intern_vl_checkpoint_dir=intern_vl_checkpoint_dir)

        # Classifier head.
        self.__classifier = nn.Sequential(
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
        dtype = next(self.__intern_vit.parameters()).dtype
        self.__classifier = self.__classifier.to(dtype)

    def forward(self, x):
        output = self.__intern_vit(x)
        return self.__classifier(output.pooler_output)

    def get_image_processor(self):
        return self.__intern_vit.get_image_processor()
