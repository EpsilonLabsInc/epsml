import torch.nn as nn

from dino_v3 import DinoV3, DinoV3Type


class DinoV3Classifier(nn.Module):
    def __init__(self, num_classes, dino_v3_checkpoint=None, dino_v3_output_dim=4096, hidden_dim=1024, dropout_rate=0.2, dino_v3_type=DinoV3Type.GIANT):
        super().__init__()
        
        print("WARNING: Because of BatchNorm1d that doesn't work on single element batches, DinoV3Classifier currently supports only batch sizes >= 2")
        
        self.dino_v3 = DinoV3(dino_v3_type=dino_v3_type, dino_v3_checkpoint=dino_v3_checkpoint)
        
        self.classifier = nn.Sequential(
            nn.Linear(dino_v3_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
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
        
        if x.shape[0] < 2:
            raise ValueError("Because of BatchNorm1d that doesn't work on single element batches, DinoV3Classifier currently supports only batch sizes >= 2")
        
        output = self.dino_v3(x)
        return self.classifier(output)
    
    def get_image_processor(self):
        return self.dino_v3.get_image_processor()
