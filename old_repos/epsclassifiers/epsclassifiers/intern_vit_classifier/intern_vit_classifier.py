import torch.nn as nn

from epsutils.training.tile_splitting_image_processor import TileSplittingImageProcessor
from intern_vit import InternVit


class InternVitClassifier(nn.Module):
    def __init__(self, num_classes, intern_vl_checkpoint_dir, intern_vit_output_dim=1024, hidden_dim=1024,
                 dropout_rate=0.2, use_tiles=False, num_tiles_x=None, num_tiles_y=None):
        super().__init__()

        print("WARNING: Because of BatchNorm1d that doesn't work on single element batches, InternVitClassifier currently supports only batch sizes >= 2")

        # InternViT model.
        self.intern_vit = InternVit(intern_vl_checkpoint_dir=intern_vl_checkpoint_dir)

        # Use tiling?
        if use_tiles:
            print(f"INFO: InternVitClassifier will be using {num_tiles_x}x{num_tiles_y} tile splitting")
            self.__image_processor = TileSplittingImageProcessor(
                image_processor=self.intern_vit.get_image_processor(), num_rows=num_tiles_y, num_cols=num_tiles_x)
            self.__intern_vit_output_dim = intern_vit_output_dim * self.__image_processor.get_num_tiles()
        else:
            print(f"INFO: InternVitClassifier will NOT be using tile splitting")
            self.__image_processor = self.intern_vit.get_image_processor()
            self.__intern_vit_output_dim = intern_vit_output_dim

        self.__hidden_dim = self.__intern_vit_output_dim
        self.__use_tiles = use_tiles

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(self.__intern_vit_output_dim, hidden_dim),
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

        if self.__use_tiles:
            # 5 dimensions indicate use of tiles: (batch_num, num_tiles, num_channels, img_height, img_width)
            assert len(x.shape) == 5
            batch, tiles, num_channels, height, width = x.shape
            x_reshaped = x.view(batch * tiles, num_channels, height, width)
            output = self.intern_vit(x_reshaped)
            reshaped_output = output.pooler_output.reshape(batch, -1)
            return self.classifier(reshaped_output)
        else:
            output = self.intern_vit(x)
            return self.classifier(output.pooler_output)

    def get_image_processor(self):
        return self.__image_processor
