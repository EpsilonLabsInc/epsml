import torch
import torch.nn as nn
from transformers import DistilBertModel

from epsutils.training.tile_splitting_image_processor import TileSplittingImageProcessor
from intern_vit import InternVit


class InternVitClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 intern_vl_checkpoint_dir,
                 intern_vit_output_dim=1024,  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
                 hidden_dim=1024,
                 dropout_rate=0.2,
                 multi_image_input=False,
                 num_multi_images=None,
                 use_text_encodings=False,
                 use_tiles=False,
                 num_tiles_x=None,
                 num_tiles_y=None):
        super().__init__()

        print("WARNING: Because of BatchNorm1d that doesn't work on single element batches, InternVitClassifier currently supports only batch sizes >= 2")

        # InternViT model.
        self.intern_vit = InternVit(intern_vl_checkpoint_dir=intern_vl_checkpoint_dir)

        if multi_image_input:
            print(f"INFO: InternVitClassifier will be using multi image input of size {num_multi_images}")
            self.__image_processor = self.intern_vit.get_image_processor()
            output_dim = intern_vit_output_dim * num_multi_images
        elif use_tiles:
            print(f"INFO: InternVitClassifier will be using {num_tiles_x}x{num_tiles_y} tile splitting")
            self.__image_processor = TileSplittingImageProcessor(
                image_processor=self.intern_vit.get_image_processor(), num_rows=num_tiles_y, num_cols=num_tiles_x)
            output_dim = intern_vit_output_dim * self.__image_processor.get_num_tiles()
        else:
            print(f"INFO: InternVitClassifier will NOT be using multi image input and will NOT be using tile splitting")
            self.__image_processor = self.intern_vit.get_image_processor()
            output_dim = intern_vit_output_dim

        # Text embeddings generator.
        if use_text_encodings:
            self.__text_embeddings_generator = DistilBertModel.from_pretrained("distilbert-base-uncased")
            hidden_size = self.__text_embeddings_generator.config.hidden_size  # Typically 768.
            output_dim += hidden_size

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
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

        self.__multi_image_input = multi_image_input
        self.__use_tiles = use_tiles
        self.__use_text_encodings = use_text_encodings

    def forward(self, images, text_encodings=None, **kwargs):
        if images.shape[0] < 2:
            raise ValueError("Because of BatchNorm1d that doesn't work on single element batches, InternVitClassifier currently supports only batch sizes >= 2")

        if self.__multi_image_input or self.__use_tiles:
            # 5 dimensions indicate use of multi image input or tiles: (batch_num, num_tiles, num_channels, img_height, img_width)
            assert len(images.shape) == 5
            batch, group, num_channels, height, width = images.shape
            images_reshaped = images.view(batch * group, num_channels, height, width)
            output = self.intern_vit(images_reshaped)
            embeddings = output.pooler_output.reshape(batch, -1)
            last_hidden_state = output.last_hidden_state.reshape(batch, -1)
        else:
            output = self.intern_vit(images)
            embeddings = output.pooler_output
            last_hidden_state = output.last_hidden_state

        if self.__use_text_encodings:
            text_output = self.__text_embeddings_generator(**text_encodings)
            # DistilBERT's last hidden state or pooler output can be used.
            # For simplicity, let's use the [CLS] token embedding from last hidden state.
            # DistilBERT doesn't have a pooler, so we often use the first token or mean pool.
            # We'll use the first token from last_hidden_state.
            text_embeddings = text_output.last_hidden_state[:, 0, :]
            text_embeddings = text_embeddings.to(embeddings.dtype)
            embeddings = torch.cat([text_embeddings, embeddings], dim=1)

        output = self.classifier(embeddings)

        return {"output": output, "embeddings": embeddings, "last_hidden_state": last_hidden_state}

    def get_image_processor(self):
        return self.__image_processor

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_intern_vit(self, num_last_layers_to_unfreeze=None):
        if num_last_layers_to_unfreeze is None:
            num_last_layers_to_unfreeze = len(self.intern_vit._InternVit__model.encoder.layers)

        for layer in self.intern_vit._InternVit__model.encoder.layers[-num_last_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
