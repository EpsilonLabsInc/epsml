import torch


class TileSplittingImageProcessor:
    def __init__(self, image_processor, num_rows=3, num_cols=3):
        self.__image_processor = image_processor
        assert self.__image_processor.crop_size["width"] == self.__image_processor.crop_size["height"]
        self.__output_image_size = self.__image_processor.crop_size["width"]

        self.__num_rows = num_rows
        self.__num_cols = num_cols

    def get_num_tiles(self):
        return self.__num_rows * self.__num_cols + 1  # +1 for extra thumbnail.

    def __call__(self, images, return_tensors="pt"):
        return self.process_images(images, return_tensors)

    def __get_tiles(self, image):
        width, height = image.size
        tile_width, tile_height = width // self.__num_cols, height // self.__num_rows

        tiles = [
            image.crop((i * tile_width, j * tile_height, (i + 1) * tile_width, (j + 1) * tile_height))
            for j in range(self.__num_rows)
            for i in range(self.__num_cols)
        ]

        tiles.append(image)  # Add original image which will serve as the thumbnail.

        return tiles

    def process_images(self, images, return_tensors="pt"):
        processed_images = []

        for image in images:
            tiles = self.__get_tiles(image)
            processed_tiles = []

            for tile in tiles:
                processed_tile = self.__image_processor(images=tile, return_tensors=return_tensors).pixel_values
                processed_tiles.append(processed_tile)

            stacked_tiles = torch.cat(processed_tiles, dim=0)
            processed_images.append(stacked_tiles)

        return torch.stack(processed_images, dim=0)
