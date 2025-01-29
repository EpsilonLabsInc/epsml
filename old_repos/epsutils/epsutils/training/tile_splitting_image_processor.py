import torch


class TileSplittingImageProcessor:
    def __init__(self, image_processor):
        self.__image_processor = image_processor
        assert self.__image_processor.crop_size["width"] == self.__image_processor.crop_size["height"]
        self.__output_image_size = self.__image_processor.crop_size["width"]

    def __call__(self, images, return_tensors="pt"):
        return self.process_images(images, return_tensors)

    def get_tiles(self, image):
        width, height = image.size
        mid_width, mid_height = width // 2, height // 2
        tiles = [
            image,
            image.crop((0, 0, mid_width, mid_height)),
            image.crop((mid_width, 0, width, mid_height)),
            image.crop((0, mid_height, mid_width, height)),
            image.crop((mid_width, mid_height, width, height))
        ]
        return tiles

    def process_images(self, images, return_tensors="pt"):
        processed_images = []

        for image in images:
            tiles = self.get_tiles(image)
            processed_tiles = []

            for tile in tiles:
                processed_tile = self.__image_processor(images=tile, return_tensors=return_tensors).pixel_values
                processed_tiles.append(processed_tile)

            stacked_tiles = torch.cat(processed_tiles, dim=0)
            processed_images.append(stacked_tiles)

        return torch.stack(processed_images, dim=0)
