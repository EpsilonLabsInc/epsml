from PIL import Image
import random

def augment_image(image,
                  min_rotation_in_degrees=-45,
                  max_rotation_in_degrees=45,
                  min_scaling=0.9,
                  max_scaling=1.1,
                  max_translation=0.15):
    """
    Applies data augmentation to a given PIL image, including random rotation, scaling, and translation.
    """

    width, height = image.size

    # Random rotation.
    angle = random.uniform(min_rotation_in_degrees, max_rotation_in_degrees)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Random scaling.
    scale = random.uniform(min_scaling, max_scaling)
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    image = image.resize((scaled_width, scaled_height), resample=Image.BICUBIC)

    # Random translation.
    max_trans_x = int(width * max_translation)
    max_trans_y = int(height * max_translation)
    trans_x = random.randint(-max_trans_x, max_trans_x)
    trans_y = random.randint(-max_trans_y, max_trans_y)
    translation_matrix = (1, 0, trans_x, 0, 1, trans_y)
    image = image.transform(image.size, Image.AFFINE, translation_matrix, resample=Image.BICUBIC)

    return image
