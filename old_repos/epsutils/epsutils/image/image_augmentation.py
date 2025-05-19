from PIL import Image
import random


def augment_image(image, rotation_in_degrees, scaling, translation):
    """
    Applies data augmentation to a given PIL image using rotation, scaling, and translation.

    Parameters:
        image (PIL.Image.Image): The input image to be augmented.
        rotation_in_degrees (float): The angle in degrees by which the image is rotated.
        scaling (float): The scaling factor applied to the image. Values > 1.0 enlarge the image,
                         while values < 1.0 shrink it.
        translation (float): The fraction of the image's width and height by which it is translated.

    Returns:
        PIL.Image.Image: The augmented image after applying rotation scaling, and translation.
    """

    width, height = image.size

    # Rotation.
    image = image.rotate(rotation_in_degrees, resample=Image.BICUBIC, expand=True)

    # Scaling.
    scaled_width = int(width * scaling)
    scaled_height = int(height * scaling)
    image = image.resize((scaled_width, scaled_height), resample=Image.BICUBIC)

    # Translation.
    trans_x = int(width * translation)
    trans_y = int(height * translation)
    translation_matrix = (1, 0, trans_x, 0, 1, trans_y)
    image = image.transform(image.size, Image.AFFINE, translation_matrix, resample=Image.BICUBIC)

    return image


def random_augment_image(image,
                         min_rotation_in_degrees=-45,
                         max_rotation_in_degrees=45,
                         min_scaling=0.9,
                         max_scaling=1.1,
                         min_translation=-0.15,
                         max_translation=0.15,
                         seed=None):
    """
    Applies random data augmentation to a given PIL image using random rotation, random scaling and random translation.
    """

    if seed is not None:
        random.seed(seed)

    width, height = image.size

    # Random rotation.
    rotation_in_degrees = random.uniform(min_rotation_in_degrees, max_rotation_in_degrees)
    image = image.rotate(rotation_in_degrees, resample=Image.BICUBIC, expand=True)

    # Random scaling.
    scaling = random.uniform(min_scaling, max_scaling)
    scaled_width = int(width * scaling)
    scaled_height = int(height * scaling)
    image = image.resize((scaled_width, scaled_height), resample=Image.BICUBIC)

    # Random translation.
    translation = random.uniform(min_translation, max_translation)
    trans_x = int(width * translation)
    trans_y = int(height * translation)
    translation_matrix = (1, 0, trans_x, 0, 1, trans_y)
    image = image.transform(image.size, Image.AFFINE, translation_matrix, resample=Image.BICUBIC)

    return image


def generate_augmentation_parameters(num_images,
                                     min_rotation_in_degrees=-45,
                                     max_rotation_in_degrees=45,
                                     min_scaling=0.9,
                                     max_scaling=1.1,
                                     min_translation=-0.15,
                                     max_translation=0.15,
                                     seed=None):
    """
    Generates a list of augmentation parameters for the given number of images.
    """

    if seed is not None:
        random.seed(seed)

    augmentation_parameters = []

    for i in range(num_images):
        # Random rotation.
        rotation_in_degrees = random.uniform(min_rotation_in_degrees, max_rotation_in_degrees)

        # Random scaling.
        scaling = random.uniform(min_scaling, max_scaling)

        # Random translation.
        translation = random.uniform(min_translation, max_translation)

        augmentation_parameters.append(
            {
                "rotation_in_degrees": rotation_in_degrees,
                "scaling": scaling,
                "translation": translation
            }
        )

    return augmentation_parameters
