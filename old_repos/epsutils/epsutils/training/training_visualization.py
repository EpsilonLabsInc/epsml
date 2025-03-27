import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output


def show_visualization_data(visualization_data, num_view_grid_columns, label_to_string_mapping=None):
    # Extract visualization data.
    inputs = visualization_data["inputs"]
    assert(len(inputs) > 0)
    labels = visualization_data["labels"].cpu()
    assert(len(labels) > 0)
    probabilities = visualization_data["probabilities"] if "probabilities" in visualization_data else None

    # Numpy cannot handle bfloat16 type, so convert it to float32.
    if inputs.dtype == torch.bfloat16:
        inputs = inputs.to(torch.float32)

    # Apply label to string mapping.
    labels = [label_to_string_mapping[label.item()] if label_to_string_mapping else label.item() for label in labels]

    # Display probabilities if they are available and only if they are scalars.
    if probabilities is not None and probabilities[0].numel() == 1:
        probabilities = [probability.item() for probability in probabilities]
    elif probabilities is not None and probabilities[0].numel() > 1:
        probabilities = None

    NUM_IMAGES = inputs.size(0)
    NUM_ROWS = math.ceil(NUM_IMAGES / num_view_grid_columns)

    clear_output(wait=True)
    plt.figure(figsize=(num_view_grid_columns * 2, NUM_ROWS * 2))

    for i in range(NUM_IMAGES):
        plt.subplot(NUM_ROWS, num_view_grid_columns, i + 1)

        # Extract image from the batch.
        input = inputs[i]

        # Only 3 element tensors (num_channels, height, width) and 4 element tensors (batch, num_channels, height, widht) are supported.
        if len(input.shape) not in (3, 4):
            raise ValueError(f"Input image has unsupported shape {input.shape}")

        input = input.cpu().numpy()

        # Concatenate batch horizontally into a single image.
        if len(input.shape) == 4:
            input = np.concatenate(input, axis=2)

        # Regroup elements: (num_channels, height, width) --> (height, width, num_channels)
        input = input.transpose(1, 2, 0)

        # Normalize image if necessary.
        if input.dtype != np.uint8:
            input_min = input.min(axis=(0, 1), keepdims=True)
            input_max = input.max(axis=(0, 1), keepdims=True)
            input = (input - input_min) / (input_max - input_min) * 255.0
            input = input.astype(np.uint8)

        plt.imshow(input, cmap="gray")

        title = f"Label: {labels[i]}"

        if probabilities is not None:
            title += f" ({probabilities[i]:.2f})"

        plt.title(title, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
