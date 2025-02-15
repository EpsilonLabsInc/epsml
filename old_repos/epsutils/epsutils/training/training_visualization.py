import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output


def show_visualization_data(visualization_data, num_view_grid_columns, label_to_string_mapping=None):
    # Extract visualization data.
    inputs = visualization_data["inputs"]
    assert(len(inputs) > 0)
    labels = visualization_data["labels"]
    assert(len(labels) > 0)
    probabilities = visualization_data["probabilities"] if "probabilities" in visualization_data else None

    # Numpy cannot handle bfloat16 type, so convert it to float32.
    if inputs.dtype == torch.bfloat16:
        inputs = inputs.to(torch.float32)

    # Apply label to string mapping.
    if label_to_string_mapping:
        labels = [label_to_string_mapping[label.item()] for label in labels]

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

        # Handle single and 3 channel images.
        if input.shape[0] == 1:
            input = input[0, :, :].cpu().numpy()
        elif input.shape[0] == 3:
            input = input.cpu().numpy().transpose(1, 2, 0)
        else:
            raise ValueError(f"Input image has unsupported number of channels {input.shape[0]}")

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
