import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output


def show_visualization_data(visualization_data, num_view_grid_columns, label_to_string_mapping=None):
    inputs = visualization_data["inputs"]

    if inputs.dtype == torch.bfloat16:
        inputs = inputs.to(torch.float32)

    if label_to_string_mapping:
        labels = [label_to_string_mapping[label.item()] for label in visualization_data["labels"]]
    else:
        labels = ["-" for label in visualization_data["labels"]]

    probabilities = [probability.item() for probability in visualization_data["probabilities"]] if "probabilities" in visualization_data else None

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
