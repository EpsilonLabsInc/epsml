import math

import matplotlib.pyplot as plt
from IPython.display import clear_output


def show_visualization_data(visualization_data, num_view_grid_columns, label_to_string_mapping=None):
    inputs = visualization_data["inputs"]
    labels = [label_to_string_mapping[label.item()] for label in visualization_data["labels"]] if label_to_string_mapping else "n/a"
    probabilities = [probability.item() for probability in visualization_data["probabilities"]] if "probabilities" in visualization_data else None

    NUM_IMAGES = inputs.size(0)
    NUM_ROWS = math.ceil(NUM_IMAGES / num_view_grid_columns)

    clear_output(wait=True)
    plt.figure(figsize=(num_view_grid_columns * 2, NUM_ROWS * 2))

    for i in range(NUM_IMAGES):
        plt.subplot(NUM_ROWS, num_view_grid_columns, i + 1)
        plt.imshow(inputs[i, 0, :, :].cpu().numpy(), cmap="gray")

        title = f"Label: {labels[i]}"

        if probabilities is not None:
            title += f" ({probabilities[i]:.2f})"

        plt.title(title, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
