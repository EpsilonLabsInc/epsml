import matplotlib.pyplot as plt


def generate_histogram(data, title=None, x_label=None, y_label=None, x_labels_rotation_angle=None, show_x_labels_with_values=False):
    if show_x_labels_with_values:
        data = {f"{key} ({value:,})": value for key, value in data.items()}

    categories = list(data.keys())
    values = list(data.values())

    fig, axes = plt.subplots()
    axes.bar(categories, values, color="skyblue")
    axes.set_xticks(categories)
    axes.set_xticklabels(categories, rotation=x_labels_rotation_angle)
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(True)
    fig.tight_layout()

    return fig, axes
