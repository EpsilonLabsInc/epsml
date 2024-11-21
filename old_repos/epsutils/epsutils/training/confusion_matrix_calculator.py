import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ConfusionMatrixCalculator:
    def __init__(self):
        pass

    def compute_confusion_matrix(self, y_true, y_pred):
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        assert y_true_np.shape == y_pred_np.shape

        y_true_flat = y_true_np.flatten()
        y_pred_flat = y_pred_np.flatten()
        cm = confusion_matrix(y_true=y_true_flat, y_pred=y_pred_flat, labels=[0, 1])
        return cm

    def compute_per_class_confusion_matrices(self, y_true, y_pred):
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        assert y_true_np.shape == y_pred_np.shape

        num_classes = y_true_np.shape[1]

        conf_matrices = []
        for class_idx in range(num_classes):
            cm = confusion_matrix(y_true_np[:, class_idx], y_pred_np[:, class_idx], labels=[0, 1])
            conf_matrices.append(cm)

        return conf_matrices

    def create_plot(self, confusion_matrices, titles=None, grid_shape=(3, 7)):
        fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(20, 10))

        if isinstance(axes, np.ndarray):
            axes = axes.ravel() # Flatten axes for easier iteration.
        else:
            axes = np.array([axes]) # Convert single Axes object to array.

        num_matrices = len(confusion_matrices)

        for i in range(num_matrices):
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices[i], display_labels=["0", "1"])
            disp.plot(ax=axes[i], cmap=plt.cm.Blues, colorbar=False)
            title = f"{i}: {titles[i]}" if titles is not None and i < len(titles) else f"{i}"
            axes[i].set_title(title)

        # Remove empty subplots (if any).
        for i in range(num_matrices, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        return fig
