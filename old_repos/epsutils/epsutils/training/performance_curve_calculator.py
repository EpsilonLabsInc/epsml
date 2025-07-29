from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


class PerformanceCurveType(Enum):
    ROC = 0
    PRC = 1


class PerformanceCurveCalculator:
    def __init__(self):
        pass

    def compute_curve(self, curve_type: PerformanceCurveType, y_true, y_prob):
        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)
        assert y_true_np.shape == y_prob_np.shape

        y_true_flat = y_true_np.flatten()
        y_prob_flat = y_prob_np.flatten()

        if curve_type == PerformanceCurveType.ROC:
            # Compute ROC.
            fpr, tpr, thresholds = roc_curve(y_true=y_true_flat, y_score=y_prob_flat)

            # Compute AUC.
            auc_value = auc(fpr, tpr)

            # Compute optimal threshold.
            diff = tpr - fpr
            optimal_idx = np.argmax(diff)
            optimal_threshold = thresholds[optimal_idx]

            curve = {
                "curve_type": curve_type,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": auc_value,
                "optimal_threshold": optimal_threshold
            }

        elif curve_type == PerformanceCurveType.PRC:
            # Compute PRC.
            precision, recall, thresholds = precision_recall_curve(y_true=y_true_flat, y_score=y_prob_flat)

            # Compute AUC.
            auc_value = auc(recall, precision)

            # Optimal threshold not computed for PRC.
            optimal_threshold = None

            curve = {
                "curve_type": curve_type,
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds,
                "auc": auc_value,
                "optimal_threshold": optimal_threshold
            }

        else:
            raise ValueError(f"Performance curve type {curve_type} not supported")

        return curve

    def compute_per_class_curves(self, curve_type: PerformanceCurveType, y_true, y_prob):
        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)
        assert y_true_np.shape == y_prob_np.shape

        num_classes = y_true_np.shape[1]
        curves = []

        for class_idx in range(num_classes):
            curve = self.compute_curve(curve_type=curve_type, y_true=y_true_np[:, class_idx], y_prob=y_prob_np[:, class_idx])
            curves.append(curve)

        return curves

    def create_plot(self, curves, titles=None, grid_shape=(3, 7), show_grid=False, x_axis_markers=None, y_axis_markers=None):
        curves = curves if isinstance(curves, list) else [curves]
        num_curves = len(curves)

        fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(20, 10))
        if isinstance(axes, np.ndarray):
            axes = axes.ravel() # Flatten axes for easier iteration.
        else:
            axes = np.array([axes]) # Convert single Axes object to array.

        # Plot curves.
        for i in range(num_curves):
            title = titles[i] if titles else None
            curve = curves[i]
            curve_type = curve["curve_type"]
            x_vals = curve["fpr"] if curve_type == PerformanceCurveType.ROC else curve["recall"]
            y_vals = curve["tpr"] if curve_type == PerformanceCurveType.ROC else curve["precision"]
            thresholds = curve["thresholds"]
            auc_value = curve["auc"]
            optimal_threshold = curve["optimal_threshold"]

            # Plot curve.
            axes[i].plot(x_vals, y_vals)

            # Track the bounding box of the previously plotted text.
            prev_bb = None

            # Plot thresholds.
            for index, threshold in enumerate(thresholds):
                t = f"{threshold:.2f}" if threshold != float("inf") else "inf"
                color = "blue" if threshold == optimal_threshold else "red"
                text = axes[i].text(x_vals[index], y_vals[index], t, fontsize=9, color=color, ha="center")

                text.figure.canvas.draw()
                bb = text.get_window_extent(renderer=text.figure.canvas.get_renderer())

                if prev_bb is not None and bb.overlaps(prev_bb):
                    text.remove()
                    continue

                prev_bb = bb

            # Plot threshold closest to 0.5.
            closest_idx = np.argmin(np.abs(thresholds - 0.5))
            axes[i].axvline(x=x_vals[closest_idx], color="green", linestyle="--", linewidth=1)
            axes[i].axhline(y=y_vals[closest_idx], color="green", linestyle="--", linewidth=1)

            # Set text.
            axes[i].set_xlabel("FPR" if curve_type == PerformanceCurveType.ROC else "Recall")
            axes[i].set_ylabel("TPR" if curve_type == PerformanceCurveType.ROC else "Precision")
            axes[i].set_title(f"{title} (AUC: {auc_value:.2f})")

            # Show grid.
            if show_grid:
                axes[i].grid(True, linestyle="--", linewidth=0.5)

            # Show markers.
            if x_axis_markers is not None:
                for marker in x_axis_markers:
                    axes[i].axvline(x=marker, color="orange", linestyle="--", linewidth=1)
            if y_axis_markers is not None:
                for marker in y_axis_markers:
                    axes[i].axhline(y=marker, color="orange", linestyle="--", linewidth=1)

        # Remove empty subplots (if any).
        for i in range(num_curves, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        return fig
