import matplotlib.pyplot as plt
import numpy as np


class ScoresDistributionGenerator:
    def __init__(self):
        pass

    def create_histogram(self, scores):
        assert scores.shape[1] == 1
        hist, bin_edges = np.histogram(scores.numpy(), bins=10, range=(0, 1))
        return hist, bin_edges

    def create_plot(self, scores, title=None):
        hist, bin_edges = self.create_histogram(scores=scores)
        bin_width = bin_edges[1] - bin_edges[0]

        fig, axes = plt.subplots()
        axes.bar(bin_edges[:-1] + bin_width / 2, hist, width=bin_width, edgecolor="black", align="center")
        axes.set_xticks(bin_edges)
        axes.set_title(title)
        axes.set_xlabel("Score")
        axes.set_ylabel("Count")
        axes.grid(True)
        fig.tight_layout()

        return fig
