import torch
from sklearn.metrics import confusion_matrix


class EvaluationMetricsCalculator:
    def __init__(self):
        self.reset()

    def add(self, logits: torch.Tensor, labels: torch.Tensor):
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).int()
        y_pred = predictions.view(-1).cpu().numpy()
        y_true = labels.int().view(-1).cpu().numpy()
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        self.__curr_tp = tp
        self.__curr_tn = tn
        self.__curr_fp = fp
        self.__curr_fn = fn
        self.__sum_tp += tp
        self.__sum_tn += tn
        self.__sum_fp += fp
        self.__sum_fn += fn

    def compute_metrics(self):
        # Current metrics.
        curr_precision = self.__curr_tp / (self.__curr_tp + self.__curr_fp) if (self.__curr_tp + self.__curr_fp) != 0.0 else 0.0
        curr_recall = self.__curr_tp / (self.__curr_tp + self.__curr_fn) if (self.__curr_tp + self.__curr_fn) != 0.0 else 0.0
        curr_f1 = 2 * (curr_precision * curr_recall) / (curr_precision + curr_recall) if (curr_precision + curr_recall) != 0.0 else 0.0
        curr_accuracy = (self.__curr_tp + self.__curr_tn) / (self.__curr_tp + self.__curr_tn + self.__curr_fp + self.__curr_fn) \
                        if (self.__curr_tp + self.__curr_tn + self.__curr_fp + self.__curr_fn) != 0.0 else 0.0

        # Cumulative metrics.
        cumulative_precision = self.__sum_tp / (self.__sum_tp + self.__sum_fp) if (self.__sum_tp + self.__sum_fp) != 0.0 else 0.0
        cumulative_recall = self.__sum_tp / (self.__sum_tp + self.__sum_fn) if (self.__sum_tp + self.__sum_fn) != 0.0 else 0.0
        cumulative_f1 = 2 * (cumulative_precision * cumulative_recall) / (cumulative_precision + cumulative_recall) if (cumulative_precision + cumulative_recall) != 0.0 else 0.0
        cumulative_accuracy = (self.__sum_tp + self.__sum_tn) / (self.__sum_tp + self.__sum_tn + self.__sum_fp + self.__sum_fn) \
                              if (self.__sum_tp + self.__sum_tn + self.__sum_fp + self.__sum_fn) != 0.0 else 0.0

        return curr_precision, curr_recall, curr_f1, curr_accuracy, cumulative_precision, cumulative_recall, cumulative_f1, cumulative_accuracy

    def reset(self):
        self.__curr_tp = 0.0
        self.__curr_tn = 0.0
        self.__curr_fp = 0.0
        self.__curr_fn = 0.0
        self.__sum_tp = 0.0
        self.__sum_tn = 0.0
        self.__sum_fp = 0.0
        self.__sum_fn = 0.0
