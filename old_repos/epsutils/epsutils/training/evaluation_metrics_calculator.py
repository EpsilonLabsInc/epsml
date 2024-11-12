import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


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

        self.__curr_precision = precision_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average="macro", zero_division=0.0)
        self.__curr_recall = recall_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average="macro", zero_division=0.0)
        self.__curr_f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average="macro", zero_division=0.0)
        self.__curr_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

        self.__sum_precision += self.__curr_precision
        self.__sum_recall += self.__curr_recall
        self.__sum_f1 += self.__curr_f1
        self.__sum_accuracy += self.__curr_accuracy

        self.__num_samples += 1

    def get_current_metric(self):
        return self.__curr_precision, self.__curr_recall, self.__curr_f1, self.__curr_accuracy

    def get_average_metrics(self):
        avg_precision = self.__sum_precision / self.__num_samples if self.__num_samples > 0 else 0.0
        avg_recall = self.__sum_recall / self.__num_samples if self.__num_samples > 0 else 0.0
        avg_f1 = self.__sum_f1 / self.__num_samples if self.__num_samples > 0 else 0.0
        avg_accuracy = self.__sum_accuracy / self.__num_samples if self.__num_samples > 0 else 0.0

        return avg_precision, avg_recall, avg_f1, avg_accuracy

    def get_accumulated_metrics(self):
        cumulative_precision = self.__sum_tp / (self.__sum_tp + self.__sum_fp) if (self.__sum_tp + self.__sum_fp) != 0.0 else 0.0
        cumulative_recall = self.__sum_tp / (self.__sum_tp + self.__sum_fn) if (self.__sum_tp + self.__sum_fn) != 0.0 else 0.0
        cumulative_f1 = 2 * (cumulative_precision * cumulative_recall) / (cumulative_precision + cumulative_recall) if (cumulative_precision + cumulative_recall) != 0.0 else 0.0
        cumulative_accuracy = (self.__sum_tp + self.__sum_tn) / (self.__sum_tp + self.__sum_tn + self.__sum_fp + self.__sum_fn) \
                              if (self.__sum_tp + self.__sum_tn + self.__sum_fp + self.__sum_fn) != 0.0 else 0.0

        return cumulative_precision, cumulative_recall, cumulative_f1, cumulative_accuracy

    def reset(self):
        self.__num_samples = 0

        self.__curr_tp = 0.0
        self.__curr_tn = 0.0
        self.__curr_fp = 0.0
        self.__curr_fn = 0.0

        self.__sum_tp += 0.0
        self.__sum_tn += 0.0
        self.__sum_fp += 0.0
        self.__sum_fn += 0.0

        self.__curr_precision = 0.0
        self.__curr_recall = 0.0
        self.__curr_f1 = 0.0
        self.__curr_accuracy = 0.0

        self.__sum_precision = 0.0
        self.__sum_recall = 0.0
        self.__sum_f1 = 0.0
        self.__sum_accuracy = 0.0
