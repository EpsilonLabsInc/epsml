import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class EvaluationMetricsCalculator:
    def __init__(self):
        self.reset()

    def add(self, logits: torch.Tensor, labels: torch.Tensor):
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).int()
        y_pred = predictions.view(-1).cpu().numpy()
        y_true = labels.int().view(-1).cpu().numpy()

        self.__curr_precision = precision_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average="macro")
        self.__curr_recall = recall_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average="macro")
        self.__curr_f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average="macro")
        self.__curr_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

        self.__sum_precision += self.__curr_precision
        self.__sum_recall += self.__curr_recall
        self.__sum_f1 += self.__curr_f1
        self.__sum_accuracy += self.__curr_accuracy

        self.__num_samples += 1

    def compute_metrics(self):
        avg_precision = self.__sum_precision / self.__num_samples if self.__num_samples > 0 else 0.0
        avg_recall = self.__sum_recall / self.__num_samples if self.__num_samples > 0 else 0.0
        avg_f1 = self.__sum_f1 / self.__num_samples if self.__num_samples > 0 else 0.0
        avg_accuracy = self.__sum_accuracy / self.__num_samples if self.__num_samples > 0 else 0.0

        return self.__curr_precision, self.__curr_recall, self.__curr_f1, self.__curr_accuracy, avg_precision, avg_recall, avg_f1, avg_accuracy

    def reset(self):
        self.__num_samples = 0
        self.__curr_precision = 0.0
        self.__curr_recall = 0.0
        self.__curr_f1 = 0.0
        self.__curr_accuracy = 0.0
        self.__sum_precision = 0.0
        self.__sum_recall = 0.0
        self.__sum_f1 = 0.0
        self.__sum_accuracy = 0.0
