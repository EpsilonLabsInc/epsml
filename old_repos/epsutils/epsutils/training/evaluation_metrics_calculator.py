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
        self.__tp += tp
        self.__tn += tn
        self.__fp += fp
        self.__fn += fn

    def compute_metrics(self):
        precision = self.__tp / (self.__tp + self.__fp) if (self.__tp + self.__fp) != 0.0 else 0.0
        recall = self.__tp / (self.__tp + self.__fn) if (self.__tp + self.__fn) != 0.0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0.0 else 0.0
        accuracy = (self.__tp + self.__tn) / (self.__tp + self.__tn + self.__fp + self.__fn) if (self.__tp + self.__tn + self.__fp + self.__fn) != 0.0 else 0.0
        return precision, recall, f1, accuracy

    def reset(self):
        self.__tp = 0.0
        self.__tn = 0.0
        self.__fp = 0.0
        self.__fn = 0.0
