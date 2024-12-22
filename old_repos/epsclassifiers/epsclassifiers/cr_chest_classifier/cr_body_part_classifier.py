import torchxrayvision as xrv


class CrBodyPartClassifier:
    def __init__(num_classes=1):
        self.__model = xrv.models.DenseNet(num_classes=num_classes)

    def model(self):
        return self.__model
