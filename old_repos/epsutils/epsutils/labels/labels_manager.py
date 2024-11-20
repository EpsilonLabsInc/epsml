class LabelsManager:
    def __init__(self, all_label_names: list[str]):
        self.__all_label_names = [label_name.upper() for label_name in all_label_names]

    def get_all_label_names(self):
        return self.__all_label_names

    def get_label_name_at(self, index):
        if index >= len(self.__all_label_names):
            raise ValueError("Index out of range")

        return self.__all_label_names[index]

    def to_multi_hot_vector(self, label_names: list[str]):
        upper_label_names = [label_name.upper() for label_name in label_names]
        multi_hot = [1 if item in upper_label_names else 0 for item in self.__all_label_names]
        return multi_hot

    def from_multi_hot_vector(self, multi_hot_vector: list[int]):
        if len(multi_hot_vector) != len(self.__all_label_names):
            raise ValueError(f"Multi hot vector length should be {len(self.__all_label_names)}")

        label_names = [self.__all_label_names[i] for i in range(len(multi_hot_vector)) if multi_hot_vector[i] == 1]
        return label_names
