
def to_multi_hot_encoding(labels, all_labels):
    multi_hot_encoding = [0] * len(all_labels)
    for label in labels:
        if label in all_labels:
            index = all_labels.index(label)
            multi_hot_encoding[index] = 1

    return multi_hot_encoding


def from_multi_hot_encoding(multi_hot_encoding, all_labels):
    return [all_labels[i] for i, val in enumerate(multi_hot_encoding) if val == 1]
