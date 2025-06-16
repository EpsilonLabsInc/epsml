
def to_multi_hot_encoding(labels, all_labels):
    multi_hot_encoding = [0] * len(all_labels)
    for label in labels:
        if label in all_labels:
            index = all_labels.index(label)
            multi_hot_encoding[index] = 1

    return multi_hot_encoding


def from_multi_hot_encoding(multi_hot_encoding, all_labels):
    return [all_labels[i] for i, val in enumerate(multi_hot_encoding) if val == 1]


def parse_structured_labels(structured_labels, treat_uncertain_as_positive=True):
    parsed_labels = {}
    for item in structured_labels:
        # Make sure body part is not repeated.
        assert item["body_part"] not in parsed_labels

        # Make sure no label is repeated to avoid scenarios where one occurence of the label would be certain and the other one uncertain.
        label_names = [label["label"] for label in item["labels"]]
        assert len(label_names) == len(set(label_names))

        parsed_labels[item["body_part"]] = [label["label"] for label in item["labels"] if treat_uncertain_as_positive or label["confidence"].strip().lower() == "certain"]

    return parsed_labels
