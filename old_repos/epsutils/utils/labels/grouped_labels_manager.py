import logging
from typing import List, Dict


class GroupedLabelsManager():
    """
    A helper class for handling grouped labels.

    Here's an example of grouped labels.

    grouped_labels = {
        "Trauma": [
            "Fractures of left lamina papyracea",
            "Cardiac tamponade",
            "Postoperative changes in hips"
        ],
        "Bleeding": [
            "Recent bleeding focus",
            "Parenchymal hyperdensity concerning for hemorrhage",
            "Increasing hemorrhage in left subdural space",
            "Right thalamic intraparenchymal hematoma",
            "Right parietal lobe involvement"
        ],
        "Inflammation": [
            "Frontoethmoidal mucosal thickening",
            "Enlarged, edematous pancreas",
            "Thickening of soft tissues in pylorus",
            "Fluid collection in right pelvis"
        ]
    }

    In the example above, "Trauma", "Bleeding" and "Inflammation" are groups, whereas "Fractures of left lamina papyracea",
    "Recent bleeding focus", "Frontoethmoidal mucosal thickening" etc. are labels. "Fractures of left lamina papyracea" label
    is grouped into "Trauma" group, "Recent bleeding focus" into "Bleeding" group and "Frontoethmoidal mucosal thickening"
    intp "Inflammation" group.
    """

    # TODO: Set allow_duplicate_labels to False once Kedar resolves the duplicates issue.
    def __init__(self, grouped_labels, allow_duplicate_labels=True):
        self.__grouped_labels = grouped_labels
        self.__allow_duplicate_labels = allow_duplicate_labels

        self.__create_groups()
        self.__create_group_mappings()
        self.__create_labels_to_groups_mapping()

    def get_groups(self):
        """
        Returns label groups.
        """
        return self.__groups

    def get_ids_to_groups(self):
        """
        Returns IDs to groups mapping.
        """
        return self.__ids_to_groups

    def get_groups_to_ids(self):
        """
        Returns groups to IDs mapping.
        """
        return self.__groups_to_ids

    def to_encoded_list(self, labels: List[str]):
        """
        Takes a list of labels, finds corresponding groups and generates a multi-hot encoded list of group appearances.

        For example, given grouped_labels = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"], "C": ["c1", "c2", "c3"] and
        labels = ["a2", "a3", "b1"], the method returns [1, 1, 0].
        """

        upper_labels = [label.upper() for label in labels]
        encoded_list = [0] * len(self.__groups)

        for label in upper_labels:
            if label not in self.__labels_to_groups:
                logging.warning(f"Label '{label}' does not have a corresponding label group")
                continue

            group = self.__labels_to_groups[label]
            if group not in self.__groups:
                logging.warning(f"Group '{group}' not in the list of groups")
                continue

            index = self.__groups.index(group)
            encoded_list[index] = 1

        return encoded_list

    def to_encoded_string(self, labels: List[str]):
        """
        Takes a list of labels, finds corresponding groups and generates a multi-hot encoded string of group appearances.

        For example, given grouped_labels = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"], "C": ["c1", "c2", "c3"] and
        labels = ["a2", "a3", "b1"], the method returns "110".
        """

        encoded_list = self.to_encoded_list(labels)
        encoded_string = ''.join(str(item) for item in encoded_list)
        return encoded_string

    def grouped_sample_labels_to_encoded_list(self, grouped_sample_labels: Dict[str, List[str]]):
        """
        Takes a dict of grouped sample labels, finds corresponding groups and generates a multi-hot encoded list of group appearances.

        For example, given grouped_labels = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"], "C": ["c1", "c2", "c3"] and
        grouped_sample_labels = {"A": ["a2"], "C": ["c2", "c3"] }, the method returns [1, 0, 1].
        """

        encoded_list = [0] * len(self.__groups)

        for group, labels in grouped_sample_labels.items():
            if len(labels) == 0:
                continue

            group = group.upper()

            if group not in self.__groups:
                raise ValueError(f"Group '{group}' not in the list of groups")

            id = self.__groups_to_ids[group]
            encoded_list[id] = 1

        return encoded_list

    def grouped_sample_labels_to_encoded_string(self, grouped_sample_labels: Dict[str, List[str]]):
        """
        Takes a dict of grouped sample labels, finds corresponding groups and generates a multi-hot encoded string of group appearances.

        For example, given grouped_labels = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"], "C": ["c1", "c2", "c3"] and
        grouped_sample_labels = {"A": ["a2"], "C": ["c2", "c3"] }, the method returns "101".
        """

        encoded_list = self.grouped_sample_labels_to_encoded_list(grouped_sample_labels)
        encoded_string = ''.join(str(item) for item in encoded_list)
        return encoded_string

    def encoded_string_to_encoded_list(self, encoded_string: str):
        """
        Converts encoded string to encoded list. For example, "110" --> [1, 1, 0].
        """

        encoded_list = [int(char) for char in encoded_string]
        return encoded_list

    def from_encoded_list(self, encoded_list: List[int]):
        """
        Takes encoded list and returns a list of corresponding group names.

        For example, given grouped_labels = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"], "C": ["c1", "c2", "c3"] and
        encoded list [1, 1, 0] the method returns ["A", "B"].
        """

        if len(encoded_list) != len(self.__groups):
            return None

        groups = []
        for i, item in enumerate(encoded_list):
            if item != 1:
                continue

            groups.append(self.__groups[i])

        return groups

    def from_encoded_string(self, encoded_string: str):
        """
        Takes encoded string and returns a list of corresponding group names.

        For example, given grouped_labels = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"], "C": ["c1", "c2", "c3"] and
        encoded string "110" the method returns ["A", "B"].
        """

        encoded_list = self.encoded_string_to_encoded_list(encoded_string)
        return self.from_encoded_list(encoded_list)

    def __create_groups(self):
        self.__groups = []
        for key in self.__grouped_labels:
            self.__groups.append(key.upper())

        self.__groups.sort()

    def __create_group_mappings(self):
        self.__groups_to_ids = {group: i for i, group in enumerate(self.__groups)}
        self.__ids_to_groups = {i: group for i, group in enumerate(self.__groups)}

    def __create_labels_to_groups_mapping(self):
        self.__labels_to_groups = {}
        for group, labels in self.__grouped_labels.items():
            for label in labels:
                if not self.__allow_duplicate_labels:
                    if label.upper() in self.__labels_to_groups:
                        raise ValueError(f"Duplicate label '{label}'")
                self.__labels_to_groups[label.upper()] = group.upper()
