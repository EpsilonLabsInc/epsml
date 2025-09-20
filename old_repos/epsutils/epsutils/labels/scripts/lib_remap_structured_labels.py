import copy
from typing import Any, Dict, Sequence

BodyPart = str
Label = str
LabelItem = Dict[str, Any]


def remap_structured_labels(
    structured_labels: Sequence[LabelItem],
    a_to_b_mapping_by_body_part: Dict[BodyPart, Dict[Label, Label]],
) -> Sequence[LabelItem]:

    updated_structured_labels = copy.deepcopy(structured_labels)

    # Process each label group
    for label_group in updated_structured_labels:
        body_part = label_group.get("body_part", "")
        if not body_part or body_part not in a_to_b_mapping_by_body_part:
            raise Exception(
                f"Body part {body_part} not found in {a_to_b_mapping_by_body_part}"
            )

        consolidated_labels_bp = a_to_b_mapping_by_body_part[body_part]

        # Process labels within this body part
        for label_item in label_group.get("labels", []):
            label_name = label_item.get("label", "")
            if not label_name:
                continue

            # Apply "No Findings" -> "No findings" fix
            if label_name == "No Findings":
                label_item["label"] = "No findings"
                label_name = "No findings"

            # Remap using the consolidated mapping
            if label_name in consolidated_labels_bp:
                mapped_label = consolidated_labels_bp[label_name]
                if mapped_label:  # Only update if mapping exists and is not empty
                    label_item["label"] = mapped_label
            else:
                raise Exception(f"Label {label_name} not found in {body_part} mapping")

        # Second pass: collapse duplicate consolidated labels
        labels_list = label_group.get("labels", [])
        if labels_list:
            # Group labels by name, keeping first occurrence and tracking confidence
            unique_labels = {}
            label_confidence_map = {}

            for label_item in labels_list:
                label_name = label_item.get("label", "")
                confidence = label_item.get("confidence", "Uncertain")

                if label_name not in unique_labels:
                    # Keep the first occurrence with all its fields
                    unique_labels[label_name] = label_item.copy()
                    label_confidence_map[label_name] = confidence
                else:
                    # Update confidence if any instance is "Certain"
                    if (
                        confidence == "Certain"
                        or label_confidence_map[label_name] == "Certain"
                    ):
                        label_confidence_map[label_name] = "Certain"

            # Update confidence in the unique labels and rebuild list
            for label_name, final_confidence in label_confidence_map.items():
                unique_labels[label_name]["confidence"] = final_confidence

            label_group["labels"] = list(unique_labels.values())

    return updated_structured_labels
