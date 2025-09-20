import pytest
import copy
from epsutils.labels.scripts.lib_remap_structured_labels import remap_structured_labels


class TestRemapStructuredLabels:
    """Test suite for the remap_structured_labels function."""

    def test_basic_label_remapping(self):
        """Test basic label remapping functionality."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "Pneumonia", "confidence": "Certain"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "Pneumonia": "Airspace opacity"
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        assert len(result) == 1
        assert result[0]["body_part"] == "Chest"
        assert len(result[0]["labels"]) == 1
        assert result[0]["labels"][0]["label"] == "Airspace opacity"
        assert result[0]["labels"][0]["confidence"] == "Certain"

    def test_no_findings_fix(self):
        """Test that 'No Findings' is converted to 'No findings'."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "No Findings", "confidence": "Certain"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "No findings": "No findings"  # Identity mapping after fix
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        assert result[0]["labels"][0]["label"] == "No findings"

    def test_duplicate_label_consolidation_certain_wins(self):
        """Test that duplicate labels are consolidated with 'Certain' taking precedence."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "Pneumonia", "confidence": "Uncertain", "extra_field": "value1"},
                    {"label": "Infection", "confidence": "Certain", "extra_field": "value2"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "Pneumonia": "Airspace opacity",
                "Infection": "Airspace opacity"  # Both map to same consolidated label
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        assert len(result[0]["labels"]) == 1
        assert result[0]["labels"][0]["label"] == "Airspace opacity"
        assert result[0]["labels"][0]["confidence"] == "Certain"
        # Should preserve extra fields from first occurrence
        assert result[0]["labels"][0]["extra_field"] == "value1"

    def test_duplicate_label_consolidation_all_uncertain(self):
        """Test that when all duplicates are uncertain, result stays uncertain."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "Pneumonia", "confidence": "Uncertain"},
                    {"label": "Infection", "confidence": "Uncertain"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "Pneumonia": "Airspace opacity",
                "Infection": "Airspace opacity"
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        assert len(result[0]["labels"]) == 1
        assert result[0]["labels"][0]["confidence"] == "Uncertain"

    def test_multiple_body_parts(self):
        """Test remapping across multiple body parts."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "Pneumonia", "confidence": "Certain"}
                ]
            },
            {
                "body_part": "Abdomen",
                "labels": [
                    {"label": "Appendicitis", "confidence": "Uncertain"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "Pneumonia": "Airspace opacity"
            },
            "Abdomen": {
                "Appendicitis": "Inflammatory condition"
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        assert len(result) == 2
        assert result[0]["labels"][0]["label"] == "Airspace opacity"
        assert result[1]["labels"][0]["label"] == "Inflammatory condition"

    def test_preserve_extra_fields(self):
        """Test that extra fields in label items are preserved."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {
                        "label": "Pneumonia",
                        "confidence": "Certain",
                        "bbox": [10, 20, 30, 40],
                        "severity": "moderate",
                        "anatomical_location": "right_upper_lobe"
                    }
                ]
            }
        ]

        mapping = {
            "Chest": {
                "Pneumonia": "Airspace opacity"
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        label_item = result[0]["labels"][0]
        assert label_item["label"] == "Airspace opacity"
        assert label_item["confidence"] == "Certain"
        assert label_item["bbox"] == [10, 20, 30, 40]
        assert label_item["severity"] == "moderate"
        assert label_item["anatomical_location"] == "right_upper_lobe"

    def test_empty_input(self):
        """Test handling of empty input."""
        result = remap_structured_labels([], {})
        assert result == []

    def test_empty_labels_list(self):
        """Test handling of body part with empty labels list."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": []
            }
        ]

        mapping = {"Chest": {}}

        result = remap_structured_labels(structured_labels, mapping)

        assert len(result) == 1
        assert result[0]["labels"] == []

    def test_missing_body_part_raises_exception(self):
        """Test that missing body part in mapping raises exception."""
        structured_labels = [
            {
                "body_part": "UnknownBodyPart",
                "labels": [
                    {"label": "SomeLabel", "confidence": "Certain"}
                ]
            }
        ]

        mapping = {"Chest": {}}

        with pytest.raises(Exception) as exc_info:
            remap_structured_labels(structured_labels, mapping)

        assert "Body part UnknownBodyPart not found" in str(exc_info.value)

    def test_missing_label_in_mapping_raises_exception(self):
        """Test that missing label in body part mapping raises exception."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "UnknownLabel", "confidence": "Certain"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "KnownLabel": "Mapped Label"
            }
        }

        with pytest.raises(Exception) as exc_info:
            remap_structured_labels(structured_labels, mapping)

        assert "Label UnknownLabel not found in Chest mapping" in str(exc_info.value)

    def test_empty_mapped_label_skipped(self):
        """Test that empty/None mapped labels are skipped."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "ObsoleteLabel", "confidence": "Certain"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "ObsoleteLabel": ""  # Empty mapping means remove label
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        # Label should remain unchanged since mapping is empty
        assert result[0]["labels"][0]["label"] == "ObsoleteLabel"

    def test_deep_copy_behavior(self):
        """Test that original input is not modified (deep copy behavior)."""
        original_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "Pneumonia", "confidence": "Certain"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "Pneumonia": "Airspace opacity"
            }
        }

        # Keep a copy of original for comparison
        original_copy = copy.deepcopy(original_labels)

        result = remap_structured_labels(original_labels, mapping)

        # Original should be unchanged
        assert original_labels == original_copy
        # Result should be different
        assert result[0]["labels"][0]["label"] == "Airspace opacity"
        assert original_labels[0]["labels"][0]["label"] == "Pneumonia"

    def test_complex_scenario(self):
        """Test a complex scenario with multiple features combined."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "Pneumonia", "confidence": "Uncertain", "region": "upper"},
                    {"label": "Infection", "confidence": "Certain", "region": "lower"},
                    {"label": "No Findings", "confidence": "Uncertain", "region": "middle"},
                    {"label": "Normal", "confidence": "Certain", "region": "base"}
                ]
            }
        ]

        mapping = {
            "Chest": {
                "Pneumonia": "Airspace opacity",
                "Infection": "Airspace opacity",  # Duplicate consolidation
                "No findings": "No findings",     # After "No Findings" fix
                "Normal": "No findings"           # Another duplicate
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        # Should have 2 unique labels: "Airspace opacity" and "No findings"
        labels = result[0]["labels"]
        assert len(labels) == 2

        # Find each label and verify
        airspace_label = next(l for l in labels if l["label"] == "Airspace opacity")
        no_findings_label = next(l for l in labels if l["label"] == "No findings")

        # Airspace opacity should be "Certain" (from Infection)
        assert airspace_label["confidence"] == "Certain"
        assert airspace_label["region"] == "upper"  # From first occurrence (Pneumonia)

        # No findings should be "Certain" (from Normal)
        assert no_findings_label["confidence"] == "Certain"
        assert no_findings_label["region"] == "middle"  # From first occurrence (No Findings -> No findings)

    def test_missing_confidence_defaults_to_uncertain(self):
        """Test that missing confidence field defaults to 'Uncertain'."""
        structured_labels = [
            {
                "body_part": "Chest",
                "labels": [
                    {"label": "Pneumonia"}  # No confidence field
                ]
            }
        ]

        mapping = {
            "Chest": {
                "Pneumonia": "Airspace opacity"
            }
        }

        result = remap_structured_labels(structured_labels, mapping)

        assert result[0]["labels"][0]["confidence"] == "Uncertain"

    def test_empty_body_part_string_raises_exception(self):
        """Test that empty body part string raises exception."""
        structured_labels = [
            {
                "body_part": "",
                "labels": [
                    {"label": "SomeLabel", "confidence": "Certain"}
                ]
            }
        ]

        mapping = {"Chest": {}}

        with pytest.raises(Exception) as exc_info:
            remap_structured_labels(structured_labels, mapping)

        assert "Body part  not found" in str(exc_info.value)