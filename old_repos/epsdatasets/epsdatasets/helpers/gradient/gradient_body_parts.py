MAJOR_BODY_PARTS = [
    "Head/Brain",
    "Neck",
    "Chest",
    "Upper Abdomen",
    "Abdomen",
    "Pelvis",
    "Spine",
    "Cervical Spine",
    "Thoracic Spine",
    "Lumbar Spine",
    "Upper Extremities",
    "Lower Extremities",
    "Blood Vessels",
    "Whole Body",
]

assert len(MAJOR_BODY_PARTS) == 14

BODY_PART_LABELS_TO_MAJOR_BODY_PARTS_MAPPING = {
    # Head/Brain.
    "Head/Brain": "Head/Brain",
    "Face": "Head/Brain",
    "Maxillofacial": "Head/Brain",
    "Head/Face": "Head/Brain",
    "Orbits": "Head/Brain",

    # Neck.
    "Neck": "Neck",

    # Chest.
    "Chest": "Chest",
    "Upper Chest": "Chest",
    "Lower Chest": "Chest",
    "Upper Back": "Chest",
    "Lower Thorax": "Chest",

    # Abdomen.
    "Upper Abdomen": "Upper Abdomen",
    "Upper Abdominal Organs": "Upper Abdomen",
    "Abdomen": "Abdomen",

    # Pelvis.
    "Pelvis": "Pelvis",

    # Spine.
    "Spine": "Spine",
    "Thoracolumbar Spine": "Spine",
    "Lower Spine": "Spine",
    "Sacral Spine": "Spine",
    "Sacrum": "Spine",
    "Lumbosacral Spine": "Spine",
    "Cervical Spine": "Cervical Spine",
    "Lower Cervical Spine": "Cervical Spine",
    "Thoracic Spine": "Thoracic Spine",
    "Upper Thoracic Spine": "Thoracic Spine",
    "Lower Thoracic Spine": "Thoracic Spine",
    "Lumbar Spine": "Lumbar Spine",
    "Upper Lumbar Spine": "Lumbar Spine",
    "Lower Lumbar Spine": "Lumbar Spine",

    # Extremities.
    "Upper Extremities": "Upper Extremities",
    "Upper Extremities (Arms, Hands, Shoulders)": "Upper Extremities",
    "Shoulders": "Upper Extremities",
    "Hands": "Upper Extremities",
    "Lower Extremities": "Lower Extremities",
    "Lower Extremities (Legs, Feet, Hips)": "Lower Extremities",
    "Hips": "Lower Extremities",
    "Legs": "Lower Extremities",
    "Feet": "Lower Extremities",

    # Other.
    "Blood Vessels": "Blood Vessels",
    "Whole Body": "Whole Body",
    "Skeleton": "Whole Body"
}
