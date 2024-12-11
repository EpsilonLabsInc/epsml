EPSILON_BODY_PARTS = [
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

assert len(EPSILON_BODY_PARTS) == 14

CSV_TO_EPSILON_BODY_PARTS_MAPPING = {
    "Head": "Head/Brain",
    "Neck": "Neck",
    "Chest": "Chest",
    "Heart": "Chest",
    "Abdomen": "Abdomen",
    "Pelvis": "Pelvis",
    "Spine": "Spine",
    "Arm": "Upper Extremities",
    "Leg": "Lower Extremities",
    "Foot": "Lower Extremities",
    "Whole Body": "Whole Body"
}

GPT_TO_EPSILON_BODY_PARTS_MAPPING = {
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

MONAI_TO_EPSILON_BODY_PARTS_MAPPING = {
    1: "Upper Abdomen",  # spleen
    2: "Abdomen",  # kidney_right
    3: "Abdomen",  # kidney_left
    4: "Upper Abdomen",  # gallbladder
    5: "Upper Abdomen",  # liver
    6: "Upper Abdomen",  # stomach
    7: "Blood Vessels",  # aorta
    8: "Blood Vessels",  # inferior_vena_cava
    9: "Blood Vessels",  # portal_vein_and_splenic_vein
    10: "Upper Abdomen",  # pancreas
    11: "Upper Abdomen",  # adrenal_gland_right
    12: "Upper Abdomen",  # adrenal_gland_left
    13: "Chest",  # lung_upper_lobe_left
    14: "Chest",  # lung_lower_lobe_left
    15: "Chest",  # lung_upper_lobe_right
    16: "Chest",  # lung_middle_lobe_right
    17: "Chest",  # lung_lower_lobe_right
    18: "Lumbar Spine",  # vertebrae_L5
    19: "Lumbar Spine",  # vertebrae_L4
    20: "Lumbar Spine",  # vertebrae_L3
    21: "Lumbar Spine",  # vertebrae_L2
    22: "Lumbar Spine",  # vertebrae_L1
    23: "Thoracic Spine",  # vertebrae_T12
    24: "Thoracic Spine",  # vertebrae_T11
    25: "Thoracic Spine",  # vertebrae_T10
    26: "Thoracic Spine",  # vertebrae_T9
    27: "Thoracic Spine",  # vertebrae_T8
    28: "Thoracic Spine",  # vertebrae_T7
    29: "Thoracic Spine",  # vertebrae_T6
    30: "Thoracic Spine",  # vertebrae_T5
    31: "Thoracic Spine",  # vertebrae_T4
    32: "Thoracic Spine",  # vertebrae_T3
    33: "Thoracic Spine",  # vertebrae_T2
    34: "Thoracic Spine",  # vertebrae_T1
    35: "Cervical Spine",  # vertebrae_C7
    36: "Cervical Spine",  # vertebrae_C6
    37: "Cervical Spine",  # vertebrae_C5
    38: "Cervical Spine",  # vertebrae_C4
    39: "Cervical Spine",  # vertebrae_C3
    40: "Cervical Spine",  # vertebrae_C2
    41: "Cervical Spine",  # vertebrae_C1
    42: "Neck",  # esophagus
    43: "Neck",  # trachea
    44: "Chest",  # heart_myocardium
    45: "Chest",  # heart_atrium_left
    46: "Chest",  # heart_ventricle_left
    47: "Chest",  # heart_atrium_right
    48: "Chest",  # heart_ventricle_right
    49: "Blood Vessels",  # pulmonary_artery
    50: "Head/Brain",  # brain
    51: "Blood Vessels",  # iliac_artery_left
    52: "Blood Vessels",  # iliac_artery_right
    53: "Blood Vessels",  # iliac_vena_left
    54: "Blood Vessels",  # iliac_vena_right
    55: "Abdomen",  # small_bowel
    56: "Abdomen",  # duodenum
    57: "Abdomen",  # colon
    58: "Chest",  # rib_left_1
    59: "Chest",  # rib_left_2
    60: "Chest",  # rib_left_3
    61: "Chest",  # rib_left_4
    62: "Chest",  # rib_left_5
    63: "Chest",  # rib_left_6
    64: "Chest",  # rib_left_7
    65: "Chest",  # rib_left_8
    66: "Chest",  # rib_left_9
    67: "Chest",  # rib_left_10
    68: "Chest",  # rib_left_11
    69: "Chest",  # rib_left_12
    70: "Chest",  # rib_right_1
    71: "Chest",  # rib_right_2
    72: "Chest",  # rib_right_3
    73: "Chest",  # rib_right_4
    74: "Chest",  # rib_right_5
    75: "Chest",  # rib_right_6
    76: "Chest",  # rib_right_7
    77: "Chest",  # rib_right_8
    78: "Chest",  # rib_right_9
    79: "Chest",  # rib_right_10
    80: "Chest",  # rib_right_11
    81: "Chest",  # rib_right_12
    82: "Upper Extremities",  # humerus_left
    83: "Upper Extremities",  # humerus_right
    84: "Chest",  # scapula_left
    85: "Chest",  # scapula_right
    86: "Chest",  # clavicula_left
    87: "Chest",  # clavicula_right
    88: "Lower Extremities",  # femur_left
    89: "Lower Extremities",  # femur_right
    90: "Pelvis",  # hip_left
    91: "Pelvis",  # hip_right
    92: "Pelvis",  # sacrum
    93: "Head/Brain",  # face
    94: "Pelvis",  # gluteus_maximus_left
    95: "Pelvis",  # gluteus_maximus_right
    96: "Pelvis",  # gluteus_medius_left
    97: "Pelvis",  # gluteus_medius_right
    98: "Pelvis",  # gluteus_minimus_left
    99: "Pelvis",  # gluteus_minimus_right
    100: "Spine",  # autochthon_left
    101: "Spine",  # autochthon_right
    102: "Pelvis",  # iliopsoas_left
    103: "Pelvis",  # iliopsoas_right
    104: "Abdomen"  # urinary_bladder
}
