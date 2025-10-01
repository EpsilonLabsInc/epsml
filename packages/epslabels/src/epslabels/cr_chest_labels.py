CR_CHEST_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Findings"
]

assert len(CR_CHEST_LABELS) == 14

EXTENDED_CR_CHEST_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Findings",
    "Other"
]

assert len(EXTENDED_CR_CHEST_LABELS) == 15

EXTENDED_CR_CHEST_LABELS_WITH_SEPARATED_FRACTURES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture (Recent)",
    "Fracture (Old/Healed)",
    "Support Devices",
    "No Findings",
    "Other"
]

assert len(EXTENDED_CR_CHEST_LABELS_WITH_SEPARATED_FRACTURES) == 16
