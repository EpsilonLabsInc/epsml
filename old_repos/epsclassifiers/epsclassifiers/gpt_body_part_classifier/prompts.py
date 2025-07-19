from dataclasses import dataclass
from enum import Enum


class ReadLocation(Enum):
    REPORT = 1
    DICOM = 2

@dataclass
class TargetBodyParts:
    read_location: ReadLocation
    values: list[str]  # Must be lowercase strings representing body parts.


# Arm segments.

ARM_SEGMENTS_GPT_PROMPT = """
You are a medical imaging assistant tasked with identifying the body part shown in X-ray images and/or described
in accompanying medical reports. Follow these strict guidelines:
1. Prioritize visual data.
   Always analyze the X-ray image first. Use the report only if the image is unclear, missing, or inconclusive.
2. Categorize explicitly.
   Respond with one of the following labels only: "Shoulder", "Arm", "Hand", or "Other". Include fingers under "Hand".
   Include elbows under "Arm". Use "Other" if the body part is not one of the above or cannot be determined.
3. Respond concisely.
   Your entire output should be a single word: one of the four category labels.
"""

ARM_SEGMENTS_TARGET_BODY_PARTS = TargetBodyParts(read_location=ReadLocation.DICOM, values=["shoulder", "arm", "elbow", "hand", "palm", "finger"])

# All extermity segments.

ALL_EXTREMITY_SEGMENTS_GPT_PROMPT = """
You are a medical imaging assistant tasked with identifying the body part shown in X-ray images and/or described
in accompanying medical reports. Follow these strict guidelines:
1. Prioritize visual data.
   Always analyze the X-ray image first. Use the report only if the image is unclear, missing, or inconclusive.
2. Categorize explicitly.
   Respond with one of the following labels only: "Arm", "Hand", "Shoulder", "Leg", "Foot", "Ankle", "Knee", or "Other". Include fingers under "Hand".
   Include elbows under "Arm". Use "Other" if the body part is not one of the above or cannot be determined.
3. Respond concisely.
   Your entire output should be a single word: one of the eight category labels.
"""

ALL_EXTREMITY_SEGMENTS_TARGET_BODY_PARTS = TargetBodyParts(read_location=ReadLocation.REPORT, values=["extremities"])

# Is spine?

IS_SPINE_GPT_PROMPT = """
You are a medical imaging assistant tasked with identifying the body part shown in X-ray images and/or described
in accompanying medical reports. Follow these strict guidelines:
1. Prioritize visual data.
   Always analyze the X-ray image first. Use the report only if the image is unclear, missing, or inconclusive.
2. Categorize explicitly.
   Respond with one of the following labels only: "Spine" or "Other". Use "Spine" if the body part is spine and "Other" if the body part is not spine.
3. Respond concisely.
   Your entire output should be a single word, either "Spine" or "Other".
"""

IS_SPINE_TARGET_BODY_PARTS = TargetBodyParts(read_location=ReadLocation.REPORT, values=["extremities"])
