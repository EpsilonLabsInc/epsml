import re


def pascal_case_to_snake_case(name):
    # Handle acronym boundaries first (e.g., StudyUID → Study_UID).
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Handle acronym followed by regular word (e.g., UIDStudy → UID_Study).
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    return name.lower()
