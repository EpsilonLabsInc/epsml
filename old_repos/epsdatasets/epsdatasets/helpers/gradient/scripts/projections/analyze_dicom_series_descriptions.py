DICOM_SERIES_DESCRIPTIONS_FILE = "/home/andrej/work/epsdatasets/epsdatasets/helpers/gradient/scripts/gradient-crs-22JUL2024-series-descriptions.csv"

def main():
    # Get all DICOM series descriptions.
    with open(DICOM_SERIES_DESCRIPTIONS_FILE, "r") as file:
        descriptions = set()
        for line in file:
            descriptions.add(line.strip())

        print("All descriptions:")
        print(descriptions)
        print(f"Total num of descriptions: {len(descriptions)}")
        print("")

    # Get frontal descriptions only.
    frontal_descriptions = set()
    for d in descriptions:
        words = d.lower().split()
        if {"pa", "ap"}.intersection(words) and {"chest", "rib", "ribs"}.intersection(words) and not {"lateral", "lat"}.intersection(words):
            frontal_descriptions.add(d)

    print("Frontal descriptions:")
    print(frontal_descriptions)
    print(f"Total num of frontal descriptions: {len(frontal_descriptions)}")

if __name__ == "__main__":
    main()
