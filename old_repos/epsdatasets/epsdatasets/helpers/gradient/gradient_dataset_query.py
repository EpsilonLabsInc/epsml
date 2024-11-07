import ast

import pandas as pd
from tqdm import tqdm


class GradientDatasetQuery:
    """
    A helper class for querying Gradient dataset using patient ID, accession number, study instance UID and series instance UID.
    """

    def __init__(self, reports_file, show_duplicates=False, raise_exception_on_error=False):
        """
        Class constructor.

        Parameters:
        reports_file (str): Gradient dataset reports file.
        show_duplicates (bool): If True, all the duplicate series in the reports file are displayed.
        raise_exception_on_error (bool): If True, an exception is raised on error.
        """

        self.__reports_file = reports_file
        self.__show_duplicates = show_duplicates
        self.__raise_exception_on_error = raise_exception_on_error

        self.__build_index()

    def get_field_names(self):
        """
        Returns all the field names in the reports file.
        """

        return self.__df.columns.tolist()

    def get_num_rows(self):
        """
        Returns number of rows in the reports file.
        """

        return len(self.__df)

    def create_key(self, patient_id, accession_number, study_instance_uid, series_instance_uid):
        """
        Constructs a key from the patient ID, accession number, study instance UID and series instane UID, in the following format:
        <patient_id>_<accession_number>_studies_<study_instance_uid>_series_<series_instance_uid>_instances

        The key is used as the parameter in the get_data() function.
        """

        return f"{patient_id}_{accession_number}_studies_{study_instance_uid}_series_{series_instance_uid}_instances"

    def get_data(self, key, field=None):
        """
        Returns report data for the given key.

        Parameters:
        key (str): Key composed of patient ID, accession number, study instance UID and series instance UID. See create_key() function for more details.
        field (str): If None, all the field values are returned, otherwise only tha data corresponding to the given key is returned.
        """

        if key not in self.__index:
            return None

        if field is not None and field not in self.__df.columns:
            return None

        index = self.__index[key]
        row = self.__df.iloc[index]

        return row if field is None else row[field]

    def __build_index(self):
        print("Reading reports file")
        self.__df = pd.read_csv(self.__reports_file, sep=",", low_memory=False)

        print("Building index")
        self.__index = {}
        num_duplicates = 0
        for row_num, row in tqdm(self.__df.iterrows(), total=self.get_num_rows(), desc="Progress"):
            # Row ID.
            row_id = row["row_id"]
            if self.__check_empty_or_none(row_id):
                err_msg = f"Invalid row ID '{row_id}'"
                if self.__raise_exception_on_error:
                    raise ValueError(err_msg)
                print(err_msg)
                continue

            # Patient ID.
            patient_id = row["PatientID"]
            if self.__check_empty_or_none(patient_id):
                err_msg = f"Invalid patient ID '{patient_id}' at row '{row_id}'"
                if self.__raise_exception_on_error:
                    raise ValueError(err_msg)
                print(err_msg)
                continue

            # Patient ID.
            accession_number = row["AccessionNumber"]
            if self.__check_empty_or_none(accession_number):
                err_msg = f"Invalid accession number '{accession_number}' at row '{row_id}'"
                if self.__raise_exception_on_error:
                    raise ValueError(err_msg)
                print(err_msg)
                continue

            # Study Instance UID.
            study_instance_uid = row["StudyInstanceUid"]
            if self.__check_empty_or_none(study_instance_uid):
                err_msg = f"Invalid study instance UID '{study_instance_uid}' at row '{row_id}'"
                if self.__raise_exception_on_error:
                    raise ValueError(err_msg)
                print(err_msg)
                continue

            # Series Instance UIDs.
            series_instance_uids = ast.literal_eval(row["SeriesInstanceUid"])
            for series_instance_uid in series_instance_uids:
                if self.__check_empty_or_none(series_instance_uid):
                    err_msg = f"Invalid series instance UID '{series_instance_uid}' at row '{row_id}'"
                    if self.__raise_exception_on_error:
                        raise ValueError(err_msg)
                    print(err_msg)
                    continue

            # Update index.
            for series_instance_uid in series_instance_uids:
                key = self.create_key(patient_id=patient_id,
                                      accession_number=accession_number,
                                      study_instance_uid=study_instance_uid,
                                      series_instance_uid=series_instance_uid)

                if key in self.__index:
                    num_duplicates += 1
                    if self.__show_duplicates:
                        err_msg = f"Key '{key}' already in index"
                        if self.__raise_exception_on_error:
                            raise ValueError(err_msg)
                        print(err_msg)

                self.__index[key] = row_num

        print(f"Number of duplicate series: {num_duplicates}")

    def __check_empty_or_none(self, s: str):
        return s is None or s.strip() == ""

if __name__ == "__main__":
    print("Running example code")

    # Create query instance.
    query = GradientDatasetQuery(reports_file="/home/andrej/data/datasets/GRADIENT-DATABASE/REPORTS/CT/output_GRADIENT-DATABASE_REPORTS_CT_ct-16ago2024-batch-1.csv")
    print(f"Report fields: {query.get_field_names()}")
    print(f"Number of rows in report: {query.get_num_rows()}")

    # Get all the data with the given key.
    key = "GRDN0003S8F5QJ9E_GRDNBMX90HQQ7MK6_studies_1.2.826.0.1.3680043.8.498.48947129802685432049227454334300671562_series_1.2.826.0.1.3680043.8.498.12509078540895248317877542916630551173_instances"
    data = query.get_data(key=key, field=None)
    print(data)
    print("")

    # Get "BodyPartExamined" using the given key.
    data = query.get_data(key=key, field="BodyPartExamined")
    print(f"BodyPartExamined: {data}")
    print("")

    # Construct a key and get the data.
    key = query.create_key(patient_id="GRDNKF0UT3H8UDK7",
                           accession_number="GRDN8HNVOQJ5726C",
                           study_instance_uid="1.2.826.0.1.3680043.8.498.99457533010358929068509763915118458007",
                           series_instance_uid="1.2.826.0.1.3680043.8.498.27988335647622416789526475090711696983")
    data = query.get_data(key=key, field=None)
    print(data)
    print("")
