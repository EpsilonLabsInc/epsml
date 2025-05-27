import copy

import pydicom
from pydicom.uid import JPEG2000, ImplicitVRLittleEndian, JPEG2000Lossless


def decompress(dataset: pydicom.Dataset) -> pydicom.Dataset:
    try:
        dicom_dataset_deepcopy = copy.deepcopy(dataset)
        dicom_dataset_deepcopy.ensure_file_meta()
        if hasattr(dicom_dataset_deepcopy, 'TransferSyntaxUID'):
            transfer_syntax_uid = dicom_dataset_deepcopy.file_meta.TransferSyntaxUID
        else:
            transfer_syntax_uid = infer_transfer_syntax_uid(dicom_dataset_deepcopy)
        dicom_dataset_deepcopy.file_meta.TransferSyntaxUID = transfer_syntax_uid
        if dicom_dataset_deepcopy.file_meta.TransferSyntaxUID.is_compressed:
            decompressed_dicom_dataset = pydicom.pixels.decompress(dicom_dataset_deepcopy)
        else:
            decompressed_dicom_dataset = None
    except Exception as e:
        decompressed_dicom_dataset = None

    return decompressed_dicom_dataset or dataset


def infer_transfer_syntax_uid(dataset: pydicom.Dataset) -> pydicom.uid.UID:
    # First, check if the dataset has a Transfer Syntax UID
    dataset.ensure_file_meta()
    if hasattr(dataset.file_meta, "TransferSyntaxUID"):
        return dataset.file_meta.TransferSyntaxUID

    # Next, check if the dataset's private tags contain a Transfer Syntax UID
    for elem in dataset:
        # Some centers use private tags to store Transfer Syntax UID information
        if elem.is_private:
            val = elem.value
            # Reduce unnecessary operations if it is not possible for val to be a string
            if type(val) is not bytes or type(val) is not str:
                continue
            if type(val) is bytes:
                # Sometimes the Syntax Transfer UID is a string encoded as bytes
                try:
                    val = val.decode()
                except UnicodeDecodeError:
                    continue  # And occasionally, a bytes typed value is not decodable to a string
            val.strip()
            # We do this to refine, a Transfer Syntax UID will *always* contain periods ('.')
            if type(val) is str and "." in val:
                potential_uid = val
                # pydicom.uid.AllTransferSyntaxes is a list of all Transfer Syntax UIDs as *strings* (not type pydicom.uid.UID)
                if potential_uid in pydicom.uid.AllTransferSyntaxes:
                    uid = pydicom.uid.UID(potential_uid)
                    return uid

    # Next, check if the pixel data is encoded as a bitstream (i.e., JPEG 2000)
    fragments = pydicom.encaps.decode_data_sequence(dataset.PixelData)
    if len(fragments) > 0 and len(fragments[0]) >= 3:
        first_bytes = fragments[0][:2]
        # JPEG 2000 spec states that all JPEG 2000 codestreams start with `FF 4F`
        if first_bytes == b"\xff\x4f":
            if hasattr(dataset, "LossyImageCompression") and int(dataset.LossyImageCompression) == 1:
                return JPEG2000
            else:
                return JPEG2000Lossless

    # If neither, assume Implicit VR Little Endian (default per DICOM spec)
    return ImplicitVRLittleEndian


def handle_dicom_compression(dataset: pydicom.Dataset) -> pydicom.Dataset:
    # Protect the integrity of reference in case `dataset` is deleted and gc.collect() is called outside of this method
    dataset_deepcopy = copy.deepcopy(dataset)
    # Ensures there is .file_meta, adds it to dataset if not
    dataset_deepcopy.ensure_file_meta()

    if not hasattr(dataset_deepcopy.file_meta, "TransferSyntaxUID"):
        transfer_syntax_uid = infer_transfer_syntax_uid(dataset_deepcopy)
        dataset_deepcopy.file_meta.TransferSyntaxUID = transfer_syntax_uid

    try:
        if dataset_deepcopy.file_meta.TransferSyntaxUID != JPEG2000Lossless and dataset_deepcopy.file_meta.TransferSyntaxUID.is_compressed:
            decompressed_dataset = pydicom.pixels.decompress(dataset_deepcopy)

            # Increases efficiency in compression
            if "RGB" in decompressed_dataset.PhotometricInterpretation:
                decompressed_dataset.PhotometricInterpretation = "YBR_RCT"

            compressed_dataset = copy.deepcopy(decompressed_dataset)

            pydicom.pixels.compress(compressed_dataset, transfer_syntax_uid=JPEG2000Lossless)

            return compressed_dataset

        elif dataset_deepcopy.file_meta.TransferSyntaxUID == JPEG2000Lossless:

            # If the already-compressed dataset uses RGB, decompress, switch to YBR_RCT, and re-compress for efficiency
            if "RGB" in dataset_deepcopy.PhotometricInterpretation:
                decompressed_dataset = copy.deepcopy(dataset_deepcopy)
                pydicom.pixels.decompress(decompressed_dataset)
                dataset_deepcopy.PhotometricInterpretation = "YBR_RCT"

                compressed_dataset = copy.deepcopy(dataset_deepcopy)

                pydicom.pixels.compress(compressed_dataset, transfer_syntax_uid=JPEG2000Lossless)
            else:
                compressed_dataset = copy.deepcopy(dataset_deepcopy)

            return compressed_dataset

        else:
            compressed_dataset = copy.deepcopy(dataset_deepcopy)
            try:
                pydicom.pixels.compress(compressed_dataset, transfer_syntax_uid=JPEG2000Lossless)
            except Exception as e:
                return dataset_deepcopy

        return compressed_dataset

    except Exception as e:
        return dataset_deepcopy
