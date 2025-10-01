import pydicom

DICOM_FILE = r"C:\Users\Andrej\Desktop\1.2.392.200046.100.2.1.277579825363249.191105084110.1.1.1.1.dcm"

dicom_file = pydicom.dcmread(DICOM_FILE, force=True)

print(dicom_file)
