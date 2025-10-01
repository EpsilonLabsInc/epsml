import logging

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

GCS_IMAGES_DIR = "gs://gradient-crs/22JUL2024"
OUTPUT_FILE = "./output/images_list.csv"
ACCESSION_NUMBERS = [
    "GRDN00JXARS4IYRZ",
    "GRDNMPJPYVYN3LYO",
    "GRDNXGYOUJ5Y2PYE",
    "GRDNZZ5D3BEQDNMQ",
    "GRDNUTALUXHPNJ6B",
    "GRDN91F9DPO96KAG",
    "GRDN10T2YNPT4SPB",
    "GRDN0HHVWWJ96O0K",
    "GRDNG13J7WFPBATN",
    "GRDNW04K3K3CJFNA",
    "GRDNOD19VBP4PNN8",
    "GRDNGRLRVR88OFML",
    "GRDNNZKXQK1RYD1A",
    "GRDNURJD8XC70F6A",
    "GRDNO1QURGF7VBAR",
    "GRDNCJ51OTJUZSND",
    "GRDNCR8YI7WKFFDG",
    "GRDN3CWJJGJT7P86",
    "GRDN2HX2EGYP3B6K",
    "GRDN6TBLD4UJ9V4S",
    "GRDNR6H4JMPOF1VF",
    "GRDNBFD776OLCXKZ",
    "GRDNAAD2G16EOC5F",
    "GRDNW7X3DWLGIASY",
    "GRDNG8YN97VBVWAM",
    "GRDNO73SYOVMWAOE",
    "GRDNOQF962U54R79",
    "GRDNHK0MBXUTCQUT",
    "GRDNYHEV834OBWGP",
    "GRDNO3X3309QPK63",
    "GRDNQZ2ARO9VPDMW",
    "GRDNU8Y7W8LLET32",
    "GRDNXLXQH1GZ6037",
    "GRDNWTQEN22X5DB4",
    "GRDNWYU1PDFPBXBV",
    "GRDNLZS4MO3T75J7",
    "GRDNI7L7ZH5Y287U",
    "GRDNIG50MSZEXKB9",
    "GRDNSFSDMJSNTZZV",
    "GRDN0O32QQLEPSGP",
    "GRDN7LH3GX2WL2VY",
    "GRDNZD9FC6XBIUZZ",
    "GRDN85UIL28NC6UY",
    "GRDNWD3PHF22KLBH",
    "GRDNM78WLQ7LRL86",
    "GRDN3ZCHQJVRMQLE",
    "GRDN0MRM42T2G04K",
    "GRDN5YCPF8C37YQO",
    "GRDNKUNCE8Z52T1C",
    "GRDNEYAUHPQB9DL0",
    "GRDN4DM8PHNH168E",
    "GRDN1K12WKOXJWRY",
    "GRDNEXJM5QAUM63U",
    "GRDNSN4L9L3QBJY8",
    "GRDNV301Z915TOGG",
    "GRDN8ODJSE4XGQN3",
    "GRDNIOXCFSKW60O4",
    "GRDNRM25XJSFXRAC",
    "GRDN4FFS5ZTYB14G",
    "GRDN3M1LHSQYQVSV",
    "GRDNQFC978EZRZ65",
    "GRDNDG6HDRST1KUL",
    "GRDN70SY4WJ8G2AY",
    "GRDNI6J233TWEYN2",
    "GRDN5P5M2TJIOTGS",
    "GRDNRJRIPN5GA8R6",
    "GRDND4I3N3J2TNRD",
    "GRDND9SGJ4IDJZS0",
    "GRDNAE39UZ8V5WXT",
    "GRDNBF963EJGSSUA",
    "GRDNUT52CHTMAHMB",
    "GRDNM53GVF76QBKZ",
    "GRDNF9JBGBSUQNYB",
    "GRDNJ2IMT51IEFJA",
    "GRDNEJDH55PASVYL",
    "GRDN15X5JESH3GLL",
    "GRDNKYQL1028QR5Z",
    "GRDNRE0X178CV64I",
    "GRDNMPF5D5DMJ7VV",
    "GRDNN4JNP8Q19JLX",
    "GRDN8STQL81QWJ11",
    "GRDNRG1QM47YD0SV",
    "GRDNIR782M935KJH",
    "GRDN55WXDJ9RVWZO",
    "GRDNP4J45KLXO73A",
    "GRDNJSJ3774MQMEY",
    "GRDNITH7QGKDRAPE",
    "GRDN0TA15185BYZN",
    "GRDNYZVHXYZBZTAK",
    "GRDNBZUA6S6ISUL0",
    "GRDNFMRXX97XU1ZC",
    "GRDNN2Y5PWME0N67",
    "GRDNVY3JCWY99ZLL",
    "GRDN2W4WV9QFJE9E",
    "GRDNQ7R24P0AB5V7",
    "GRDNB16NA1HVAF7N",
    "GRDNGKB02BVJ3URF",
    "GRDNXR4GYHDGYY9N",
    "GRDNA38JS5FUPFOI",
    "GRDNQ2T3UYIEC7H1",
    "GRDNEYLADSNJTAVL",
    "GRDNG7LJOLS60LJ1",
    "GRDNQFSWZF8GIV1O",
    "GRDN4LLKEFUQJJCA",
    "GRDNYZ3NOCP7YDHE",
    "GRDNZJWT8BMUZ5GL",
    "GRDN1B2GWBTE9ROF",
    "GRDNZJ6PHOBCV6LF",
    "GRDNHQJ4YU9V5E3T",
    "GRDNJEXUW66CDFNK",
    "GRDNOW1H8HXVCO64",
    "GRDN1WRHJ66L4GTL",
    "GRDNYYORMKDZV9XZ",
    "GRDNXCEMTKGNBW53",
    "GRDNPBYP3YBEOJU2",
    "GRDNC2QKKO5P5EUT",
    "GRDNC54PFWDORE9V",
    "GRDNZWUZ05TVRQZT",
    "GRDNB99MUWAWFBQ0",
    "GRDNQJ53MF1XAV8N",
    "GRDNAYTXFWVI6247",
    "GRDN28MT669436T7",
    "GRDNE46ZEPLTZ7EV",
    "GRDNMQEESQDBGFKC",
    "GRDN1WI0Q3BPW0OS",
    "GRDNADOZLQUK0I44",
    "GRDNP56KKNWOCRS1",
    "GRDN15SGS3BW8TQ5",
    "GRDNE5CUNCPGWCDE",
    "GRDNVIK56OVDGPJE",
    "GRDNHXRYQ9NNYP5I",
    "GRDNJI1TZ0NDWFZE",
    "GRDN3B1PXUI460ZF",
    "GRDNAY0ED61COBGH",
    "GRDNH7DD92TIURCH",
    "GRDNSTBRGNRMVNG8"
]


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    print(f"Getting a list of TXT files in {GCS_IMAGES_DIR}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_IMAGES_DIR)
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_dir=gcs_data["gcs_path"])
    files_in_bucket = [file for file in files_in_bucket if file.endswith(".txt")]
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total files found: {len(files_in_bucket)}")

    # TODO: Optimize for speed.
    for accession_number in tqdm(ACCESSION_NUMBERS, total=len(ACCESSION_NUMBERS), desc="Processing"):
        found = False
        for file in files_in_bucket:
            if file.split("_")[1] == accession_number:
                logging.info(f"gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/{file.replace('_', '/').replace('.txt', '.dcm')}")
                found = True
                break

        if not found:
            logging.error(f"ERROR: Accession number {accession_number} not found")


if __name__ == "__main__":
    main()
