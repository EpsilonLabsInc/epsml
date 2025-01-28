from io import BytesIO

import numpy as np
import torch
from PIL import Image

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.labels.cr_chest_labels import EXTENDED_CR_CHEST_LABELS
from epsclassifiers.intern_vit_classifier import InternVitClassifier

INITIAL_CHECKPOINT_DIR = "/home/andrej/mnt/models/training/internvl2.5_26b_finetune_lora_20241229_184000_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_no_label/checkpoint-58670"
TRAINING_CHECKPOINT = "/home/andrej/mnt/models/output/intern_vit_classifier-finetuning-on-gradient_cr/checkpoint/internvit_classifier_26b_no_labels_checkpoint.pt"
SHOW_CONDENSED_RESULTS = True
IMAGE_GCS_URI_LIST = [
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNDAVUSE39U08F/GRDN4FFS5ZTYB14G/studies/1.2.826.0.1.3680043.8.498.78181900056882406186628082793731723194/series/1.2.826.0.1.3680043.8.498.50034729269282488548399764152887492003/instances/1.2.826.0.1.3680043.8.498.55440357540027523193668070801511156312.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNSHWOJKIY53SY/GRDNAYTXFWVI6247/studies/1.2.826.0.1.3680043.8.498.95250369936666688034957243334415451518/series/1.2.826.0.1.3680043.8.498.23207201404760221380499247009432380291/instances/1.2.826.0.1.3680043.8.498.88066182359908319664341751962096219172.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNR9F0WLIU4Q7K/GRDNYHEV834OBWGP/studies/1.2.826.0.1.3680043.8.498.63638281040759594113425847139668256995/series/1.2.826.0.1.3680043.8.498.23825941860887763738056391316520370765/instances/1.2.826.0.1.3680043.8.498.51550151906806483011969399326807784527.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDN70YQJZW3BITK/GRDNKUNCE8Z52T1C/studies/1.2.826.0.1.3680043.8.498.86815754653460776630436660820862275876/series/1.2.826.0.1.3680043.8.498.51172097898141628460063891832925609280/instances/1.2.826.0.1.3680043.8.498.19392623491459181915983237310184909883.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNG266MGSSB1P8/GRDNFMRXX97XU1ZC/studies/1.2.826.0.1.3680043.8.498.72255198824247411067613445076454548934/series/1.2.826.0.1.3680043.8.498.92412775771660950280001292317203355479/instances/1.2.826.0.1.3680043.8.498.96604636927473579176803079629229429627.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDN05X9HSU78CMR/GRDNRJRIPN5GA8R6/studies/1.2.826.0.1.3680043.8.498.71494275942348117093886830458219780176/series/1.2.826.0.1.3680043.8.498.79826189301530796344211949795847350494/instances/1.2.826.0.1.3680043.8.498.31780071889023397757844273315798049503.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNFZOFU0ZEXB1J/GRDNEJDH55PASVYL/studies/1.2.826.0.1.3680043.8.498.59571246887651695830950819170857761548/series/1.2.826.0.1.3680043.8.498.50515595107492712627701956784065012032/instances/1.2.826.0.1.3680043.8.498.74892123108144672294999962914604159544.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNBQJJQNI134R6/GRDNQFC978EZRZ65/studies/1.2.826.0.1.3680043.8.498.64233318143898210930619073134632435374/series/1.2.826.0.1.3680043.8.498.57311919969822295887312894811543358072/instances/1.2.826.0.1.3680043.8.498.45865483472129609517267999457602912695.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNZ6NN915VQVAR/GRDNJSJ3774MQMEY/studies/1.2.826.0.1.3680043.8.498.63194655629113201951861275517198779088/series/1.2.826.0.1.3680043.8.498.46194060305944664190796002988663745310/instances/1.2.826.0.1.3680043.8.498.31760137998290225949175815443832954806.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNW5KXHQ7MKC8U/GRDN70SY4WJ8G2AY/studies/1.2.826.0.1.3680043.8.498.14760510045278457052392351401146456096/series/1.2.826.0.1.3680043.8.498.68996971786509213773213946703455024554/instances/1.2.826.0.1.3680043.8.498.69015658280035770291956337570560865811.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDN6IYG0MAAVG1T/GRDNDG6HDRST1KUL/studies/1.2.826.0.1.3680043.8.498.47768046247524320405186736575700327171/series/1.2.826.0.1.3680043.8.498.29311314504057877150877402681657906338/instances/1.2.826.0.1.3680043.8.498.51912959000425461880992043320021161783.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNJUZT41KYS3JW/GRDNURJD8XC70F6A/studies/1.2.826.0.1.3680043.8.498.72258103927636062690933587311857966986/series/1.2.826.0.1.3680043.8.498.54145272485215477342714975365428561587/instances/1.2.826.0.1.3680043.8.498.61998020371619952401069922386290206006.dcm",
    "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNK53OPGW0PHC6/GRDNZWUZ05TVRQZT/studies/1.2.826.0.1.3680043.8.498.35860937454444545590650895573749828639/series/1.2.826.0.1.3680043.8.498.74721565886517822288028482272113371082/instances/1.2.826.0.1.3680043.8.498.23129326874596845932359008219299658547.dcm"
]


def main():
    classifier = InternVitClassifier(num_classes=len(EXTENDED_CR_CHEST_LABELS), intern_vl_checkpoint_dir=INITIAL_CHECKPOINT_DIR, intern_vit_output_dim=3200)
    training_checkpoint = torch.load(TRAINING_CHECKPOINT)
    classifier.load_state_dict(training_checkpoint["model_state_dict"])
    classifier = classifier.to("cuda")
    classifier.eval()
    image_processor = classifier.get_image_processor()

    for image_path in IMAGE_GCS_URI_LIST:
        gcs_data = gcs_utils.split_gcs_uri(image_path)
        data = gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        image = dicom_utils.get_dicom_image(BytesIO(data), custom_windowing_parameters={"window_center": 0, "window_width": 0})
        image = image.astype(np.float32)
        eps = 1e-10
        image = (image - image.min()) / (image.max() - image.min() + eps) * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image = image.convert("RGB")

        pixel_values = image_processor(images=[image, image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # Run inference.
        output = classifier(pixel_values)[0]
        probabilities = torch.sigmoid(output)
        indices = torch.where(probabilities >= 0.5)[0]
        labels = [EXTENDED_CR_CHEST_LABELS[i.item()] for i in indices]

        if SHOW_CONDENSED_RESULTS:
            print(labels)
        else:
            print("--------------------------------------------------------")
            print("")
            print(f"GCS URI: {image_path}")
            print(f"Labels: {labels}")
            print(f"All labels: {EXTENDED_CR_CHEST_LABELS}")
            print(f"Probabilities: {probabilities}")
            print("")

        del pixel_values
        del output
        del probabilities
        del indices
        del labels
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
