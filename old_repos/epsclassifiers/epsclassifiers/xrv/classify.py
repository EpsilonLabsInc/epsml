import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv

from epsutils.dicom import dicom_utils
from epsutils.image import image_utils


def main(config):
    img = dicom_utils.get_dicom_image(config.img_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    img = image_utils.numpy_array_to_pil_image(img, convert_to_uint8=True, convert_to_rgb=True)
    img = np.array(img)

    # Make sure image has at least 2 dimensions.
    assert len(img.shape) >= 2

    # Normalize image to [-1024, 1024] range.
    img = xrv.datasets.normalize(img, 255)

    # In case of multi-channel images, select the first channel.
    if len(img.shape) > 2:
        img = img[:, :, 0]

    img = img[None, :, :]

    # The models will resize the input to the correct size so this is optional.
    if config.resize:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    else:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    # Transform image.
    img = transform(img)

    # Get model.
    model = xrv.models.get_model(config.weights)
    model.eval()

    # Run prediction.
    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)

        if config.cuda:
            img = img.cuda()
            model = model.cuda()

        if config.feats:
            feats = model.features(img)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            output["feats"] = list(feats.cpu().detach().numpy().reshape(-1))

        preds = model(img).cpu()
        output["preds"] = dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))

    # Print results.
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default="", help='')
    parser.add_argument('img_path', type=str)
    parser.add_argument('-weights', type=str,default="densenet121-res224-all")
    parser.add_argument('-feats', default=False, help='', action='store_true')
    parser.add_argument('-cuda', default=False, help='', action='store_true')
    parser.add_argument('-resize', default=False, help='', action='store_true')

    config = parser.parse_args()

    main(config)
