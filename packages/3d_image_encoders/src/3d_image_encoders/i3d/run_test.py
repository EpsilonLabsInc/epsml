import argparse
import copy
import json

from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from i3d_resnet import I3DResNet


def run_test(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder("data/dummy-dataset", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    class_idx = json.load(open("data/imagenet_class_index.json"))
    imagenet_classes = [class_idx[str(k)][1] for k in range(len(class_idx))]

    print(f"Cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.resnet_nb == 50:
        resnet = torchvision.models.resnet50(pretrained=True)
    elif args.resnet_nb == 101:
        resnet = torchvision.models.resnet101(pretrained=True)
    elif args.resnet_nb == 152:
        resnet = torchvision.models.resnet152(pretrained=True)
    else:
        raise ValueError("Argument resnet_nb should be in [50|101|152], got {args.resnet_nb} instead")

    i3d_resnet = I3DResNet(copy.deepcopy(resnet), args.frame_nb)
    i3d_resnet.eval()
    i3d_resnet = i3d_resnet.to(device)

    resnet.eval()
    resnet = resnet.to(device)

    with torch.no_grad():
        for index, (image, label) in enumerate(dataset):
            image_2d = image.unsqueeze(0).to(device)
            output_2d = resnet(image_2d)
            probs_2d = torch.softmax(output_2d, dim=1)
            top_probs, top_indices = torch.topk(probs_2d, k=3)
            print("Top ResNet results:")
            for prob, idx in zip(top_probs[0], top_indices[0]):
                print(f"  Class: {imagenet_classes[idx.item()]}, Probability: {prob.item()}")

            image_3d = image.unsqueeze(0).unsqueeze(2).repeat(1, 1, args.frame_nb, 1, 1).to(device)

            print(type(image))
            print(image.shape)
            print(image_3d.shape)

            output_3d = i3d_resnet(image_3d)
            probs_3d = torch.softmax(output_3d, dim=1)
            top_probs, top_indices = torch.topk(probs_3d, k=3)
            print("Top I3D ResNet results:")
            for prob, idx in zip(top_probs[0], top_indices[0]):
                print(f"  Class: {imagenet_classes[idx.item()]}, Probability: {prob.item()}")

            out_diff = output_2d - output_3d
            print(f"Mean abs error between 2D and I3D: {out_diff.abs().mean()}")
            print(f"Maximum error between 2D and I3D: {out_diff.max()}")

            if args.display_samples:
                img_np = image.numpy().transpose(1, 2, 0)
                plt.imshow(img_np)
                plt.show()

            print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Performs I3D test")
    parser.add_argument("dataset_path", type=str, help="Dataset path")
    parser.add_argument("--resnet_nb", type=int, default=50, help="What version of ResNet to use, in [50|101|152]")
    parser.add_argument("--display_samples", action="store_true", help="Whether to display samples and associated scores for 3d inflated resnet")
    parser.add_argument("--top_k", type=int, default="5", help="When display_samples, number of top classes to display")
    parser.add_argument("--frame_nb", type=int, default="16", help="Number of video_frames to use (should be a multiple of 8)")
    args = parser.parse_args()
    run_test(args)
