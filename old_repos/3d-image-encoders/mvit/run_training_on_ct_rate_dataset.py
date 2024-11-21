import torch
import torchvision
from torchvision.models.video import mvit_v2_s

from epsdatasets.helpers.ctrate.ct_rate_dataset_helper import CtRateDatasetHelper
from epsutils.training import training_utils
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsType, MlopsParameters
from epsutils.training.sample_balanced_bce_with_logits_loss import SampleBalancedBCEWithLogitsLoss

if __name__ == "__main__":
    # General settings.
    model_name = "mvit"
    dataset_name = "ct_rate"
    output_dir = "/home/andrej/data/ct-rate/output"

    # CT-RATE dataset helper.
    training_file = "/home/andrej/data/ct-rate/train_predicted_labels.csv"
    validation_file = "/home/andrej/data/ct-rate/valid_predicted_labels.csv"

    # Training settings.
    perform_intra_epoch_validation = True
    send_wandb_notification = True
    device = "cuda"
    device_ids = None  # Use one (the default) GPU.
    # device_ids = [0, 1, 2, 3]  # Use 4 GPUs.
    num_training_workers_per_gpu = 4
    num_validation_workers_per_gpu = 4
    half_model_precision = False
    use_pretrained_model = False
    learning_rate = 1e-6
    warmup_ratio = 1 / 5
    num_epochs = 3
    num_steps_per_checkpoint = 5000
    gradient_accumulation_steps = 2
    training_batch_size = 1
    validation_batch_size = 1
    images_mean = 0.2567
    images_std = 0.1840
    target_image_size = 224
    normalization_depth = 112
    sample_slices = True
    pos_weight_fact = 1.0
    loss_function = SampleBalancedBCEWithLogitsLoss(pos_weight_fact=pos_weight_fact)  # torch.nn.BCEWithLogitsLoss()

    experiment_name = f"{model_name}-finetuning-on-{dataset_name}"
    mlops_experiment_name = f"{experiment_name}"
    experiment_dir = f"{output_dir}/{experiment_name}"
    save_model_filename = f"{experiment_dir}/{experiment_name}.pt"
    save_model_weights_filename = f"{experiment_dir}/{experiment_name}-weights.pt"
    save_parallel_model_filename = f"{experiment_dir}/{experiment_name}-parallel.pt"
    checkpoint_dir = f"{experiment_dir}/checkpoint"

    print(f"Target volume dimensions: {target_image_size}x{target_image_size}x{normalization_depth}")

    # Load the dataset.
    print("Loading the dataset")
    dataset_helper = CtRateDatasetHelper(training_file=training_file, validation_file=validation_file)

    # Get number of labels.
    labels = dataset_helper.get_labels()
    num_labels = len(labels)
    print(f"Number of labels: {num_labels}")

    # Create the model and replace its input and head layers.
    print("Creating the MVIT model")
    model = mvit_v2_s(pretrained=use_pretrained_model, num_classes=num_labels)
    org_conv_proj = model.conv_proj
    model.conv_proj = torch.nn.Conv3d(
        1,  # Input channels
        org_conv_proj.out_channels,
        kernel_size=org_conv_proj.kernel_size,
        stride=org_conv_proj.stride,
        padding=org_conv_proj.padding,
        bias=org_conv_proj.bias is not None
    )

    for param in model.parameters():
        param.requires_grad = True

    if half_model_precision:
        model.half()

    # Prepare the training data.
    print("Preparing the training data")

    training_parameters = TrainingParameters(learning_rate=learning_rate,
                                             warmup_ratio=warmup_ratio,
                                             num_epochs=num_epochs,
                                             gradient_accumulation_steps=gradient_accumulation_steps,
                                             training_batch_size=training_batch_size,
                                             validation_batch_size=validation_batch_size,
                                             criterion=loss_function,
                                             checkpoint_dir=checkpoint_dir,
                                             perform_intra_epoch_validation=perform_intra_epoch_validation,
                                             num_steps_per_checkpoint=num_steps_per_checkpoint,
                                             num_training_workers_per_gpu=num_training_workers_per_gpu,
                                             num_validation_workers_per_gpu=num_validation_workers_per_gpu)

    mlops_parameters = MlopsParameters(mlops_type=MlopsType.WANDB,
                                       experiment_name=mlops_experiment_name,
                                       notes=f"Volume size = {target_image_size}x{target_image_size}x{normalization_depth}, "
                                             f"sample_slices={sample_slices}, use_pretrained_model={use_pretrained_model}, pos_weight_fact={pos_weight_fact}",
                                       send_notification=send_wandb_notification)

    training_helper = TorchTrainingHelper(model=model,
                                          dataset_helper=dataset_helper,
                                          device=device,
                                          device_ids=device_ids,
                                          training_parameters=training_parameters,
                                          mlops_parameters=mlops_parameters)

    def get_torch_image(item, split):
        images = dataset_helper.get_pil_image(item=item,
                                              split=split,
                                              target_image_size=target_image_size,
                                              normalization_depth=normalization_depth,
                                              sample_slices=sample_slices)

        tensors = [training_utils.convert_pil_image_to_normalized_torch_tensor(image=image, use_half_precision=half_model_precision) for image in images]
        stacked_tensor = torch.stack(tensors)
        # Instead of the tensor shape (num_slices, num_channels, image_height, image_width),
        # which is obtained by stacking the tensors, the model requires the following shape:
        # (num_channels, num_slices, image_height, image_width), which is obtained by
        # premuting the dimensions.
        stacked_tensor = stacked_tensor.permute(1, 0, 2, 3)
        return stacked_tensor

    def collate_function_for_training(samples):
        images = torch.stack([get_torch_image(item, "train") for item in samples])
        labels = torch.stack([dataset_helper.get_torch_label(item) for item in samples])
        return images, labels

    def collate_function_for_validation(samples):
        images = torch.stack([get_torch_image(item, "valid") for item in samples])
        labels = torch.stack([dataset_helper.get_torch_label(item) for item in samples])
        return images, labels

    training_helper.start_training(collate_function_for_training=collate_function_for_training, collate_function_for_validation=collate_function_for_validation)
    training_helper.save_model_weights(model_weights_file_name=save_model_weights_filename)
