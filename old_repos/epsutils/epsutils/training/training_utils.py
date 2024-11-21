import numpy as np
import pynvml
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


def get_torch_model_size_in_mib(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    total_size = param_size + buffer_size
    return total_size // 1024**2


def get_num_torch_parameters(model, requires_grad_only=True):
    if requires_grad_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    else:
        return sum(param.numel() for param in model.parameters())


def get_gpu_utilization_in_mib():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024**2


def create_lr_scheduler_with_warmup_and_cosine_annealing(optimizer, total_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))  # Linear warmup
        else:
            return 1.0

    warmup_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_steps - warmup_steps, eta_min=0.0)

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return lr_scheduler


def convert_pil_image_to_normalized_torch_tensor(image, normalization_mean=None, normalization_std=None, use_half_precision=False):
    # Cast image to float32.
    image_np = np.array(image).astype(np.float32)

    # Normalize to [0, 1] range.
    epsilon = 1e-10
    min_val = np.min(image_np)
    max_val = np.max(image_np)
    image_np = (image_np - min_val) / (max_val - min_val + epsilon)

    # Use half precision.
    if use_half_precision:
        image_np = image_np.astype(np.float16)

    # Normalize image.
    if normalization_mean is not None and normalization_std is not None:
        image_np = (image_np - normalization_mean) / normalization_std

    # Convert to torch tensor.
    tensor = torch.from_numpy(image_np).unsqueeze(0)

    return tensor
