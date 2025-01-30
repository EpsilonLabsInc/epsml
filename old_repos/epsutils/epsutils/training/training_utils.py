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


def create_lr_scheduler(optimizer, total_steps, warmup_steps=None, apply_cosine_annealing=False):
    schedulers = []
    milestones = []

    # Warmup.
    if warmup_steps is not None:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))  # Linear warmup.
            else:
                return 1.0

        warmup_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    # Cosine annealing.
    if apply_cosine_annealing:
        cosine_steps = total_steps - warmup_steps if warmup_steps else total_steps
        cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=cosine_steps, eta_min=0.0)
        schedulers.append(cosine_scheduler)

    lr_scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

    return lr_scheduler


def convert_pil_image_to_normalized_torch_tensor(image, normalization_mean=None, normalization_std=None, use_half_precision=False):
    # Convert to numpy array.
    image_np = np.array(image)
    dtype = image_np.dtype
    assert dtype in (np.int16, np.uint16)

    # Cast image to float32.
    image_np = image_np.astype(np.float32)

    # Normalize to [0, 1] range.
    if dtype == np.uint16:
        image_np /= 65535
    elif dtype == np.int16:
        image_np = image_np - image_np.min()
        image_np /= 65535

    # Use half precision.
    if use_half_precision:
        image_np = image_np.astype(np.float16)

    # Normalize image.
    if normalization_mean is not None and normalization_std is not None:
        image_np = (image_np - normalization_mean) / normalization_std

    # Convert to torch tensor.
    tensor = torch.from_numpy(image_np).unsqueeze(0)

    return tensor
