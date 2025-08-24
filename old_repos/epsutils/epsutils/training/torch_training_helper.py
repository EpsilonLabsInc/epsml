import inspect
import json
import math
import os
from collections.abc import Mapping
from datetime import datetime
from enum import Enum

import mlflow
import torch
import wandb
from torch.optim import AdamW
from torch.nn.parallel.scatter_gather import scatter
from tqdm.auto import tqdm

from epsutils.training import training_utils
from epsutils.training.confusion_matrix_calculator import ConfusionMatrixCalculator
from epsutils.training.evaluation_metrics_calculator import EvaluationMetricsCalculator
from epsutils.training.performance_curve_calculator import PerformanceCurveCalculator, PerformanceCurveType
from epsutils.training.scores_distribution_generator import ScoresDistributionGenerator


def custom_scatter(inputs, target_gpus, dim=0):
    """Custom scatter function for nested list structures"""
    if isinstance(inputs, dict):
        scattered_inputs = {}
        for key, value in inputs.items():
            if key == 'images' and isinstance(value, list):
                # Find max sublist length for padding
                max_sublist_len = max(len(sublist) for sublist in value)
                
                # Split the outer list across devices
                chunk_size = len(value) // len(target_gpus)
                remainder = len(value) % len(target_gpus)
                
                scattered_images = []
                scattered_masks = []
                start_idx = 0
                for i, gpu_id in enumerate(target_gpus):
                    # Give remainder to first few GPUs
                    current_chunk_size = chunk_size + (1 if i < remainder else 0)
                    end_idx = start_idx + current_chunk_size
                    
                    # Get the chunk and move all tensors to the target device
                    chunk = value[start_idx:end_idx]
                    device_chunk = []
                    device_masks = []
                    
                    for sublist in chunk:
                        device_sublist = []
                        sublist_mask = []
                        
                        # Add real images and mark as valid (1)
                        for tensor in sublist:
                            if isinstance(tensor, torch.Tensor):
                                device_sublist.append(tensor.to(f'cuda:{gpu_id}'))
                                sublist_mask.append(1)
                            else:
                                device_sublist.append(tensor)
                                sublist_mask.append(1)
                        
                        # Pad sublist to max length with zeros and mark as invalid (0)
                        while len(device_sublist) < max_sublist_len:
                            if len(device_sublist) > 0 and isinstance(device_sublist[0], torch.Tensor):
                                # Create zero tensor with same shape as first tensor
                                zero_tensor = torch.zeros_like(device_sublist[0]).to(f'cuda:{gpu_id}')
                                device_sublist.append(zero_tensor)
                                sublist_mask.append(0)
                            else:
                                break  # Can't pad if we don't know tensor shape
                        
                        device_chunk.append(device_sublist)
                        device_masks.append(sublist_mask)
                    
                    scattered_images.append(device_chunk)
                    scattered_masks.append(device_masks)
                    start_idx = end_idx
                
                scattered_inputs[key] = scattered_images
                scattered_inputs['image_masks'] = scattered_masks
            else:
                # Use default scatter for other keys
                scattered_inputs[key] = scatter(value, target_gpus, dim)
        return [dict(zip(scattered_inputs.keys(), values)) 
                for values in zip(*scattered_inputs.values())]
    else:
        return scatter(inputs, target_gpus, dim)


class CustomDataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return custom_scatter(inputs, device_ids), \
               custom_scatter(kwargs, device_ids) if kwargs else [{}] * len(device_ids)


class TrainingParameters:
    def __init__(
            self,
            learning_rate=1e-6,
            warmup_ratio=0.1,
            apply_cosine_annealing=True,
            num_epochs=10,
            gradient_accumulation_steps=None,
            training_batch_size=4,
            validation_batch_size=4,
            min_allowed_batch_size=None,
            criterion=torch.nn.CrossEntropyLoss(),
            checkpoint_dir="checkpoint",
            last_checkpoint=None,
            moving_average_window_width=100,
            perform_intra_epoch_validation=False,
            intra_epoch_validation_step=1000,
            num_intra_epoch_validation_batches=500,
            num_steps_per_checkpoint=None,
            num_training_workers_per_gpu=4,
            num_validation_workers_per_gpu=4,
            save_visualizaton_data_during_training=False,
            save_visualizaton_data_during_validation=False,
            pause_on_validation_visualization=False):
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.apply_cosine_annealing = apply_cosine_annealing
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.min_allowed_batch_size = min_allowed_batch_size
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.last_checkpoint = last_checkpoint
        self.moving_average_window_width = moving_average_window_width
        self.perform_intra_epoch_validation = perform_intra_epoch_validation
        self.intra_epoch_validation_step = intra_epoch_validation_step
        self.num_intra_epoch_validation_batches = num_intra_epoch_validation_batches
        self.num_steps_per_checkpoint = num_steps_per_checkpoint
        self.num_training_workers_per_gpu = num_training_workers_per_gpu
        self.num_validation_workers_per_gpu = num_validation_workers_per_gpu
        self.save_visualizaton_data_during_training = save_visualizaton_data_during_training
        self.save_visualizaton_data_during_validation = save_visualizaton_data_during_validation
        self.pause_on_validation_visualization = pause_on_validation_visualization


class MlopsType(Enum):
    MLFLOW = "mlflow"
    WANDB = "wandb"


class MlopsParameters:
    def __init__(self, mlops_type, experiment_name, run_name=None, notes="", label_names=None, log_metric_step=100, send_notification=False, uri=None):
        try:
            self.mlops_type = MlopsType(mlops_type)
        except ValueError:
            raise ValueError(f"Unknown MLOps type: {mlops_type} (valid options are 'mlflow' and 'wandb')")

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.notes = notes
        self.label_names = label_names
        self.log_metric_step = log_metric_step
        self.send_notification = send_notification
        self.uri = uri


class DataWrapper:
    def __init__(self, data):
        self.__data = data
        self.__primary_tensor = self.__extract_primary_tensor(data)

    @property
    def shape(self):
        return self.__primary_tensor.shape if self.__primary_tensor is not None else None

    def size(self, dim):
        return self.__primary_tensor.size(dim) if self.__primary_tensor is not None else None

    def to(self, device):
        if isinstance(self.__data, torch.Tensor):
            return self.__data.to(device)
        elif isinstance(self.__data, dict):
            return {
                key: (
                    {sub_key: sub_value.to(device) if isinstance(sub_value, torch.Tensor) else sub_value
                    for sub_key, sub_value in value.items()} if isinstance(value, dict) or isinstance(value, Mapping)
                    else value.to(device) if isinstance(value, torch.Tensor)
                    else [[item.to(device) for item in sublist if isinstance(item, torch.Tensor)] for sublist in value] if isinstance(value, list)
                    else value
                )
                for key, value in self.__data.items()
            }
        return None

    def get_primary_tensor(self):
        return self.__primary_tensor

    def get_value(self, name):
        if isinstance(self.__data, dict) and name in self.__data:
            return self.__data[name]
        else:
            return None

    def __extract_primary_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, dict):
            if "images" in data and isinstance(data["images"], torch.Tensor):
                return data["images"]
            elif "images" in data and isinstance(data["images"], list):
                return torch.zeros(len(data["images"]))
            else:
                for value in data.values():
                    if isinstance(value, torch.Tensor):
                        return value
        return None


class TorchTrainingHelper:
    def __init__(self, model, dataset_helper, device, device_ids, training_parameters: TrainingParameters, mlops_parameters: MlopsParameters, multi_gpu_padding=False, **kwargs):
        self.__model = model
        self.__parallel_model = None
        self.__optimizer = None
        self.__dataset_helper = dataset_helper
        self.__train_data_loader = None
        self.__validation_data_loader = None
        self.__device = device
        self.__device_ids = device_ids
        self.__training_parameters = training_parameters
        self.__mlops_parameters = mlops_parameters
        self.__custom_parameters = kwargs
        self.__multi_gpu_padding = multi_gpu_padding

        # Dump training information.
        print("Creating TorchTrainingHelper")
        print(f"Selected device: {self.__device}")
        print(f"Selected device IDs: {self.__device_ids}")
        print("Using the following training parameters:")
        print(f"{self.__training_parameters.__dict__}")
        print("Using the following MLOps parameters:")
        if self.__mlops_parameters is not None:
            print(f"{self.__mlops_parameters.__dict__}")
        else:
            print("None")

        # Connect to MLOps.
        if self.__mlops_parameters is not None:
            self.__connect_to_mlops()

        # In order for the DataParallel to work correctly, all the data must always be on the first CUDA device.
        if self.__device_ids is not None:
            self.__device = "cuda:0"

        # Create checkpoint dir.
        os.makedirs(self.__training_parameters.checkpoint_dir, exist_ok=True)

    def __del__(self):
        # Disconnect from MLOps.
        if self.__mlops_parameters is not None:
            self.__disconnect_from_mlops()

    def start_training(self, collate_function_for_training=None, collate_function_for_validation=None):
        torch.cuda.empty_cache()

        # Create train data loader.
        self.__train_data_loader = self.__dataset_helper.get_torch_train_data_loader(
            collate_function=collate_function_for_training,
            batch_size=self.__training_parameters.training_batch_size,
            num_workers=self.__training_parameters.num_training_workers_per_gpu * len(self.__device_ids)
                if self.__device_ids is not None else self.__training_parameters.num_training_workers_per_gpu)

        # Create validation data loader.
        try:
            self.__validation_data_loader = self.__dataset_helper.get_torch_validation_data_loader(
                collate_function=collate_function_for_validation if collate_function_for_validation is not None else collate_function_for_training,
                batch_size=self.__training_parameters.validation_batch_size,
                num_workers=self.__training_parameters.num_validation_workers_per_gpu * len(self.__device_ids)
                    if self.__device_ids is not None else self.__training_parameters.num_validation_workers_per_gpu)
        except:
            self.__validation_data_loader = None

        # Create parallel model.
        if self.__multi_gpu_padding:
            self.__parallel_model = CustomDataParallel(self.__model, device_ids=self.__device_ids)
        else:
            self.__parallel_model = torch.nn.DataParallel(self.__model, device_ids=self.__device_ids)
        self.__parallel_model.to(self.__device)
        self.__optimizer = AdamW(self.__parallel_model.parameters(), lr=self.__training_parameters.learning_rate)

        # Create LR scheduler.
        total_steps = len(self.__train_data_loader) * self.__training_parameters.num_epochs
        if self.__training_parameters.warmup_ratio:
            warmup_steps = min(math.ceil(total_steps * self.__training_parameters.warmup_ratio), total_steps)
            print(f"Total training steps = Num batches x Num epochs = {len(self.__train_data_loader)} x {self.__training_parameters.num_epochs} = {total_steps}")
            print(f"Warmup steps = Total steps x Warmup ratio = {total_steps} x {self.__training_parameters.warmup_ratio} = {warmup_steps}")
        else:
            warmup_steps = None
        self.__lr_scheduler = training_utils.create_lr_scheduler(
            optimizer=self.__optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            apply_cosine_annealing=self.__training_parameters.apply_cosine_annealing)

        # Model statistics.
        self.__model_size_in_mib = training_utils.get_torch_model_size_in_mib(self.__parallel_model)
        self.__model_dtype = next(self.__parallel_model.parameters()).dtype
        self.__num_model_params = training_utils.get_num_torch_parameters(self.__parallel_model, requires_grad_only=False)
        print(f"Model size: {self.__model_size_in_mib:.2f} MiB")
        print(f"Model's dtype: {self.__model_dtype}")
        print(f"Num model params: {self.__num_model_params}")

        # Load checkpoint.
        if self.__training_parameters.last_checkpoint:
            print(f"Loading checkpoint {self.__training_parameters.last_checkpoint}")
            checkpoint = torch.load(self.__training_parameters.last_checkpoint)
            last_epoch = checkpoint["epoch"] - 1 # Convert epoch number from one-based to zero-based.
            start_epoch = last_epoch + 1  # Continue with the next epoch.
            self.__model.load_state_dict(checkpoint["model_state_dict"])
            self.__parallel_model.load_state_dict(checkpoint["parallel_model_state_dict"])
            self.__optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            start_epoch = 0

        print("Training started")

        # Log training parameters.
        if self.__mlops_parameters is not None:
            self.__log_params()

        # Training loop.
        for epoch in range(start_epoch, self.__training_parameters.num_epochs):
            print("")
            print("--------------------")
            print(f"EPOCH {epoch + 1}/{self.__training_parameters.num_epochs}")
            print("--------------------")

            # Train.
            print("")
            print("TRAINING")
            self.__train_epoch(epoch)

            # Save checkpoint.
            self.__save_checkpoint(epoch=epoch)

            # Per epoch validation.
            if self.__validation_data_loader is not None:
                print("")
                print("PER EPOCH VALIDATION")
                self.__validate(step=epoch, validation_type="Validation")

        print("")
        print("Training finished")

    def run_validation(self, collate_function_for_validation=None):
        torch.cuda.empty_cache()

        # Create validation data loader.
        self.__validation_data_loader = self.__dataset_helper.get_torch_validation_data_loader(
            collate_function=collate_function_for_validation,
            batch_size=self.__training_parameters.validation_batch_size,
            num_workers=self.__training_parameters.num_validation_workers_per_gpu * len(self.__device_ids)
                if self.__device_ids is not None else self.__training_parameters.num_validation_workers_per_gpu)

        # Create parallel model.
        if self.__multi_gpu_padding:
            self.__parallel_model = CustomDataParallel(self.__model, device_ids=self.__device_ids)
        else:
            self.__parallel_model = torch.nn.DataParallel(self.__model, device_ids=self.__device_ids)
        self.__parallel_model.to(self.__device)

        # Model statistics.
        self.__model_size_in_mib = training_utils.get_torch_model_size_in_mib(self.__parallel_model)
        self.__model_dtype = next(self.__parallel_model.parameters()).dtype
        self.__num_model_params = training_utils.get_num_torch_parameters(self.__parallel_model, requires_grad_only=False)
        print(f"Model size: {self.__model_size_in_mib:.2f} MiB")
        print(f"Model's dtype: {self.__model_dtype}")
        print(f"Num model params: {self.__num_model_params}")

        # Load checkpoint.
        if self.__training_parameters.last_checkpoint:
            print(f"Loading checkpoint {self.__training_parameters.last_checkpoint}")
            checkpoint = torch.load(self.__training_parameters.last_checkpoint)
            epoch = checkpoint["epoch"] - 1 # Convert epoch number from one-based to zero-based.
            self.__model.load_state_dict(checkpoint["model_state_dict"])
            self.__parallel_model.load_state_dict(checkpoint["parallel_model_state_dict"])
        else:
            epoch = 0

        # Log training parameters.
        if self.__mlops_parameters is not None:
            self.__log_params()

        print("VALIDATION")
        self.__validate(step=epoch, validation_type="Validation")

    def compute_metrics(self, probs, outputs, targets):
        assert self.__mlops_parameters is not None
        assert len(probs) == len(outputs) == len(targets)

        evaluation_metrics_calculator = EvaluationMetricsCalculator()
        for output, target in zip(outputs, targets):
            evaluation_metrics_calculator.add(output, target, skip_sigmoid=True)

        acc_precision, acc_recall, acc_f1, acc_accuracy = evaluation_metrics_calculator.get_accumulated_metrics()
        map = evaluation_metrics_calculator.get_macro_mean_average_precision(outputs, targets, skip_sigmoid=True)
        values = {
            f"Accumulated Precision": acc_precision,
            f"Accumulated Recall": acc_recall,
            f"Accumulated F1": acc_f1,
            f"Accumulated Accuracy": acc_accuracy,
            f"Macro Mean Average Precision": map["macro_mean_average_precision"],
            f"Average Precision Pos": map["average_precision_pos"],
            f"Average Precision Neg": map["average_precision_neg"]
        }
        self.__log_metric(values=values, step=0)

        # Log confusion matrix.
        calc = ConfusionMatrixCalculator()
        cm = calc.compute_confusion_matrix(y_true=targets, y_pred=outputs)
        plot = calc.create_plot(confusion_matrices=[cm], titles=["All labels"], grid_shape=(1, 1))
        self.__log_confusion_matrix(plot, "CM")

        # Log confusion matrices.
        cms = calc.compute_per_class_confusion_matrices(y_true=targets, y_pred=outputs)
        plot = calc.create_plot(confusion_matrices=cms, titles=self.__mlops_parameters.label_names)
        self.__log_confusion_matrix(plot, "Per-Label CMs")

        # Log ROC.
        calc = PerformanceCurveCalculator()
        roc = calc.compute_curve(curve_type=PerformanceCurveType.ROC, y_true=targets, y_prob=probs)
        plot = calc.create_plot(curves=[roc], titles=["All labels"], grid_shape=(1, 1))
        self.__log_performance_curve(plot, "ROC")

        # Log per-label ROCs.
        rocs = calc.compute_per_class_curves(curve_type=PerformanceCurveType.ROC, y_true=targets, y_prob=probs)
        plot = calc.create_plot(curves=rocs, titles=self.__mlops_parameters.label_names)
        self.__log_performance_curve(plot, "Per-Label ROCs")

        # Log PRC.
        prc = calc.compute_curve(curve_type=PerformanceCurveType.PRC, y_true=targets, y_prob=probs)
        titles = [f"{self.__mlops_parameters.run_name}\nAll labels"]
        plot = calc.create_plot(curves=[prc], titles=titles, grid_shape=(1, 1), show_grid=True, x_axis_markers=[0.8, 0.9], y_axis_markers=[0.8, 0.9])
        self.__log_performance_curve(plot, "PRC")

        # Log per-label PRCs.
        prcs = calc.compute_per_class_curves(curve_type=PerformanceCurveType.PRC, y_true=targets, y_prob=probs)
        if self.__mlops_parameters.label_names is not None:
            titles = [f"{self.__mlops_parameters.run_name}\n{label_name}" for label_name in self.__mlops_parameters.label_names]
        else:
            titles = [f"{self.__mlops_parameters.run_name}\n{None}"] * len(prcs)
        plot = calc.create_plot(curves=prcs, titles=titles, show_grid=True, x_axis_markers=[0.8, 0.9], y_axis_markers=[0.8, 0.9])
        self.__log_performance_curve(plot, "Per-Label PRCs")

        # Log scores distribution.
        gen = ScoresDistributionGenerator()
        plot = gen.create_plot(scores=probs, title="Scores distribution")
        self.__log_scores_distribution(plot, "Scores distribution")

    def save_model(self, model_file_name, parallel_model_file_name):
        torch.save(self.__parallel_model.module, model_file_name)
        print(f"Model saved as '{model_file_name}'")

        torch.save(self.__parallel_model, parallel_model_file_name)
        print(f"Parallel model saved as '{parallel_model_file_name}'")

    def save_model_weights(self, model_weights_file_name):
        torch.save(self.__parallel_model.module.state_dict(), model_weights_file_name)
        print(f"Model weights saved as '{model_weights_file_name}'")

    def get_device_ids_used(self):
        if self.__device_ids is None:
            return list(range(torch.cuda.device_count()))
        else:
            return self.__device_ids

    def __train_epoch(self, epoch):
        self.__parallel_model.train()

        losses = AverageMeter(moving_average_window_width=self.__training_parameters.moving_average_window_width)
        evaluation_metrics_calculator = EvaluationMetricsCalculator()
        tqdm_loader = tqdm(self.__train_data_loader)
        num_batches = len(self.__train_data_loader)

        for idx, batch in enumerate(tqdm_loader):
            # Skip step if batch is None.
            if batch is None:
                print("WARNING: Received null batch during training")
                continue

            # Compute step.
            step = epoch * num_batches + idx

            # Get the inputs.
            data, target = batch
            data = DataWrapper(data)

            # Skip step if batch size is less than min allowed batch size.
            batch_size = data.shape[0]
            if self.__training_parameters.min_allowed_batch_size and batch_size < self.__training_parameters.min_allowed_batch_size:
                print(f"Batch size {batch_size} < min allowed batch size {self.__training_parameters.min_allowed_batch_size}, skipping this training step")
                continue

            # Save visualization data.
            if self.__training_parameters.save_visualizaton_data_during_training:
                torch.save({"inputs": data.get_primary_tensor(), "labels": target}, "visualization_data.pt")

            # Forward pass.
            device_data = data.to(self.__device)
            if isinstance(device_data, dict):
                outputs = self.__parallel_model(**device_data)
            else:
                outputs = self.__parallel_model(device_data)

            # Get output.
            outputs = outputs["output"] if isinstance(outputs, dict) else outputs

            # Compute loss.
            loss = self.__training_parameters.criterion(outputs, target.to(self.__device))

            # If loss is a vector, it needs to be averaged.
            if loss.numel() > 1:
                loss = loss.mean()

            # Normalize loss if gradient accumulation is used.
            if self.__training_parameters.gradient_accumulation_steps is not None:
                loss = loss / self.__training_parameters.gradient_accumulation_steps

            # Calculate gradients.
            loss.backward()

            # Optimization step.
            if self.__training_parameters.gradient_accumulation_steps is None or idx % self.__training_parameters.gradient_accumulation_steps == 0:
                self.__optimizer.step()
                self.__optimizer.zero_grad()

            # LR scheduler step.
            self.__lr_scheduler.step()

            # Update data.
            losses.update(loss.item(), data.size(0))
            evaluation_metrics_calculator.add(outputs, target)
            tqdm_loader.set_description(
                f"Loss = {losses.val:.4f}, MA loss = {losses.moving_average:.4f}, AVG loss = {losses.avg:.4f}, LR = {self.__optimizer.param_groups[0]['lr']:.4e}")

            # Log MLOps metrics.
            if self.__mlops_parameters is not None and idx % self.__mlops_parameters.log_metric_step == 0:
                avg_precision, avg_recall, avg_f1, avg_accuracy = evaluation_metrics_calculator.get_average_metrics()
                values = {
                    "Training LR": self.__optimizer.param_groups[0]["lr"],
                    "Training MA Loss": losses.moving_average,
                    "Training AVG Loss": losses.avg,
                    "Training MA Precision": avg_precision,
                    "Training MA Recall": avg_recall,
                    "Training MA F1": avg_f1,
                    "Training MA Accuracy": avg_accuracy,
                }
                self.__log_metric(values, step)
                evaluation_metrics_calculator.reset()

            # Check if we need to save the checkpoint.
            if self.__training_parameters.num_steps_per_checkpoint is not None and step and step % self.__training_parameters.num_steps_per_checkpoint == 0:
                self.__save_checkpoint(epoch=epoch, step=step)

            # Intra-epoch validation.
            if (self.__training_parameters.perform_intra_epoch_validation and
                self.__validation_data_loader is not None and
                idx > 0 and
                idx % self.__training_parameters.intra_epoch_validation_step == 0):

                print("")
                print("")
                print("INTRA-EPOCH VALIDATION")
                self.__validate(step=step,
                                validation_type="Intra Epoch Validation",
                                num_batches=self.__training_parameters.num_intra_epoch_validation_batches)

                print("")
                print("")
                print("CONTINUE TRAINING")
                self.__parallel_model.train()

        # Log per-epoch MLOps metrics.
        if self.__mlops_parameters is not None:
            self.__log_metric({"Training Loss Per Epoch": losses.avg}, epoch)

    def __validate(self, step, validation_type, num_batches=None):
        self.__parallel_model.eval()

        validation_losses = AverageMeter(moving_average_window_width=self.__training_parameters.moving_average_window_width)
        evaluation_metrics_calculator = EvaluationMetricsCalculator()
        tqdm_loader = tqdm(self.__validation_data_loader)

        all_targets = torch.empty(0)
        all_outputs = torch.empty(0)
        all_embeddings = torch.empty(0)
        all_file_names = []

        pause = True

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_loader):
                # Skip step if batch is None.
                if batch is None:
                    print("WARNING: Received null batch during validation")
                    continue

                # Break earlier if necessary.
                if num_batches is not None and idx >= num_batches:
                    break

                # Get the inputs.
                data, target = batch
                data = DataWrapper(data)

                # Skip step if batch size is less than min allowed batch size.
                batch_size = data.shape[0]
                if self.__training_parameters.min_allowed_batch_size and batch_size < self.__training_parameters.min_allowed_batch_size:
                    print(f"Batch size {batch_size} < min allowed batch size {self.__training_parameters.min_allowed_batch_size}, skipping this validation step")
                    continue

                # Predict.
                device_data = data.to(self.__device)
                if isinstance(device_data, dict):
                    outputs = self.__parallel_model(**device_data)
                else:
                    outputs = self.__parallel_model(device_data)

                # Get embeddings and the output.
                embeddings = outputs["embeddings"] if isinstance(outputs, dict) and "embeddings" in outputs else None
                outputs = outputs["output"] if isinstance(outputs, dict) else outputs

                # Compute loss.
                loss = self.__training_parameters.criterion(outputs, target.to(self.__device))

                # Save visualization data.
                if self.__training_parameters.save_visualizaton_data_during_validation:
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).float()
                    torch.save({"inputs": data.get_primary_tensor(), "labels": predictions, "probabilities": probabilities}, "visualization_data.pt")

                    if self.__training_parameters.pause_on_validation_visualization and pause:
                        key = input("Press 'q' to skip pause or any other key to continue")
                        if key == "q":
                            pause = False

                # Stack all targets and outputs.
                all_targets = target if all_targets.numel() == 0 else torch.cat((all_targets, target), dim=0)
                all_outputs = outputs if all_outputs.numel() == 0 else torch.cat((all_outputs, outputs), dim=0)

                # Stack embeddings.
                if embeddings is not None:
                    all_embeddings = embeddings if all_embeddings.numel() == 0 else torch.cat((all_embeddings, embeddings), dim=0)

                # Add file names.
                file_names = data.get_value("file_names")
                if file_names is not None:
                    all_file_names.extend(file_names)

                # If 'loss' is a vector, it needs to be averaged.
                if loss.numel() > 1:
                    loss = loss.mean()

                # Updated validation losses.
                validation_losses.update(loss.item(), data.size(0))

                # Update evaluation metrics calculator.
                evaluation_metrics_calculator.add(outputs, target)

                # Refresh tqdm loader.
                tqdm_loader.set_description(f"Validation loss = {validation_losses.val:.4f}, " \
                                            f"validation MA loss = {validation_losses.moving_average:.4f}, " \
                                            f"validation AVG loss = {validation_losses.avg:.4f}")

        # Log MLOps metrics.
        if self.__mlops_parameters is not None:
            acc_precision, acc_recall, acc_f1, acc_accuracy = evaluation_metrics_calculator.get_accumulated_metrics()
            map = evaluation_metrics_calculator.get_macro_mean_average_precision(all_outputs, all_targets)
            values = {
                f"{validation_type} Accumulated Precision": acc_precision,
                f"{validation_type} Accumulated Recall": acc_recall,
                f"{validation_type} Accumulated F1": acc_f1,
                f"{validation_type} Accumulated Accuracy": acc_accuracy,
                f"{validation_type} Loss": validation_losses.avg,
                f"{validation_type} Macro Mean Average Precision": map["macro_mean_average_precision"],
                f"{validation_type} Average Precision Pos": map["average_precision_pos"],
                f"{validation_type} Average Precision Neg": map["average_precision_neg"]
            }
            self.__log_metric(values, step)

            if validation_type == "Validation":
                all_targets = all_targets.tolist()
                all_probs = torch.sigmoid(all_outputs).cpu().to(torch.float32)
                all_outputs = (all_probs > 0.5).int().tolist()

                # Log confusion matrix.
                calc = ConfusionMatrixCalculator()
                cm = calc.compute_confusion_matrix(y_true=all_targets, y_pred=all_outputs)
                plot = calc.create_plot(confusion_matrices=[cm], titles=["All labels"], grid_shape=(1, 1))
                self.__log_confusion_matrix(plot, "Validation CM")

                # Log confusion matrices.
                cms = calc.compute_per_class_confusion_matrices(y_true=all_targets, y_pred=all_outputs)
                plot = calc.create_plot(confusion_matrices=cms, titles=self.__mlops_parameters.label_names)
                self.__log_confusion_matrix(plot, "Validation Per-Label CMs")

                # Log ROC.
                calc = PerformanceCurveCalculator()
                roc = calc.compute_curve(curve_type=PerformanceCurveType.ROC, y_true=all_targets, y_prob=all_probs)
                plot = calc.create_plot(curves=[roc], titles=["All labels"], grid_shape=(1, 1))
                self.__log_performance_curve(plot, "Validation ROC")

                # Log per-label ROCs.
                rocs = calc.compute_per_class_curves(curve_type=PerformanceCurveType.ROC, y_true=all_targets, y_prob=all_probs)
                plot = calc.create_plot(curves=rocs, titles=self.__mlops_parameters.label_names)
                self.__log_performance_curve(plot, "Validation Per-Label ROCs")

                # Log PRC.
                prc = calc.compute_curve(curve_type=PerformanceCurveType.PRC, y_true=all_targets, y_prob=all_probs)
                plot = calc.create_plot(curves=[prc], titles=["All labels"], grid_shape=(1, 1))
                self.__log_performance_curve(plot, "Validation PRC")

                # Log per-label PRCs.
                prcs = calc.compute_per_class_curves(curve_type=PerformanceCurveType.PRC, y_true=all_targets, y_prob=all_probs)
                plot = calc.create_plot(curves=prcs, titles=self.__mlops_parameters.label_names)
                self.__log_performance_curve(plot, "Validation Per-Label PRCs")

                # Log scores distribution.
                if all_probs.shape[1] == 1:
                    gen = ScoresDistributionGenerator()
                    plot = gen.create_plot(scores=all_probs, title="Scores distribution")
                    self.__log_scores_distribution(plot, "Validation scores distribution")

                # Save prediction probabilities.
                if all_file_names:
                    self.__save_prediction_probs(epoch=step, file_names=all_file_names, targets=all_targets, outputs=all_outputs, probs=all_probs)

                # Save embeddings.
                self.__save_embeddings(embeddings=all_embeddings, epoch=step)

    def __save_checkpoint(self, epoch, step=None):
        checkpoint = {
            "epoch": epoch + 1,
            "step": None if step is None else step + 1,
            "parallel_model_state_dict": self.__parallel_model.state_dict(),
            "model_state_dict": self.__parallel_model.module.state_dict(),
            "optimizer_state_dict": self.__optimizer.state_dict(),
            "training_parameters_dict": self.__training_parameters.__dict__,
            "mlops_parameters_dict": self.__mlops_parameters.__dict__ if self.__mlops_parameters is not None else None
        }

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_utc"

        if step is None:
            torch.save(checkpoint, os.path.join(self.__training_parameters.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_{timestamp}.pt"))
        else:
            torch.save(checkpoint, os.path.join(self.__training_parameters.checkpoint_dir, f"checkpoint_step_{step + 1}_{timestamp}.pt"))

    def __save_prediction_probs(self, epoch, file_names, targets, outputs, probs):
        assert len(file_names) == len(targets) == len(outputs) == len(probs)

        prediction_probs_data = [
            {
                "file_name": file_names[i],
                "target": training_utils.convert_tensor(targets[i]),
                "output": training_utils.convert_tensor(outputs[i]),
                "prob": training_utils.convert_tensor(probs[i]),
                "hit": targets[i] == outputs[i]
            } for i in range(len(targets))
        ]

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_utc"
        prediction_probs_filename = os.path.join(self.__training_parameters.checkpoint_dir, f"prediction_probs_epoch_{epoch + 1}_{timestamp}.jsonl")

        with open(prediction_probs_filename, "w") as jsonl_file:
            for item in prediction_probs_data:
                jsonl_file.write(json.dumps(item) + "\n")

    def __save_embeddings(self, embeddings, epoch):
        if embeddings.numel() == 0:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_utc"
        embeddings_file = os.path.join(self.__training_parameters.checkpoint_dir, f"embeddings_epoch_{epoch + 1}_{timestamp}.pt")
        torch.save(embeddings, embeddings_file)

    def __connect_to_mlops(self):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            mlflow.set_tracking_uri(self.__mlops_parameters.uri)
            experiment = mlflow.get_experiment_by_name(self.__mlops_parameters.experiment_name)
            experiment_id = mlflow.create_experiment(self.__mlops_parameters.experiment_name) if experiment is None else experiment.experiment_id
            mlflow.set_experiment(self.__mlops_parameters.experiment_name)
            mlflow.set_tag("mlflow.note.content", self.__mlops_parameters.notes)
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.login()
            self.__wandb_run = wandb.init(project=self.__mlops_parameters.experiment_name, name=self.__mlops_parameters.run_name, notes=self.__mlops_parameters.notes)
            if self.__mlops_parameters.send_notification:
                self.__wandb_run.alert(title=self.__mlops_parameters.experiment_name, text=self.__mlops_parameters.notes)
            wandb.define_metric("Steps")
            wandb.define_metric("*", step_metric="Steps")
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")

    def __disconnect_from_mlops(self):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            pass
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            self.__wandb_run.finish()
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")

    def __log_params(self):
        extra_params = {
            "num_model_params": self.__num_model_params,
            "model_size_in_mib": self.__model_size_in_mib,
            "model_dtype": self.__model_dtype,
            "training_dataset_size": len(self.__train_data_loader.dataset) if self.__train_data_loader is not None else None,
            "validation_dataset_size": len(self.__validation_data_loader.dataset) if self.__validation_data_loader is not None else None
        }

        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            mlflow.log_params(self.__training_parameters.__dict__)
            mlflow.log_params(self.__mlops_parameters.__dict__)
            mlflow.log_params(self.__custom_parameters)
            mlflow.log_params(extra_params)
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.config.update(self.__training_parameters.__dict__)
            wandb.config.update(self.__mlops_parameters.__dict__)
            wandb.config.update(self.__custom_parameters)
            wandb.config.update(extra_params)
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")

    def __log_confusion_matrix(self, confusion_matrix_plot, title):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            print("Logging confusion matrix is not supported for MLflow")
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.log({title: wandb.Image(confusion_matrix_plot)})
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")

    def __log_performance_curve(self, performance_curve_plot, title):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            print("Logging performance curve is not supported for MLflow")
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.log({title: wandb.Image(performance_curve_plot)})
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")

    def __log_scores_distribution(self, scores_distribution_plot, title):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            print("Logging scores distribution is not supported for MLflow")
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.log({title: wandb.Image(scores_distribution_plot)})
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")

    def __log_metric(self, values, step):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            for key, value in values.items():
                mlflow.log_metric(key, f"{value:.4f}", step=step)
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            values["Steps"] = step
            wandb.log(values)
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")


class AverageMeter(object):
    def __init__(self, moving_average_window_width):
        self.__moving_average_window_width = moving_average_window_width
        self.reset()

    def reset(self):
        self.val = 0
        self.moving_average = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # Update moving average.
        self.history.extend([val] * n)
        if len(self.history) > self.__moving_average_window_width:
            self.history = self.history[-self.__moving_average_window_width:]
        self.moving_average = sum(self.history) / len(self.history)
