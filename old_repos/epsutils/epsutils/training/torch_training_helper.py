import inspect
import math
import os
from enum import Enum

import mlflow
import torch
import wandb
from torch.optim import AdamW
from tqdm.auto import tqdm

from epsutils.training import training_utils
from epsutils.training.evaluation_metrics_calculator import EvaluationMetricsCalculator
from datetime import datetime


class TrainingParameters:
    def __init__(
            self,
            learning_rate=1e-6,
            warmup_ratio=0.1,
            num_epochs=10,
            gradient_accumulation_steps=None,
            training_batch_size=4,
            validation_batch_size=4,
            criterion=torch.nn.CrossEntropyLoss(),
            checkpoint_dir="checkpoint",
            moving_average_window_width=100,
            perform_intra_epoch_validation=False,
            intra_epoch_validation_step=1000,
            num_intra_epoch_validation_batches=500,
            num_steps_per_checkpoint=None,
            num_training_workers_per_gpu=4,
            num_validation_workers_per_gpu=4
            ):
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.moving_average_window_width = moving_average_window_width
        self.perform_intra_epoch_validation = perform_intra_epoch_validation
        self.intra_epoch_validation_step = intra_epoch_validation_step
        self.num_intra_epoch_validation_batches = num_intra_epoch_validation_batches
        self.num_steps_per_checkpoint = num_steps_per_checkpoint
        self.num_training_workers_per_gpu = num_training_workers_per_gpu
        self.num_validation_workers_per_gpu = num_validation_workers_per_gpu


class MlopsType(Enum):
    MLFLOW = "mlflow"
    WANDB = "wandb"


class MlopsParameters:
    def __init__(self, mlops_type, experiment_name, notes="", log_metric_step=100, send_notification=False, uri=None):
        try:
            self.mlops_type = MlopsType(mlops_type)
        except ValueError:
            raise ValueError(f"Unknown MLOps type: {mlops_type} (valid options are 'mlflow' and 'wandb')")

        self.experiment_name = experiment_name
        self.notes = notes
        self.log_metric_step = log_metric_step
        self.send_notification = send_notification
        self.uri = uri


class TorchTrainingHelper:
    def __init__(self, model, dataset_helper, device, device_ids, training_parameters: TrainingParameters, mlops_parameters: MlopsParameters):
        self.__model = model
        self.__parallel_model = None
        self.__dataset_helper = dataset_helper
        self.__device = device
        self.__device_ids = device_ids
        self.__training_parameters = training_parameters
        self.__mlops_parameters = mlops_parameters

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

        # Determine whether the model is single- or multi-parameter.
        forward_method = self.__model.forward
        signature = inspect.signature(forward_method)
        parameters = list(signature.parameters.keys())
        self.__is_multi_parameter_model = len(parameters) > 1

        # Connect to MLOps.
        if self.__mlops_parameters is not None:
            self.__connect_to_mlops()

        # In order for the DataParallel to work correctly, all the data must always be on the first CUDA device.
        if self.__device_ids is not None:
            self.__device = "cuda:0"

        # Create checkpoint dir.
        os.makedirs(self.__training_parameters.checkpoint_dir, exist_ok=True)

    def start_training(self, collate_function):
        torch.cuda.empty_cache()

        # Create train data loader.
        self.__train_data_loader = self.__dataset_helper.get_torch_train_data_loader(
            collate_function=collate_function,
            batch_size=self.__training_parameters.training_batch_size,
            num_workers=self.__training_parameters.num_training_workers_per_gpu * len(self.__device_ids)
                if self.__device_ids is not None else self.__training_parameters.num_training_workers_per_gpu)

        # Create validation data loader.
        try:
            self.__validatation_data_loader = self.__dataset_helper.get_torch_validation_data_loader(
                collate_function=collate_function,
                batch_size=self.__training_parameters.validation_batch_size,
                num_workers=self.__training_parameters.num_validation_workers_per_gpu * len(self.__device_ids)
                    if self.__device_ids is not None else self.__training_parameters.num_validation_workers_per_gpu)
        except:
            self.__validatation_data_loader = None

        # Create parallel model.
        self.__parallel_model = torch.nn.DataParallel(self.__model, device_ids=self.__device_ids)
        self.__parallel_model.to(self.__device)
        self.__optimizer = AdamW(self.__parallel_model.parameters(), lr=self.__training_parameters.learning_rate)

        # Create LR scheduler.
        total_steps = len(self.__train_data_loader) * self.__training_parameters.num_epochs
        warmup_steps = min(math.ceil(total_steps * self.__training_parameters.warmup_ratio), total_steps)
        print(f"Total training steps = Num batches x Num epochs = {len(self.__train_data_loader)} x {self.__training_parameters.num_epochs} = {total_steps}")
        print(f"Warmup steps = Total steps x Warmup ratio = {total_steps} x {self.__training_parameters.warmup_ratio} = {warmup_steps}")
        self.__lr_scheduler = training_utils.create_lr_scheduler_with_warmup_and_cosine_annealing(
            optimizer=self.__optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

        # Model statistics.
        self.__model_size_in_mib = training_utils.get_torch_model_size_in_mib(self.__parallel_model)
        self.__model_dtype = next(self.__parallel_model.parameters()).dtype
        num_params = training_utils.get_num_torch_parameters(self.__parallel_model, requires_grad_only=False)
        print(f"Model size: {self.__model_size_in_mib:.2f} MiB")
        print(f"Model's dtype: {self.__model_dtype}")
        print(f"Num model params: {num_params}")

        print("Training started")

        # Log training parameters.
        if self.__mlops_parameters is not None:
            self.__log_params()

        # Training loop.
        for epoch in range(self.__training_parameters.num_epochs):
            print("")
            print("--------------------")
            print(f"EPOCH {epoch + 1}/{self.__training_parameters.num_epochs}")
            print("--------------------")

            # Train.
            print("")
            print("TRAINING")
            self.__train_epoch(epoch)

            # Save checkpoint.
            checkpoint = {
                "epoch": epoch + 1,
                "parallel_model_state_dict": self.__parallel_model.state_dict(),
                "model_state_dict": self.__parallel_model.module.state_dict(),
                "optimizer_state_dict": self.__optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(self.__training_parameters.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

            # Per epoch validation.
            if self.__validatation_data_loader is not None:
                print("")
                print("PER EPOCH VALIDATION")
                self.__validate(step=epoch, validation_type="Validation")

        print("")
        print("Training finished")

    def save_model(self, model_file_name, parallel_model_file_name):
        torch.save(self.__parallel_model.module, model_file_name)
        print(f"Model saved as '{model_file_name}'")

        torch.save(self.__parallel_model, parallel_model_file_name)
        print(f"Parallel model saved as '{parallel_model_file_name}'")

    def save_model_weights(self, model_weights_file_name):
        torch.save(self.__parallel_model.module.state_dict(), model_weights_file_name)
        print(f"Model weights saved as '{model_weights_file_name}'")

    def __train_epoch(self, epoch):
        self.__parallel_model.train()

        losses = AverageMeter(moving_average_window_width=self.__training_parameters.moving_average_window_width)
        evaluation_metrics_calculator = EvaluationMetricsCalculator()
        tqdm_loader = tqdm(self.__train_data_loader)
        num_batches = len(self.__train_data_loader)

        for idx, batch in enumerate(tqdm_loader):
            step = epoch * num_batches + idx

            # Get the inputs.
            data, target = batch

            # Forward pass.
            if self.__is_multi_parameter_model:
                outputs = self.__parallel_model(data.to(self.__device), target.to(self.__device))
                loss = outputs.loss
            else:
                outputs = self.__parallel_model(data.to(self.__device))
                loss = self.__training_parameters.criterion(outputs, target.to(self.__device))

            # If 'loss' is a vector, it needs to be averaged.
            if loss.numel() > 1:
                loss = loss.mean()

            # Gradient accumulation.
            if self.__training_parameters.gradient_accumulation_steps is not None:
                loss = loss / self.__training_parameters.gradient_accumulation_steps

            # Update losses.
            losses.update(loss.item(), data.size(0))

            # Check if we need to checkpoint.
            if self.__training_parameters.num_steps_per_checkpoint is not None and step and step % self.__training_parameters.num_steps_per_checkpoint == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "parallel_model_state_dict": self.__parallel_model.state_dict(),
                    "model_state_dict": self.__parallel_model.module.state_dict(),
                    "optimizer_state_dict": self.__optimizer.state_dict()
                }
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                torch.save(checkpoint, os.path.join(self.__training_parameters.checkpoint_dir, f"checkpoint_step_{step}_{timestamp}.pt"))

            # Update evaluation metrics calculator.
            evaluation_metrics_calculator.add(outputs, target)

            # Refresh tqdm loader.
            tqdm_loader.set_description(
                f"Loss = {losses.val:.4f}, MA loss = {losses.moving_average:.4f}, " \
                f"AVG loss = {losses.avg:.4f}, LR = {self.__optimizer.param_groups[0]['lr']:.4e}")

            # Log intermediate MLOps metrics.
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

            # Calculate gradients.
            loss.backward()

            # Optimization step.
            if self.__training_parameters.gradient_accumulation_steps is None or idx % self.__training_parameters.gradient_accumulation_steps == 0:
                self.__optimizer.step()
                self.__optimizer.zero_grad()

            # LR scheduler step.
            self.__lr_scheduler.step()

            # Intra-epoch validation.
            if (self.__training_parameters.perform_intra_epoch_validation and
                self.__validatation_data_loader is not None and
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
        tqdm_loader = tqdm(self.__validatation_data_loader)

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_loader):
                if num_batches is not None and idx >= num_batches:
                    break

                # Get the inputs.
                data, target = batch

                # Predict.
                if self.__is_multi_parameter_model:
                    outputs = self.__parallel_model(data.to(self.__device), target.to(self.__device))
                    loss = outputs.loss
                else:
                    outputs = self.__parallel_model(data.to(self.__device))
                    loss = self.__training_parameters.criterion(outputs, target.to(self.__device))

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
            values = {
                f"{validation_type} Accumulated Precision": acc_precision,
                f"{validation_type} Accumulated Recall": acc_recall,
                f"{validation_type} Accumulated F1": acc_f1,
                f"{validation_type} Accumulated Accuracy": acc_accuracy,
                f"{validation_type} Loss": validation_losses.avg
            }
            self.__log_metric(values, step)

    def __connect_to_mlops(self):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            mlflow.set_tracking_uri(self.__mlops_parameters.uri)
            experiment = mlflow.get_experiment_by_name(self.__mlops_parameters.experiment_name)
            experiment_id = mlflow.create_experiment(self.__mlops_parameters.experiment_name) if experiment is None else experiment.experiment_id
            mlflow.set_experiment(self.__mlops_parameters.experiment_name)
            mlflow.set_tag("mlflow.note.content", self.__mlops_parameters.notes)
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.login()
            run = wandb.init(project=self.__mlops_parameters.experiment_name, notes=self.__mlops_parameters.notes)
            if self.__mlops_parameters.send_notification:
                run.alert(title=self.__mlops_parameters.experiment_name, text=self.__mlops_parameters.notes)
            wandb.define_metric("Steps")
            wandb.define_metric("*", step_metric="Steps")
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")

    def __log_params(self):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            mlflow.log_params(self.__training_parameters.__dict__)
            mlflow.log_params(self.__mlops_parameters.__dict__)
            mlflow.log_params({"model_size_in_mib": self.__model_size_in_mib, "model_dtype": self.__model_dtype})
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.config.update(self.__training_parameters.__dict__)
            wandb.config.update(self.__mlops_parameters.__dict__)
            wandb.config.update({"model_size_in_mib": self.__model_size_in_mib, "model_dtype": self.__model_dtype})
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
