import inspect
import json
import math
import os
from datetime import datetime
from enum import Enum

import mlflow
import torch
import wandb
from torch.optim import AdamW
from tqdm.auto import tqdm

from epsutils.training import training_utils
from epsutils.training.confusion_matrix_calculator import ConfusionMatrixCalculator
from epsutils.training.evaluation_metrics_calculator import EvaluationMetricsCalculator
from epsutils.training.scores_distribution_generator import ScoresDistributionGenerator


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
            self.__validatation_data_loader = self.__dataset_helper.get_torch_validation_data_loader(
                collate_function=collate_function_for_validation if collate_function_for_validation is not None else collate_function_for_training,
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
            if self.__validatation_data_loader is not None:
                print("")
                print("PER EPOCH VALIDATION")
                self.__validate(step=epoch, validation_type="Validation")

        print("")
        print("Training finished")

    def run_validation(self, collate_function_for_validation=None):
        torch.cuda.empty_cache()

        # Create validation data loader.
        self.__validatation_data_loader = self.__dataset_helper.get_torch_validation_data_loader(
            collate_function=collate_function_for_validation,
            batch_size=self.__training_parameters.validation_batch_size,
            num_workers=self.__training_parameters.num_validation_workers_per_gpu * len(self.__device_ids)
                if self.__device_ids is not None else self.__training_parameters.num_validation_workers_per_gpu)

        # Create parallel model.
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
            if len(batch) == 2:
                data, target = batch
            elif len(batch) == 3:
                data, target, _ = batch
            else:
                raise ValueError("Unexpected batch format")

            # Skip step if data is None.
            if data is None:
                print("WARNING: Received null batch during training")
                continue

            # Skip step if batch size is less than min allowed batch size.
            batch_size = data.shape[0]
            if self.__training_parameters.min_allowed_batch_size and batch_size < self.__training_parameters.min_allowed_batch_size:
                print(f"Batch size {batch_size} < min allowed batch size {self.__training_parameters.min_allowed_batch_size}, skipping this training step")
                continue

            # Save visualization data.
            if self.__training_parameters.save_visualizaton_data_during_training:
                torch.save({"inputs": data, "labels": target}, "visualization_data.pt")

            # Forward pass.
            if self.__is_multi_parameter_model:
                outputs = self.__parallel_model(data.to(self.__device), target.to(self.__device))
                loss = outputs.loss
            else:
                outputs = self.__parallel_model(data.to(self.__device))
                outputs = outputs["output"] if isinstance(outputs, dict) else outputs
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

        all_targets = torch.empty(0)
        all_outputs = torch.empty(0)
        all_embeddings = torch.empty(0)
        all_file_names = []

        pause = True

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_loader):
                if num_batches is not None and idx >= num_batches:
                    break

                # Get the inputs.
                if len(batch) == 2:
                    data, target = batch
                elif len(batch) == 3:
                    data, target, file_names = batch
                else:
                    raise ValueError("Unexpected batch format")

                # Skip step if data is None.
                if data is None:
                    print("WARNING: Received null batch during validation")
                    continue

                # Skip step if batch size is less than min allowed batch size.
                batch_size = data.shape[0]
                if self.__training_parameters.min_allowed_batch_size and batch_size < self.__training_parameters.min_allowed_batch_size:
                    print(f"Batch size {batch_size} < min allowed batch size {self.__training_parameters.min_allowed_batch_size}, skipping this validation step")
                    continue

                # Predict.
                if self.__is_multi_parameter_model:
                    outputs = self.__parallel_model(data.to(self.__device), target.to(self.__device))
                    embeddings = None
                    loss = outputs.loss
                else:
                    outputs = self.__parallel_model(data.to(self.__device))
                    embeddings = outputs["embeddings"] if isinstance(outputs, dict) and "embeddings" in outputs else None
                    outputs = outputs["output"] if isinstance(outputs, dict) else outputs
                    loss = self.__training_parameters.criterion(outputs, target.to(self.__device))

                # Save visualization data.
                if self.__training_parameters.save_visualizaton_data_during_validation:
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).float()
                    torch.save({"inputs": data, "labels": predictions, "probabilities": probabilities}, "visualization_data.pt")

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
                if len(batch) == 3:
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
            values = {
                f"{validation_type} Accumulated Precision": acc_precision,
                f"{validation_type} Accumulated Recall": acc_recall,
                f"{validation_type} Accumulated F1": acc_f1,
                f"{validation_type} Accumulated Accuracy": acc_accuracy,
                f"{validation_type} Loss": validation_losses.avg
            }
            self.__log_metric(values, step)

            if validation_type == "Validation":
                all_targets = all_targets.tolist()
                all_probs = torch.sigmoid(all_outputs)
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

                # Log scores distribution.
                if all_probs.shape[1] == 1:
                    gen = ScoresDistributionGenerator()
                    scores = all_probs.cpu().to(torch.float32)
                    plot = gen.create_plot(scores=scores, title="Scores distribution")
                    self.__log_scores_distribution(plot, "Validation scores distribution")

                # Save a list of misclassified samples.
                if all_file_names:
                    self.__save_misclassified(epoch=step, file_names=all_file_names, targets=all_targets, outputs=all_outputs, probs=all_probs)

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
            "mlops_parameters_dict": self.__mlops_parameters.__dict__
        }

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_utc"

        if step is None:
            torch.save(checkpoint, os.path.join(self.__training_parameters.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_{timestamp}.pt"))
        else:
            torch.save(checkpoint, os.path.join(self.__training_parameters.checkpoint_dir, f"checkpoint_step_{step + 1}_{timestamp}.pt"))

    def __save_misclassified(self, epoch, file_names, targets, outputs, probs):
        assert len(file_names) == len(targets) == len(outputs) == len(probs)

        misclassified = [
            {
                "file_name": file_names[i],
                "target": training_utils.convert_tensor(targets[i]),
                "output": training_utils.convert_tensor(outputs[i]),
                "probs": training_utils.convert_tensor(probs[i])
            } for i in range(len(targets)) if targets[i] != outputs[i]
        ]

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_utc"
        misclassified_file = os.path.join(self.__training_parameters.checkpoint_dir, f"misclassified_epoch_{epoch + 1}_{timestamp}.jsonl")

        with open(misclassified_file, "w") as jsonl_file:
            for item in misclassified:
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
            run = wandb.init(project=self.__mlops_parameters.experiment_name, name=self.__mlops_parameters.run_name, notes=self.__mlops_parameters.notes)
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
            mlflow.log_params({"num_model_params": self.__num_model_params, "model_size_in_mib": self.__model_size_in_mib, "model_dtype": self.__model_dtype})
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.config.update(self.__training_parameters.__dict__)
            wandb.config.update(self.__mlops_parameters.__dict__)
            wandb.config.update({"num_model_params": self.__num_model_params, "model_size_in_mib": self.__model_size_in_mib, "model_dtype": self.__model_dtype})
        else:
            raise ValueError(f"Unsupported MLOps type {self.__mlops_parameters.mlops_type}")

    def __log_confusion_matrix(self, confusion_matrix_plot, title):
        if self.__mlops_parameters.mlops_type == MlopsType.MLFLOW:
            print("Logging confusion matrix is not supported for MLflow")
        elif self.__mlops_parameters.mlops_type == MlopsType.WANDB:
            wandb.log({title: wandb.Image(confusion_matrix_plot)})
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
