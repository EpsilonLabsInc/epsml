import os

import torch
from transformers import TrainingArguments, Trainer

from epsutils.training import training_utils


class TrainingParameters:
    def __init__(self, learning_rate=1e-6, num_epochs=10, batch_size=4, checkpoint_dir="checkpoint"):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir


class HuggingFaceTrainingHelper:
    def __init__(self, model, dataset_helper, device, device_ids, training_parameters: TrainingParameters):

        self.__model = model
        self.__parallel_model = None
        self.__dataset_helper = dataset_helper
        self.__device = device
        self.__device_ids = device_ids
        self.__training_parameters = training_parameters

        # In order for the DataParallel to work correctly, all the data must always be on the first CUDA device.
        if self.__device_ids is not None:
            self.__device = "cuda:0"

        # Create checkpoint dir.
        os.makedirs(self.__training_parameters.checkpoint_dir, exist_ok=True)

    def start_training(self, collate_function):
        torch.cuda.empty_cache()

        self.__parallel_model = torch.nn.DataParallel(self.__model, device_ids=self.__device_ids)
        self.__parallel_model.to(self.__device)

        args = TrainingArguments(
                    num_train_epochs=self.__training_parameters.num_epochs,
                    remove_unused_columns=False,
                    per_device_train_batch_size=self.__training_parameters.batch_size,
                    gradient_accumulation_steps=8,
                    warmup_steps=5,
                    learning_rate=self.__training_parameters.learning_rate,
                    weight_decay=1e-5,
                    adam_beta2=0.999,
                    logging_steps=100,
                    optim="adamw_torch",
                    save_strategy="steps",
                    save_steps=100,
                    push_to_hub=False,
                    save_total_limit=1,
                    output_dir=self.__training_parameters.checkpoint_dir,
                    bf16=True,
                    report_to=["tensorboard"],
                    dataloader_pin_memory=False)

        trainer = Trainer(model=self.__parallel_model,
                          train_dataset=self.__dataset_helper.get_hugging_face_train_dataset(),
                          data_collator=collate_function,
                          args=args)

        print("Training started")

        model_size_in_mib = training_utils.get_torch_model_size_in_mib(self.__parallel_model)
        model_dtype = next(self.__parallel_model.parameters()).dtype
        print(f"Model size: {model_size_in_mib:.2f} MiB")
        print(f"Model's dtype: {model_dtype}")

        trainer.train()

        print("Training finished")

    def save_model(self, model_file_name, parallel_model_file_name):
        torch.save(self.__parallel_model.module, model_file_name)
        print(f"Model saved as '{model_file_name}'")

        torch.save(self.__parallel_model, parallel_model_file_name)
        print(f"Parallel model saved as '{parallel_model_file_name}'")
