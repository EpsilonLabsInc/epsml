from abc import ABC, abstractmethod

from IPython.display import clear_output


class BaseDatasetHelper(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._load_dataset(*args, **kwargs)

    @abstractmethod
    def _load_dataset(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_pil_image(self, item):
        pass

    @abstractmethod
    def get_torch_image(self, item, processor):
        pass

    @abstractmethod
    def get_labels(self):
        pass

    @abstractmethod
    def get_ids_to_labels(self):
        pass

    @abstractmethod
    def get_labels_to_ids(self):
        pass

    @abstractmethod
    def get_torch_label(self, item):
        pass

    @abstractmethod
    def get_pandas_full_dataset(self):
        pass

    @abstractmethod
    def get_hugging_face_train_dataset(self):
        pass

    @abstractmethod
    def get_hugging_face_validation_dataset(self):
        pass

    @abstractmethod
    def get_hugging_face_test_dataset(self):
        pass

    @abstractmethod
    def get_torch_train_dataset(self):
        pass

    @abstractmethod
    def get_torch_validation_dataset(self):
        pass

    @abstractmethod
    def get_torch_test_dataset(self):
        pass

    @abstractmethod
    def get_torch_train_data_loader(self, collate_function, batch_size, num_workers):
        pass

    @abstractmethod
    def get_torch_validation_data_loader(self, collate_function, batch_size, num_workers):
        pass

    @abstractmethod
    def get_torch_test_data_loader(self, collate_function, batch_size, num_workers):
        pass

    def view_full_dataset(self):
        print("Viewing full dataset")

        df = self.get_pandas_full_dataset()
        column_names_list = df.columns.tolist()
        i = 0

        while 0 <= i < len(df):
            clear_output()

            item = df.iloc[i]
            print(f"Item {i + 1}/{len(df)}:")

            for column_name in column_names_list:
                if "image" in column_name.lower():
                    image = self.get_pil_image(item)
                    image.show()
                else:
                    print(f"- {column_name}: {item[column_name]}")

            key = input("Press a key to continue...")

            if key == 'q':
                clear_output()
                print("Done")
                break
            elif key == 'p':  # 'p' as 'previous'.
                i = max(0, i - 1)
            else:
                i += 1
