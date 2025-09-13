import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        shape_code_folder_name: str,
        image_folder_name: str,
        transform,
        max_length: int = 8192,
        split: str = "train",
        dtype=torch.float32,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.transform = transform
        self.max_length = max_length
        self.split = split
        self.dtype = dtype

        self.shape_code_folder_path = (
            self.dataset_root_folder_path + shape_code_folder_name + "/"
        )
        self.image_root_folder_path = (
            self.dataset_root_folder_path + image_folder_name + "/"
        )

        assert os.path.exists(self.shape_code_folder_path)
        assert os.path.exists(self.image_root_folder_path)

        self.output_error = False

        self.invalid_image_file_path_list = []

        self.paths_list = []

        print("[INFO][ImageDataset::__init__]")
        print("\t start load mash and image datasets...")
        for root, _, files in os.walk(self.image_root_folder_path):
            if len(files) == 0:
                continue

            rel_folder_path = os.path.relpath(root, self.image_root_folder_path)

            shape_code_file_path = (
                self.shape_code_folder_path + rel_folder_path + ".npy"
            )

            if not os.path.exists(shape_code_file_path):
                continue

            image_file_path_list = []
            for file in files:
                if not file.endswith(".jpg") or file.endswith("_tmp.jpg"):
                    continue

                image_file_path_list.append(root + "/" + file)

            if len(image_file_path_list) == 0:
                continue

            image_file_path_list.sort()

            self.paths_list.append([shape_code_file_path, image_file_path_list])

        self.paths_list.sort(key=lambda x: x[0])

        return

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        shape_code_file_path, image_file_path_list = self.paths_list[index]

        if not os.path.exists(shape_code_file_path):
            if self.output_error:
                print("[ERROR][ImageDataset::__getitem__]")
                print("\t this shape code file is not valid!")
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        image_file_idx = np.random.choice(len(image_file_path_list))

        image_file_path = image_file_path_list[image_file_idx]

        if image_file_path in self.invalid_image_file_path_list:
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        try:
            image = Image.open(image_file_path)
        except KeyboardInterrupt:
            print("[INFO][imageDataset::__getitem__]")
            print("\t stopped by the user (Ctrl+C).")
            exit()
        except Exception as e:
            if self.output_error:
                print("[ERROR][imageDataset::__getitem__]")
                print("\t this npy file is not valid!")
                print("\t image_file_path:", image_file_path)
                print("\t error info:", e)

            self.invalid_image_file_path_list.append(image_file_path)
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        image = image.convert("RGB")

        image = self.transform(image)

        shape_code = np.load(shape_code_file_path)
        assert shape_code is not None, (
            "[ERROR][ImageDataset::__getitem__] shape_code_params is None!"
        )

        expand_shape_code = shape_code
        if expand_shape_code.shape[0] < self.max_length:
            expand_shape_code = np.ones([self.max_length], dtype=np.int64) * 256
            expand_shape_code[: shape_code.shape[0]] = shape_code

        expand_shape_code = torch.from_numpy(expand_shape_code).to(torch.int64)

        input_shape_code = expand_shape_code[: self.max_length - 1]
        next_shape_code = expand_shape_code[1 : self.max_length]

        data = {
            "input_shape_code": input_shape_code,
            "next_shape_code": next_shape_code,
            "image": image.to(self.dtype),
        }

        return data
