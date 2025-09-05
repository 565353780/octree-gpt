import os
import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from dino_v2_detect.Module.detector import Detector as DINODetector

from octree_gpt.Dataset.image import ImageDataset
from octree_gpt.Model.hy3d_gpt import HY3DGPT


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_amp: bool = False,
        quick_test: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.context_dim = 1024
        self.n_heads = 8  # 16
        self.d_head = 64
        self.depth = 8  # 16
        self.depth_single_blocks = 16  # 32

        self.gt_sample_added_to_logger = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=257)

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_amp,
            quick_test,
        )
        return

    def createDatasets(self) -> bool:
        model_type = "large"
        model_file_path = "./data/dinov2_vitl14_reg4_pretrain.pth"
        dtype = "auto"

        if not os.path.exists(model_file_path):
            print("[ERROR][BaseDiffusionTrainer::createDatasets]")
            print("\t DINOv2 model not found!")
            print("\t model_file_path:", model_file_path)
            exit()

        self.dino_detector = DINODetector(
            model_type, model_file_path, dtype, self.device
        )

        eval = True
        self.dataloader_dict["dino"] = {
            "dataset": ImageDataset(
                self.dataset_root_folder_path,
                "Objaverse_82K/shape_code",
                "Objaverse_82K/render_jpg_v2",
                self.dino_detector.transform,
                8192,
                "train",
                self.dtype,
            ),
            "repeat_num": 1,
        }

        if eval:
            self.dataloader_dict["eval"] = {
                "dataset": ImageDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/shape_code",
                    "Objaverse_82K/render_jpg_v2",
                    self.dino_detector.transform,
                    8192,
                    "eval",
                    self.dtype,
                ),
            }

        if "eval" in self.dataloader_dict.keys():
            self.dataloader_dict["eval"]["dataset"].paths_list = self.dataloader_dict[
                "eval"
            ]["dataset"].paths_list[:4]

        return True

    def createModel(self) -> bool:
        self.model = HY3DGPT(
            context_dim=self.context_dim,
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth,
            depth_single_blocks=self.depth_single_blocks,
        ).to(self.device, dtype=self.dtype)
        return True

    def getCondition(self, data_dict: dict) -> dict:
        if "image" in data_dict.keys():
            image = data_dict["image"]
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.to(self.device)

            dino_feature = self.dino_detector.detect(image)

            data_dict["condition"] = dino_feature
        elif "embedding" in data_dict.keys():
            embedding = data_dict["embedding"]

            if embedding.ndim == 1:
                embedding = embedding.view(1, 1, -1)
            if embedding.ndim == 2:
                embedding = embedding.unsqueeze(1)
            elif embedding.ndim == 4:
                embedding = torch.squeeze(embedding, dim=1)

            data_dict["condition"] = embedding.to(self.device)
        else:
            print("[ERROR][BaseDiffusionTrainer::getCondition]")
            print("\t valid condition type not found!")
            exit()

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        gt_next_shape_code = data_dict["next_shape_code"]
        pred_next_shape_code = result_dict["next_shape_code"]

        gt = gt_next_shape_code.view(-1)  # (seq_len, vocab_size)
        pred = pred_next_shape_code.view(-1, 257)  # (seq_len)
        loss = self.loss_fn(pred, gt)

        loss_dict = {
            "Loss": loss,
        }

        return loss_dict

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        data_dict = self.getCondition(data_dict)

        if is_training:
            data_dict["drop_prob"] = 0.0
        else:
            data_dict["drop_prob"] = 0.0

        return data_dict

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        # FIXME: skip this since it will occur NCCL error
        return True

        dataset = self.dataloader_dict["dino"]["dataset"]

        model.eval()

        data_dict = dataset.__getitem__(1)
        data_dict = self.getCondition(data_dict)

        condition = data_dict["condition"]

        print("[INFO][BaseDiffusionTrainer::sampleModelStep]")
        print("\t start sample shape code....")

        if not self.gt_sample_added_to_logger:
            # render gt here

            # self.logger.addPointCloud("GT_MASH/gt_mash", pcd, self.step)

            self.gt_sample_added_to_logger = True

        # self.logger.addPointCloud(model_name + "/pcd_" + str(i), pcd, self.step)

        return True
