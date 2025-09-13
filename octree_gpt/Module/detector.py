import os
import torch
import numpy as np

from dino_v2_detect.Module.detector import Detector as DINODetector

from octree_gpt.Model.hy3d_gpt import HY3DGPT


class Detector(object):
    def __init__(
        self,
        model_file_path: str,
        dino_model_file_path: str,
        use_ema: bool = True,
        device: str = "cpu",
    ) -> None:
        self.context_dim = 1024
        self.n_heads = 8
        self.d_head = 64
        self.depth = 8
        self.depth_single_blocks = 16

        self.context_dim = 1024
        self.n_heads = 4  # 16
        self.d_head = 32
        self.depth = 4  # 16
        self.depth_single_blocks = 8  # 32

        self.use_ema = use_ema
        self.device = device

        self.model = HY3DGPT(
            context_dim=self.context_dim,
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth,
            depth_single_blocks=self.depth_single_blocks,
        ).to(self.device)

        self.loadModel(model_file_path)

        model_type = "large"
        dtype = "auto"
        self.dino_detector = DINODetector(
            model_type, dino_model_file_path, dtype, device
        )
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model_file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(
            model_file_path, map_location=torch.device(self.device), weights_only=False
        )

        if self.use_ema:
            self.model.load_state_dict(model_dict["ema_model"])
        else:
            self.model.load_state_dict(model_dict["model"])

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    def getCondition(
        self,
        condition: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        condition_tensor = condition.type(torch.float32).to(self.device)

        if condition_tensor.ndim == 1:
            condition_tensor = condition_tensor.view(1, 1, -1)
        elif condition_tensor.ndim == 2:
            condition_tensor = condition_tensor.unsqueeze(1)
        elif condition_tensor.ndim == 4:
            condition_tensor = torch.squeeze(condition_tensor, dim=1)

        condition_tensor = condition_tensor.repeat(
            *([batch_size] + [1] * (condition_tensor.ndim - 1))
        )

        return condition_tensor

    @torch.no_grad()
    def detect(self, condition: torch.Tensor, max_length: int) -> np.ndarray:
        self.model.eval()

        condition_tensor = self.getCondition(condition, 1)

        input_shape_code = torch.tensor([[255]], device=condition.device)

        shape_code = [255]

        # past_key_values = None

        for step in range(max_length):
            # 模型前向传播，传入 past_key_values 实现缓存加速
            next_shape_code = self.model.forwardData(
                input_shape_code,  # 当前输入 token
                condition_tensor,  # 条件向量
                # past_key_values=past_key_values,  # KV 缓存
            )

            # past_key_values = outputs.past_key_values  # 更新缓存

            # 采样或贪婪选择下一个 token
            next_token = torch.argmax(
                next_shape_code[:, -1, :], dim=-1, keepdim=True
            )  # shape: [1, 1]

            # 保存 token 或特征（例如 vocab_id，或 embedding）
            shape_code.append(next_token.item())

            # 可选：终止符判断
            if next_token.item() == 256:
                break

        return np.array(shape_code)

    def samplePipeline(
        self,
        condition_image_file_path: str,
        max_length: int,
        save_folder_path: str,
    ) -> bool:
        if not os.path.exists(condition_image_file_path):
            print("[ERROR][Detector::samplePipeline]")
            print("\t condition image file not exist!")
            print("\t condition_image_file_path:", condition_image_file_path)
            return False

        os.makedirs(save_folder_path, exist_ok=True)

        condition = self.dino_detector.detectFile(condition_image_file_path)

        print("start generate shape code....")
        shape_code = self.detect(condition, max_length)

        np.save(save_folder_path, shape_code)
        return True
