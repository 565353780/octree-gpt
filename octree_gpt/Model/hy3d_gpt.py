import torch

from octree_gpt.Model.hy3ddit import Hunyuan3DDiT


class HY3DGPT(torch.nn.Module):
    def __init__(
        self,
        context_dim=1024,
        n_heads=16,
        d_head=64,
        depth=16,
        depth_single_blocks=32,
    ):
        super().__init__()

        hidden_size = d_head * n_heads
        self.model = Hunyuan3DDiT(
            context_in_dim=context_dim,
            hidden_size=hidden_size,
            mlp_ratio=4.0,
            num_heads=n_heads,
            depth=depth,
            depth_single_blocks=depth_single_blocks,
            axes_dim=[d_head],
            theta=10_000,
            qkv_bias=True,
            time_factor=1000,
        )
        return

    def forwardCondition(
        self, input_shape_code: torch.Tensor, condition: torch.Tensor
    ) -> dict:
        next_shape_code = self.model(input_shape_code, cond=condition)

        result_dict = {"next_shape_code": next_shape_code}

        return result_dict

    def forwardData(
        self, input_shape_code: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        result_dict = self.forwardCondition(input_shape_code, condition)

        vt = result_dict["vt"]

        return vt

    def forward(self, data_dict: dict) -> dict:
        input_shape_code = data_dict["input_shape_code"]
        condition = data_dict["condition"]
        drop_prob = data_dict["drop_prob"]

        if drop_prob > 0:
            drop_mask = torch.rand_like(condition) <= drop_prob
            condition[drop_mask] = 0

        result_dict = self.forwardCondition(input_shape_code, condition)

        return result_dict
