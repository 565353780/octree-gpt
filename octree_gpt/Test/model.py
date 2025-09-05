import torch
from torch.amp import autocast

from octree_gpt.Model.hy3d_gpt import HY3DGPT


def test():
    dtype = torch.float32
    device = "cuda:0"

    model = HY3DGPT(
        context_dim=1024,
        n_heads=8,
        d_head=64,
        depth=8,
        depth_single_blocks=16,
    ).to(device, dtype=dtype)

    data_dict = {
        "input_shape_code": torch.randint(
            0, 255, [2, 8192], dtype=torch.int64, device=device
        ),
        "next_shape_code": torch.randint(
            0, 255, [2, 8192], dtype=torch.int64, device=device
        ),
        "condition": torch.randn([2, 1397, 1024], dtype=dtype, device=device),
        "drop_prob": 0.0,
    }

    with autocast("cuda", dtype=torch.float16):
        result_dict = model(data_dict)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=257)

    gt_next_shape_code = data_dict["next_shape_code"]
    pred_next_shape_code = result_dict["next_shape_code"]

    gt = gt_next_shape_code.view(-1)  # (seq_len, vocab_size)
    pred = pred_next_shape_code.view(-1, 257)  # (seq_len)
    loss = loss_fn(pred, gt)

    loss.backward()

    print(result_dict["next_shape_code"].shape)
    print(loss)
    return True
