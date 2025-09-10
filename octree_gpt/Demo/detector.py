import sys

sys.path.append("../dino-v2-detect/")

import os
import torch

from mash_diffusion.Module.detector import Detector


def demo(timestamp: str):
    cfm_model_file_path = "/vepfs-cnbja62d5d769987/lichanghao/github/MASH/mash-diffusion/output/test/model_last.pth"
    dino_model_file_path = "./data/dinov2_vitl14_reg4_pretrain.pth"
    occ_model_file_path = None
    # cfm_use_ema = True
    occ_use_ema = True
    device = "cuda:7"

    gpu_memory_Gb = 24
    occ_batch_size = int((-2 + 7.0 * gpu_memory_Gb / 12.0) * 100000)  # 24G->12, 12G->5

    # timestamp = "20250718_14:06:54"
    sample_data_folder_path = "./output/sample/" + timestamp + "/"
    sample_data_folder_path = "/nvme0pnt/lichanghao/chLi/Dataset/ShuMei/test/"
    if not os.path.exists(sample_data_folder_path):
        print("[ERROR][cfm_sampler::demo]")
        print("\t sample data folder not exist!")
        return False

    sample_mash_per_shape = 1

    timestamp_num = 2
    save_results_only = True

    recon_wnnc = False
    recon_occ = False
    render_pcd = False

    smooth_wnnc = True and recon_wnnc
    smooth_occ = True and recon_occ
    render_wnnc = True and recon_wnnc
    render_wnnc_smooth = True and recon_wnnc and smooth_wnnc
    render_occ = True and recon_occ
    render_occ_smooth = True and recon_occ and smooth_occ

    cfm_sampler = CFMSampler(
        cfm_model_file_path,
        dino_model_file_path,
        occ_model_file_path,
        False,
        occ_use_ema,
        device,
        occ_batch_size,
        recon_wnnc,
        recon_occ,
        smooth_wnnc,
        smooth_occ,
        render_pcd,
        render_wnnc,
        render_wnnc_smooth,
        render_occ,
        render_occ_smooth,
    )
    ema_cfm_sampler = CFMSampler(
        cfm_model_file_path,
        dino_model_file_path,
        occ_model_file_path,
        True,
        occ_use_ema,
        device,
        occ_batch_size,
        recon_wnnc,
        recon_occ,
        smooth_wnnc,
        smooth_occ,
        render_pcd,
        render_wnnc,
        render_wnnc_smooth,
        render_occ,
        render_occ_smooth,
    )

    shape_idxs = os.listdir(sample_data_folder_path)
    # shape_idxs.sort(key=int)

    for shape_idx in shape_idxs:
        print("start sample for shape No." + str(shape_idx) + " ...")
        shape_idx, shape_type = shape_idx.split(".")

        current_data_folder_path = sample_data_folder_path + shape_idx + "/"
        # gt_mash_file_path = current_data_folder_path + "gt_mash.npy"
        condition_image_file_path = current_data_folder_path + "condition_image.jpg"
        condition_image_file_path = (
            sample_data_folder_path + shape_idx + "." + shape_type
        )
        cfm_sampler.samplePipeline(
            condition_image_file_path,
            current_data_folder_path + "model/",
            sample_mash_per_shape,
            timestamp_num,
            save_results_only,
        )
        ema_cfm_sampler.samplePipeline(
            condition_image_file_path,
            current_data_folder_path + "ema/",
            sample_mash_per_shape,
            timestamp_num,
            save_results_only,
        )

    cfm_sampler.waitRender()
    ema_cfm_sampler.waitRender()

    torch.cuda.empty_cache()
    return True
