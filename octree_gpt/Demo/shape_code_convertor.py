import sys

sys.path.append("../octree-shape")
sys.path.append("../data-convert")

import os

from octree_gpt.Module.shape_code_convertor import ShapeCodeConvertor


def demo():
    home = os.environ["HOME"]

    source_root_folder_path = home + "/chLi/Dataset/Objaverse_82K/mesh/"
    target_root_folder_path = home + "/chLi/Dataset/Objaverse_82K/shape_code/"
    source_type = ".obj"
    target_type = ".npy"
    depth_max = 8

    shape_code_convertor = ShapeCodeConvertor(
        source_root_folder_path,
        target_root_folder_path,
        depth_max,
    )

    shape_code_convertor.convertAll(source_type, target_type)
    return True
