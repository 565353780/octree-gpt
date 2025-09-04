import numpy as np

from octree_shape.Module.octree_builder import OctreeBuilder

from data_convert.Module.base_convertor import BaseConvertor


class ShapeCodeConvertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        depth_max: int = 10,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.depth_max = depth_max
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        octree_builder = OctreeBuilder(source_path, self.depth_max)

        shape_code = octree_builder.getShapeCode()

        shape_code = np.array(shape_code, dtype=np.uint8)

        np.save(target_path, shape_code)
        return True
