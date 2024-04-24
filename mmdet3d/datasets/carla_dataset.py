from typing import Callable, List, Union
from mmdet3d.structures import LiDARInstance3DBoxes
import numpy as np

from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class CarlaDataset(Det3DDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Carla Dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to "velodyne".
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to "LiDAR" in this dataset. Available options includes

            - "LiDAR": box in LiDAR coordinates
            - "Depth": box in depth coordinates, usually for indoor dataset
            - "Camera": box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list(float), optional): The range of point cloud used
            to filter invalid predicted boxes.
            Default: [-85, -85, -5, 85, 85, 5].
    """

    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist'),
        'palette': [(165, 42, 42), (106, 0, 228), (119, 11, 32)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = None,
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [-61.4, -10.6, -0.1, -35.4, 13.3, 3.9],
                 **kwargs) -> None:
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        self.pts_prefix = data_prefix['pts']

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')
        self.pcd_limit_range = pcd_limit_range

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info = super().parse_data_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """

        ann_info = super().parse_ann_info(info)

        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
