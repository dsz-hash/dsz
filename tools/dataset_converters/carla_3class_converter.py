import os
from pathlib import Path
from typing import List, Set
import pickle
import random

import numpy as np
import mmengine
import open3d as o3d

from .carla_3class_data_utils import DataInfo


class CarlaConverter:
    def __init__(
            self,
            data_path: Path,
            out_path: Path,
            raw_data_infos: List[DataInfo],
            num_workers: int,
            lidar_idxs: Set[int],
    ):
        self.data_path = data_path
        self.out_path = out_path
        self.raw_data_infos = raw_data_infos
        self.num_workers = num_workers
        self.lidar_idxs = lidar_idxs
        self.velodyne_path = self.out_path / "velodyne"
        if not self.velodyne_path.exists():
            self.velodyne_path.mkdir(parents=True)

        self.velodyne_src_indices_path = self.out_path / "velodyne_src_indices"
        if not self.velodyne_src_indices_path.exists():
            self.velodyne_src_indices_path.mkdir(parents=True)

    def convert(self):
        print('Start converting ...')
        data_infos = mmengine.track_parallel_progress(
            self.convert_one,
            tasks=self.raw_data_infos,
            nproc=self.num_workers,
        )
        print(f'Before removing None, data_infos: {len(data_infos)}')
        data_infos = [data_info for data_info in data_infos if data_info is not None]
        print(f'After removing None, data_infos: {len(data_infos)}')

        r = random.Random()
        r.seed(233)
        r.shuffle(data_infos)
        random.shuffle(data_infos)

        num_train_infos = int(len(data_infos) * 0.8)
        train_data_infos = data_infos[:num_train_infos]
        val_data_infos = data_infos[num_train_infos:]

        with open(self.out_path / "carla_infos_train.pkl", "wb") as f:
            pickle.dump(train_data_infos, f)

        with open(self.out_path / "carla_infos_val.pkl", "wb") as f:
            pickle.dump(val_data_infos, f)

        print('\nFinished ...')

    def convert_one(self, raw_data_info: DataInfo):
        points_list = []
        src_indices = []
        for i, lidar_info in enumerate(raw_data_info.lidars):
            if lidar_info.index not in self.lidar_idxs:
                continue

            pc = o3d.io.read_point_cloud(
                str(self.data_path / lidar_info.pc_path)
            )
            # TODO 已经是世界坐标下，不需要在平移旋转点云
            # pc.rotate(lidar_info.sensor_rot)
            # pc.translate(lidar_info.sensor_trans)

            points = np.asarray(pc.points).astype(np.float32)
            colors = np.asarray(pc.colors).astype(np.float32)
            points_list.append(
                np.concatenate([points, colors[:, :1]], axis=-1)
            )
            src_indices.append(np.ones((points.shape[0],), dtype=np.int64) * lidar_info.index)

        points = np.concatenate(points_list, axis=0)
        if points.shape[0] != 0:
            points[:, 0] = -points[:, 0]
            points.tofile(str(self.velodyne_path / raw_data_info.scene_id) + ".bin")

            src_indices = np.concatenate(src_indices, axis=0)
            src_indices.tofile(str(self.velodyne_src_indices_path / raw_data_info.scene_id) + ".bin")

            points_min, points_max = points[:, :3].min(0), points[:, :3].max(0)
            points_range = np.concatenate([points_min, points_max], axis=0)

            gt_bboxes_3d = []
            gt_names = []
            difficulty = []
            srcs = []
            for vehicle in raw_data_info.vehicles:
                bbox = vehicle.get_bbox()
                assert len(bbox) == 7
                gt_bboxes_3d.append(bbox)
                gt_names.append("Car")
                difficulty.append(0)
                srcs.append(vehicle.srcs)

            for pedestrian in raw_data_info.pedestrians:
                bbox = pedestrian.get_bbox()
                assert len(bbox) == 7
                gt_bboxes_3d.append(bbox)
                gt_names.append("Pedestrian")
                difficulty.append(0)
                srcs.append(pedestrian.srcs)

            for cyclist in raw_data_info.cyclists:
                bbox = cyclist.get_bbox()
                assert len(bbox) == 7
                gt_bboxes_3d.append(bbox)
                gt_names.append("Cyclist")
                difficulty.append(0)
                srcs.append(cyclist.srcs)

            data_info = {
                "scene_id": raw_data_info.scene_id,
                "annos": {
                    "bboxes_3d": gt_bboxes_3d,
                    "name": gt_names,
                    "difficulty": difficulty,
                    "srcs": srcs,
                },
                "points_range": points_range,
            }

            return data_info
        else:

            return None
