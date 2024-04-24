import pickle
import open3d as o3d
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

with open(
        'F:/mmdetection3D/mmdetection3d/result/val_result/pred_instances_3d.pkl',
        "rb") as f:
    results = pickle.load(f)
with open(
        'F:/mmdetection3D/data/raw_infos/infos_convert_v2/carla_infos_val.pkl',
        "rb") as f:
    pre = pickle.load(f)
num = 0
pcd_name = pre['data_list'][num]['lidar_points']['lidar_path']
points = np.fromfile(f'F:/mmdetection3D/data/raw_infos/velodyne/{pcd_name}', dtype=np.float32)
points = points.reshape((-1, 4))
points_xyz = points[:, :3]
points_xyz = points_xyz[points_xyz[:, 2] > 0.1]
sceneid = pre['data_list'][num]['sample_idx']
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_xyz)

bboxes = []
instances = pre['data_list'][num]['instances']
for i, box in enumerate(instances):
    box_3d = box['bbox_3d']
    center = box_3d[:3]
    extend = box_3d[3:6]
    yaw = box_3d[6]
    center[2] += extend[2] / 2
    r = R.from_euler("z", yaw, degrees=False)

    bbox = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=r.as_matrix(),
        extent=extend,
    )
    bbox.color = [0, 1, 0]
    bboxes.append(bbox)
# for j in range(1):
#     with open(f'D:/carlaCode/co-simulation/Sumo/result/town05/type=1_height=8/lidar/002337_lidar{j}.yaml', 'r') as file:
#         data = yaml.safe_load(file)
#     for vehicle_id, vehicle_data in data["vehicles"].items():
#         center = np.array(vehicle_data["center"]) + np.array(vehicle_data["location"])
#         extent = np.array(vehicle_data["extent"]) * 2
#         yaw = vehicle_data["rotation"][1]
#         center[0] = -center[0]
#         r = R.from_euler("z", -yaw, degrees=True)
#
#         bbox = o3d.geometry.OrientedBoundingBox(
#             center=center,
#             R=r.as_matrix(),
#             extent=extent,
#         )
#         bbox.color = [0, 1, 0]
#         bboxes.append(bbox)
#
#     for vehicle_id, vehicle_data in data["pedestrians"].items():
#         center = np.array(vehicle_data["center"]) + np.array(vehicle_data["location"])
#         extent = np.array(vehicle_data["extent"]) * 2
#         yaw = vehicle_data["rotation"][1]
#         center[0] = -center[0]
#         r = R.from_euler("z", -yaw, degrees=True)
#
#         bbox = o3d.geometry.OrientedBoundingBox(
#             center=center,
#             R=r.as_matrix(),
#             extent=extent,
#         )
#         bbox.color = [0, 1, 0]
#         bboxes.append(bbox)
#
#     for vehicle_id, vehicle_data in data["cyclists"].items():
#         center = np.array(vehicle_data["center"]) + np.array(vehicle_data["location"])
#         extent = np.array(vehicle_data["extent"]) * 2
#         yaw = vehicle_data["rotation"][1]
#         center[0] = -center[0]
#         r = R.from_euler("z", -yaw, degrees=True)
#
#         bbox = o3d.geometry.OrientedBoundingBox(
#             center=center,
#             R=r.as_matrix(),
#             extent=extent,
#         )
#         bbox.color = [0, 1, 0]
#         bboxes.append(bbox)


bboxes1 = []
locs = results[num]['location']
dims = results[num]['dimensions']
rots = results[num]['rotation_y']
ids = results[num]['sample_idx']
for i in range(len(ids)):
    center = locs[i]
    center[2] += dims[i][2]/2
    r = R.from_euler("z", rots[i], degrees=False)

    bbox = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=r.as_matrix(),
        extent=dims[i],
    )
    bbox.color = [100, 100, 0]
    bboxes1.append(bbox)

bbox_min = np.array([-61.4, -10.6, -0.1])
bbox_max = np.array([-35.4, 13.3, 3.9])

# Create an axis-aligned bounding box (AABB)
aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
aabb.color = [0, 1, 0]

o3d.visualization.draw([pcd] + bboxes + bboxes1+[aabb])
