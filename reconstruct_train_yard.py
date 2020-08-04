import json
import os

import cv2
import numpy as np
from imutils.paths import list_images

from Camera import Camera
from Pose import MatchedPose
from neural_network.openpose_wrapper import static_openpose_25_kp
from pose_3d_association import track_3d_poses
from pose_association import association_pose
from util_tools.draw_utils import draw_skeleton_in_frame, draw_pose_2d_in_image
from util_tools.fitting import poly_cure_fitting
from util_tools.utils import get_kp_rectangle
from vis.simple_3d_vis import simple_3d_vis

"""
（0）读取数据;
"""

root_path = '../src/demo1/'
cam_ids = [0, 2, 3, 4]
# root_path = '../src/demo2/'
# cam_ids = [2, 3, 4]

camera_parameters_root = root_path + 'camera_parameters/'

camera_calibration_paths = [os.path.join(camera_parameters_root, 'camera{}'.format(cam_id)) for cam_id in cam_ids]

cameras = []
for camera_calibration_path in camera_calibration_paths:
    intrinsic_path = os.path.join(camera_calibration_path, 'intrinsic_processed_kalibr_people_mp_cm.json')
    extrinsic_path = os.path.join(camera_calibration_path, 'extrinsic_processed_kalibr_people_mp_cm.json')

    with open(intrinsic_path, 'r') as f:
        intrinsic = json.load(f)
        K = np.array(intrinsic['intrinsic'], dtype=np.float32)
        D = np.array(intrinsic['distortion_coefficients'], dtype=np.float32)

    with open(extrinsic_path, 'r') as f:
        extrinsic = json.load(f)
        rvec = np.array(extrinsic['rvec'], dtype=np.float32)
        tvec = np.array(extrinsic['tvec'], dtype=np.float32).reshape((3, 1))

        R, _ = cv2.Rodrigues(rvec)
        projection = np.hstack((R, tvec))
        P = K @ projection
        C = -np.asmatrix(R).T @ tvec
    cameras.append(Camera(P, R, K, C, D))

images_dirs = [root_path + 'images.old/camera{}'.format(cam_id) for cam_id in cam_ids]
camera_images = [sorted(list(list_images(image_dir)),
                        key=lambda x: int(os.path.basename(x).split('.')[0])
                        ) for image_dir in images_dirs]

skeleton_dirs = [root_path + 'skeletons/camera{}'.format(cam_id) for cam_id in cam_ids]

"""
(1) correct errors in the output of the pose detector;
"""

"""
(2) apply a fast greedy algorithm for associating 2D pose detections between camera views;
"""

vis_in_2d = False
vis_in_3d = True

frame_based_generated_3d_poses = []

frame_index = 35
frame_total_nums = 60

while frame_index < frame_total_nums:
    frame_poses = []
    for i, (images, camera, camera_id, skeleton_dir) in enumerate(
            zip(camera_images, cameras, list(range(len(cameras))), skeleton_dirs)):

        if len(images) <= 0:
            continue

        image_path = images[frame_index]
        image_name = os.path.basename(image_path).split('.')[0]
        frame = cv2.imread(image_path)
        img_height, img_width = frame.shape[:2]

        # poses = openpose_25_kp(frame)
        poses = static_openpose_25_kp(skeleton_dir, image_name)
        if isinstance(poses, np.ndarray) and poses.ndim == 3 and len(poses) >= 1:
            for k, pose in enumerate(poses):
                if vis_in_2d:
                    draw_skeleton_in_frame(frame, pose, show_skeleton_labels=True)
                    xmin, xmax, ymin, ymax = get_kp_rectangle(pose, img_width, img_height)
                    cv2.putText(frame, str(k), (int((xmin + xmax) / 2), int(ymax)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2)

                pose, scores = np.asarray(pose[:, :2]).astype(np.float32), pose[:, 2]
                undistorted_pose = cv2.undistortPoints(pose, camera.camera_matrix, camera.dist_coefs, None,
                                                       None, camera.camera_matrix).reshape((25, 2))
                frame_poses.append(MatchedPose(undistorted_pose, scores, camera_id))

        if vis_in_2d:
            cv2.imshow('res', frame)
            cv2.waitKey(0)

    # for end

    frame_index += 1

    print('frame {} begin to match 3d pose'.format(frame_index))
    generated_3d_poses = association_pose(frame_poses, cameras, frame_index)

    # 计算每一个3D pose距离地面的高度
    for generated_3d_pose in generated_3d_poses:
        joints_h = generated_3d_pose.pose3d[:, 2]
        score_h = generated_3d_pose.is_valid_joint

        joints_h_true = []
        for i in [11, 14, 19, 20, 21, 22, 23, 24]:
            if score_h[i]:
                joints_h_true.append(-joints_h[i])

        h = np.average(joints_h_true) - 10
        if len(joints_h_true) > 0 and h > 0:
            generated_3d_pose.height_above_ground = h

    frame_based_generated_3d_poses.append(generated_3d_poses)
    print('frame {} generated poses num: {}'.format(frame_index, len(generated_3d_poses)))

    pose3ds, is_valid_joint_list = [], []
    for p, generated_pose in enumerate(generated_3d_poses):
        pose3d = generated_pose.pose3d
        is_valid_joint = generated_pose.is_valid_joint

        pose3ds.append(np.swapaxes(np.copy(pose3d), 0, 1))
        is_valid_joint_list.append(is_valid_joint)

    if len(pose3ds) > 0 and vis_in_3d:
        simple_3d_vis(frame_index, pose3ds, is_valid_joint_list)

# while end


"""
（3）use the associated poses to generate and track 3D skeletons。
"""

# 每条路径中第一个GeneratedPose，GeneratedPose中记录了具体的3D-Pose坐标、其所在的帧以及指向下一个GeneratedPose的指针。
trajectory_heads = track_3d_poses(frame_based_generated_3d_poses, max_frame_difference=30,
                                  distant_thresh=10)

trajectories = []

for i, trajectory_head in enumerate(trajectory_heads):
    print('{}th poses'.format(i))

    trajectoy = {}
    head = trajectory_head
    trajectoy[head.frame_index] = head
    while head.next:
        head = head.next
        trajectoy[head.frame_index] = head

    trajectories.append(trajectoy)

"""
（4）拟合球员距离地面高度并可视化
"""
import util_tools.draw_utils as draw

coords = []
for trajectoy in trajectories:

    x = []
    y = []

    for frame in trajectoy.keys():
        x.append(frame)
        y.append(trajectoy[frame].height_above_ground)

    x, y = poly_cure_fitting(x, y)
    coords.append((x, y))

    for frame in trajectoy.keys():
        if y[x.index(frame)] > 0:
            trajectoy[frame].height_above_ground = y[x.index(frame)]

draw.plot_single_player_height_above_ground(coords).show()

"""
（5）重投影结果到视频
"""

# camera_images = [list(list_images(image_dir)) for image_dir in images_dirs]

for i, (images, camera, camera_id) in enumerate(zip(camera_images, cameras, cam_ids)):

    image_path = images[0]
    image_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    out_video_name = root_path + 'videos_out/{}.mp4'.format(camera_id)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    res_video = cv2.VideoWriter(out_video_name, fourcc, float(10), (int(width), int(height)), True)

    for frame_index in range(len(images)):
        # if frame_index + 1 > frame_total_nums:
        #     break

        image_path = images[frame_index]
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)

        frame = cv2.putText(img, '{}th frame'.format(frame_index), (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        for idx, trajectoy in enumerate(trajectories):
            current_pose = trajectoy.get(frame_index)
            if current_pose is not None:
                pose3d = current_pose.pose3d
                is_valid_joint = current_pose.is_valid_joint

                pose_2d = camera.projection_pose(pose3d)

                frame = draw_pose_2d_in_image(frame, pose_2d, idx, is_valid_joint)
                cv2.putText(frame, 'HAG:{:.2f}cm'.format(current_pose.height_above_ground),
                            (int(np.average(pose_2d[19:25, 0])), int(np.average(pose_2d[:, 1]))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        res_video.write(frame)
        frame_index += 1
