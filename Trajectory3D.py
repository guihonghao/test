import numpy as np
from util_tools.draw_utils import draw_pose_2d_in_image

def group_consecutive(a):
    return np.split(a, np.where(np.diff(a) != 1)[0] + 1)

class Trajectory3D:
    def __init__(self, start_frame, end_frame, id, trajectory):
        self.joint_num = 25
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.id = id

        self.skeleton_3ds = np.zeros(
            (
                self.end_frame + 1 - self.start_frame, self.joint_num, 3
            ), dtype=np.float32
        )

        self.score_3ds = np.zeros(
            (
                self.end_frame + 1 - self.start_frame,  self.joint_num
            )
        )

        self.frame_list = list()

        for frame_index, generated_3d_pose in trajectory.items():
            self.skeleton_3ds[frame_index - self.start_frame] = generated_3d_pose.pose3d
            self.score_3ds[frame_index - self.start_frame] = generated_3d_pose.score_3d
            self.frame_list.append(frame_index)

        self.post_process()

    def post_process(self):
        print('{} --> {}'.format(self.start_frame, self.end_frame))
        for i in range(self.joint_num):
            valid_frame_list = np.where(self.score_3ds[:, i] > 0)[0] + self.start_frame
            frame_groups = group_consecutive(valid_frame_list)
            for j, (p_group, b_group) in enumerate(zip(frame_groups[:-1], frame_groups[1:])):
                pf, bf = p_group[-1], b_group[0]

                p_loc = self.skeleton_3ds[pf - self.start_frame, i]
                b_loc = self.skeleton_3ds[bf - self.start_frame, i]
                diff = bf - pf + 1
                pf, bf = pf - self.start_frame, bf - self.start_frame
                for k in range(3):
                    interpolates = np.linspace(p_loc[k], b_loc[k], diff)
                    print('it',interpolates.shape)
                    print('sk_3d',self.skeleton_3ds[pf: bf + 1, i, k].shape)
                    self.skeleton_3ds[pf: bf + 1, i, k] = interpolates

    def show_fth_pose_in_image(self, aa, f, camera):
        if self.start_frame<=f<=self.end_frame:
            pose_3d = self.skeleton_3ds[f - self.start_frame]
            pose_2d = camera.projection_pose(pose_3d)
            aa = draw_pose_2d_in_image(aa, pose_2d, self.id, self.score_3ds[f-self.start_frame] > 0)
        return aa







