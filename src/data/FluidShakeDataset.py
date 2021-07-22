import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import pickle
from util import get_image_to_tensor_balanced, get_mask_to_tensor



class FluidShakeDataset(torch.utils.data.Dataset):
    """
    Dataset from Yunzhu Li
    """

    def __init__(self, path, stage='train', image_size=(128, 128), word_scale=1.0, img_format='png'):
        """

        :param path:
        :param split:
        :param image_size:
        :param word_scale:
        """
        assert stage in ['train', 'test', 'val']
        super(FluidShakeDataset, self).__init__()

        self.base_path = path
        self.img_format = img_format

        all_traj = sorted(os.listdir(path))
        num_traj = len(all_traj)

        # split in small set
        train_traj = all_traj[:10]
        test_traj = all_traj[10:12]
        val_traj = all_traj[12:13]

        if stage == "train":
            self.traj_path = train_traj
        elif stage == 'test':
            self.traj_path = test_traj
        else:
            self.traj_path = val_traj

        self.image_size = image_size
        self.world_scale = word_scale
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        self.lindisp = False
        # ?
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )
        self.z_near, self.z_far = 1, 9.5

        self.num_frames = len(os.listdir(os.path.join(self.base_path, self.traj_path[0] + "/0/")))  # 300

        self.num_poses = len(os.listdir(os.path.join(self.base_path, self.traj_path[0]))) - 2

        print(
            "FluidShaking Dataset Loading ---- number of views = {} number of frames = {} \n Trajectory Path List \n".format(
                self.num_poses, self.num_frames))
        print(self.traj_path)

    def __len__(self):
        return len(self.traj_path) * self.num_frames

    def __getitem__(self, item):

        traj_id = item // self.num_frames
        fram_id = item % self.num_frames

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []

        print_intrin = True

        for pos_id in range(self.num_poses):
            cam_info_path = os.path.join(self.base_path,
                                         self.traj_path[traj_id])
            cam_info = pickle.load(open(cam_info_path + "/info.p", 'rb'))

            view_mat = cam_info['viewMatrix'][pos_id][fram_id]
            view_mat = np.linalg.inv(np.transpose(view_mat))  # 4 * 4
            pose = view_mat

            proj_mat = cam_info['projMatrix'][pos_id][fram_id]
            focal = proj_mat[0, 0]
            focal = focal * 0.5 * self.image_size[0]

            proj_mat = np.transpose(proj_mat)  # 4 * 4

            """
            perspective projection
            {
            C 0  0  0
            0 D  0  0 
            0 0  A  B
            0 0 -1  0
            }

            A = - (f + n) / (f - n)
            B = - (2nf) / (f - n)
            C = 2n / (r - l)
            D = 3n / (t - b)

            W = r - l ; H = t - b

            =>

            n, f = B / (A - 1), B / (A + 1)

            """

            C, D, A, B = proj_mat[0][0], proj_mat[1][1], proj_mat[2][-2], proj_mat[2][-1]

            n_near, n_far = B / (A - 1), B / (A + 1)

            realworld_W, realworld_H = 2 * n_near / C, 2 * n_near / D  # assert L=R fixed

            len_perpixel = realworld_W / self.image_size[0]

            cx, cy = self.image_size[0] // 2, self.image_size[1] // 2

            # if print_intrin:
            #     print("focal : [{}]".format(focal))
            #     print_intrin = False

            # base/0/0/0.jpg
            img = imageio.imread(os.path.join(self.base_path,
                                              self.traj_path[traj_id],
                                              str(pos_id),
                                              str(fram_id) + ".{}".format(self.img_format)))

            img[(img == 0).all(-1), :] = 255

            img_tensor = self.image_to_tensor(img)

            # mask : all black
            mask = (img != 0).all(axis=-1)[..., None].astype(np.uint8) * 255  # H * W * 1
            mask_tensor = self.mask_to_tensor(mask)
            # ?
            pose = torch.from_numpy(pose).float()  # @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]

            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            # focal *= scale
            # cx *= scale
            # cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale

        focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": "traj-{} frame-{}".format(traj_id, fram_id),
            "img_id": item,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result


if __name__ == "__main__":
    fluidshaking_dataset = FluidPourDataset(path='/home/htxue/src/water_pour_small/', stage='test',
                                            image_size=(128, 128), word_scale=1.0)










