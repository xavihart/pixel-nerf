import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "", "src"))
)

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
from scipy.interpolate import CubicSpline
import tqdm
import matplotlib.pylab as plt
import trimesh
from dotmap import DotMap

def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="0 1 2 3 4",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=400,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=8.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/home/htxue/data/mit/pixel-nerf/"
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    return parser



args, conf = util.args.parse_args(extra_args, default_conf="conf/default_mv.conf")
args.resume = True



print(args)
device = util.get_cuda(args.gpu_id[0])
dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False
)

data = dset[args.subset]
data_path = data["path"]
print("Data instance loaded:", data_path)

images = data["images"]  # (NV, 3, H, W)
poses = data["poses"]  # (NV, 4, 4)
focal = data["focal"]


if isinstance(focal, float):
    # Dataset implementations are not consistent about
    # returning float or scalar tensor in case of fx=fy
    focal = torch.tensor(focal, dtype=torch.float32)
focal = focal[None]

c = data.get("c")
if c is not None:
    c = c.to(device=device).unsqueeze(0)

NV, _, H, W = images.shape


focal = focal.to(device=device)

source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1
assert not (source >= NV).any()


print("H, W:",H, W)

if random_source:
    src_view = torch.randint(0, NV, (1,))
else:
    src_view = source


if args.scale != 1.0:
    Ht = int(H * args.scale)
    Wt = int(W * args.scale)
    if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
        warnings.warn(
            "Inexact scaling, please check {} times ({}, {}) is integral".format(
                args.scale, H, W
            )
        )
    H, W = Ht, Wt

net = make_model(conf["model"]).to(device=device)
net.load_weights(args)
print('src views', src_view)
net.encode(
        images[src_view].unsqueeze(0),
        poses[src_view].unsqueeze(0).to(device=device),
        focal,
        c=c,
    )


renderer = NeRFRenderer.from_conf(
    conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size,
).to(device=device)

render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

# Get the distance from camera to origin
z_near = dset.z_near
z_far = dset.z_far

N = 256
tx = ty = tz = np.linspace(-5, 5, N + 1)

ty += 3

query_pts = np.stack(np.meshgrid(tx, ty, tz), -1).astype(np.float32)


print(query_pts.shape)
sh = query_pts.shape
flat = query_pts.reshape([-1, 3])
flat = torch.from_numpy(flat).to(args.gpu_id[0])


fn = lambda i0, i1: net(flat[None, i0:i1, :], viewdirs=torch.zeros(flat[i0:i1].shape).to(args.gpu_id[0]))
# fn = lambda i0, i1: net(flat[None, i0:i1, :], viewdirs=None)
chunk = 1024 * 64
raw = np.concatenate([fn(i, i + chunk)[0].detach().cpu().numpy() for i in range(0, flat.shape[0], chunk)], 0)
raw = np.reshape(raw, list(sh[:-1]) + [-1])
sigma = np.maximum(raw[..., -1], 0.)

print(raw.shape)
plt.hist(np.maximum(0, sigma.ravel()), log=True)
plt.show()


import mcubes
threshold = 10.
print('fraction occupied', np.mean(sigma > threshold))
vertices, triangles = mcubes.marching_cubes(sigma, threshold)
print('done', vertices.shape, triangles.shape)



np.save('vertices.npy', vertices)
np.save('triangles.npy', triangles)



print("Done")


# os.environ["PYOPENGL_PLATFORM"] = "egl"
# import pyrender
# scene = pyrender.Scene()
# scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
#
# # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
# camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#
# camera_pose = pose_spherical(-20., -40., 1.).numpy()
# nc = pyrender.Node(camera=camera, matrix=camera_pose)
# scene.add_node(nc)
#
# # Set up the light -- a point light in the same spot as the camera
# light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
# nl = pyrender.Node(light=light, matrix=camera_pose)
# scene.add_node(nl)
#
# # Render the scene
# r = pyrender.OffscreenRenderer(640, 480)
# color, depth = r.render(scene)
#
# plt.imshow(color)
# plt.show()
# plt.imshow(depth)
# plt.show()