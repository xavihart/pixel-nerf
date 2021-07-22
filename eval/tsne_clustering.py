import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
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
from tqdm import tqdm
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
    parser.add_argument(
        "--voxel_num",
        type=int,
        default=100
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        default='tsne'
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

net = make_model(conf["model"], using_intermediate_feature=True).to(device=device)
net.load_weights(args)
print('src views', src_view)
net.encode(
        images[src_view].unsqueeze(0),
        poses[src_view].unsqueeze(0).to(device=device),
        focal,
        c=c,
    )



feature_list = []
def inter_feature_hook(module, input, output):
    feature_list.append(output.data)



renderer = NeRFRenderer.from_conf(
    conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size,
).to(device=device)

render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

# Get the distance from camera to origin
z_near = dset.z_near
z_far = dset.z_far

N = args.voxel_num
print(args.name)
if "pour" in args.name:
    ty = np.linspace(1, 9, N + 1)
    tx = np.linspace(-4, 4, N + 1)
    tz = np.linspace(-3, 5, N + 1)
if "shake" in args.name:
    ty = np.linspace(0, 3, N + 1)
    tx = np.linspace(-1.5, 1.5, N + 1)
    tz = np.linspace(-1.5, 1.5, N + 1)

# ty = np.linspace(1, 9, N + 1)
# tx = np.linspace(-4, 4, N + 1)
# tz = np.linspace(-3, 5, N + 1)
query_pts = np.stack(np.meshgrid(tx, ty, tz), -1).astype(np.float32)


print(query_pts.shape)
sh = query_pts.shape
flat = query_pts.reshape([-1, 3])
flat = torch.from_numpy(flat).to(args.gpu_id[0])


fn = lambda i0, i1: net(flat[None, i0:i1, :], viewdirs=torch.zeros(flat[i0:i1].shape).to(args.gpu_id[0]))
# fn = lambda i0, i1: net(flat[None, i0:i1, :], viewdirs=None)
chunk = 1024

# sigma_list = []
# feature_list = []
#
# for i in tqdm(range(0, flat.shape[0], chunk)):
#     feature, out = fn(i, i + chunk)
#     feature_list.append(feature[0].detach().cpu().numpy())
#     sigma_list.append(out[0].detach().cpu().numpy())

feature = np.concatenate([fn(i, i + chunk)[0][0].detach().cpu().numpy() for i in tqdm(range(0, flat.shape[0], chunk))], 0)
raw = np.concatenate([fn(i, i + chunk)[1][0].detach().cpu().numpy() for i in tqdm(range(0, flat.shape[0], chunk))], 0)


# feature = np.concatenate(feature_list, 0)
# sigma = np.concatenate(sigma_list, 0)

sigma = np.reshape(raw, list(sh[:-1]) + [-1]) # N * N * N * 4
# sigma = sigma.view(-1, sigma.shape[-1])
sigma = np.maximum(sigma[..., -1], 0.)

print("calculating cluster information, using {} to get the decomposed representation".format(args.cluster_method))
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

if args.cluster_method == 'tsne':
    tsne = TSNE(n_components=3)
    sigma_flatten = sigma.flatten()
    valid = sigma_flatten > 0
    feature_tsne = tsne.fit_transform(feature[valid])
    feature = feature_tsne
    for i in range(3):
        feature[:, i] = (feature[:, i] - feature[:, i].min()) / (feature[:, i].max() - feature[:, i].min())
elif args.cluster_method == 'pca':
    pca = PCA(n_components=3)
    feature_pca = pca.fit_transform(feature)
    feature = feature_pca
    for i in range(3):
        feature[:, i] = (feature[:, i] - feature[:, i].min()) / (feature[:, i].max() - feature[:, i].min())
elif args.cluster_method == 'mean':
    feature_r = feature.mean(-1)
    feature_r = np.expand_dims(feature_r, -1).repeat(3, -1)
    print(feature_r.shape)
    feature = (feature_r - feature_r.min()) / (feature_r.max() - feature_r.min())
elif args.cluster_method == 'vertrgb':
    feature = np.maximum(raw[:, :3], 0.)


# feature_tsne = feature_tsne.reshape(list(sh[:-1]) + [-1]) # N * N * N * 3
saving_path = os.path.join(args.root, 'experimental', 'mesh_color', 'water_pour_S{}_resolution{}_feature{}/'.format(
    args.subset, args.voxel_num, args.cluster_method))

if not os.path.exists(saving_path):
    os.makedirs(saving_path)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# draw 3D plot
t = [i for i in range(N+1)]

sigma_flatten = sigma.flatten()
valid = sigma_flatten > 0

np.save(saving_path + 'feature_all.npy', feature)
np.save(saving_path + 'sigmma_all.npy', sigma_flatten)



x, y, z = np.meshgrid(t, t, t)
x, y, z = x.flatten(), y.flatten(), z.flatten()

color = feature

fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
if args.cluster_method == 'tsne':
    ax3D.scatter(x[valid], z[valid], y[valid], s=10, c=color, marker='o') # tsne is operated on filtered points
else:
    ax3D.scatter(x[valid], z[valid], y[valid], s=10, c=color[valid], marker='o')
ax3D.set_xlim3d(0, 100)
ax3D.set_ylim3d(0, 100)
ax3D.set_zlim3d(0, 100)
plt.show()






plt.hist(np.maximum(0, sigma.ravel()), log=True)
plt.savefig(saving_path + 'hist.jpg')

# import mcubes
# threshold = 5
# print('fraction occupied', np.mean(sigma > threshold))
# vertices, triangles = mcubes.marching_cubes(sigma, threshold)
# print('done', vertices.shape, triangles.shape)
#
#
#
# n_vert = vertices.shape[0]
#
# vert_index = vertices[:, 0] * (N + 1) * (N + 1)+ vertices[:, 1] * (N+1) + vertices[:, 2]
# vert_index = vert_index.astype(np.int)
#
# vert_rgb = feature[vert_index]
#
#
#
# src_view_images = np.hstack(images[src_view])
# print(src_view_images.shape)
#
#
# imageio.imwrite(saving_path + 'src_view.png', (((src_view_images.transpose(1, 2, 0)+1)/2)*255).astype(np.uint8))
# np.save(saving_path + 'vertices.npy', vertices)
# np.save(saving_path + 'triangles.npy', triangles)
# np.save(saving_path + 'color.npy', vert_rgb)
# util.save_obj(vertices, triangles, saving_path + "model.obj", vert_rgb=vert_rgb)
#
# print("object saved!")