from estimater import *
from datareader import *
from Utils import *
import trimesh
import numpy as np
import os

# import your custom reader
# from my_rosbag_reader import RosbagFoundationPoseReader

set_logging_format()
set_seed(0)

root = "/home/anahita/Dataset/rosbag_extracted_300m"
mesh_file = f"{root}/CAD_car/racecar0_highres.ply"
split_file = "/home/anahita/self-supervised-depth-completion/splits/iac_train.txt"
mask_file = f"{root}/mask/mask_005020.png"   # use your saved mask

debug_dir = "./debug_rosbag"
os.makedirs(debug_dir, exist_ok=True)

reader = RosbagFoundationPoseReader(
    root=root,
    split_file=split_file,
    mask_file=mask_file,
    use_lidar_mask=False,
    start_id=5020,   # or 5020 depending on where you want to start
)

mesh = trimesh.load(mesh_file)

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()

est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=debug_dir,
    debug=2,
    glctx=glctx,
)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

for i in range(len(reader.color_files)):

    color = reader.get_color(i)
    depth = reader.get_depth(i)

    if i == 0:
        mask = reader.get_mask(i)

        pose = est.register(
            K=reader.K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=5,
        )

    else:
        pose = est.track_one(
            rgb=color,
            depth=depth,
            K=reader.K,
            iteration=2,
        )

    center_pose = pose @ np.linalg.inv(to_origin)

    vis = draw_posed_3d_box(
        reader.K,
        img=color,
        ob_in_cam=center_pose,
        bbox=bbox,
    )

    vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=0.5,
        K=reader.K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )

    cv2.imshow("FoundationPose", vis[..., ::-1])
    np.savetxt(f"{debug_dir}/{reader.id_strs[i]}.txt", pose.reshape(4, 4))
    # print(reader.id_strs[i], pose)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    