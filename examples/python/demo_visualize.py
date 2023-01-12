import os
import sys
from tqdm import tqdm

import open3d as o3d
import numpy as np

import argparse


# import pypatchworkpp

cur_dir = os.path.dirname(os.path.abspath(__file__))
# input_cloud_filepath = os.path.join(cur_dir, "../../data/000000.bin")

try:
    patchwork_module_path = os.path.join(cur_dir, "../../build/python_wrapper")
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)


# def read_bin(bin_path):
#     scan = np.fromfile(bin_path, dtype=np.float32)
#     scan = scan.reshape((-1, 4))

#     return scan


def read_pcds(
    path: str, add_intensity: bool = False, return_filenames: bool = False
) -> np.ndarray:
    """
    Reads a pointcloud with open3d and returns it as a numpy ndarray
    Args:
        path (str): path to the pointcloud to be read or the directory containing it/them
    Returns:
        np.ndarray: the pointcloud(s) as a numpy ndarray (dims: (pointcloud x) points x coordinates)
    """
    if os.path.isdir(path):
        pointclouds = []
        filenames = sorted(os.listdir(path))
        for filename in filenames:
            if filename[-4:] != ".pcd":
                continue
            pcd = np.asarray(
                o3d.io.read_point_cloud(os.path.join(path, filename)).points
            )
            if add_intensity:
                pcd = np.pad(pcd, pad_width=((0, 0), (0, 1)), constant_values=255)
            pointclouds.append(pcd)

        if return_filenames:
            return pointclouds, filenames
        return pointclouds

    elif os.path.isfile(path):
        pcd = o3d.io.read_point_cloud(path).points
        if add_intensity:
            pcd = np.pad(pcd, pad_width=((0, 0), (0, 1)), constant_values=255)
        return [np.asarray(pcd.points)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point-clouds",
        type=str,
        required=True,
        help="Path to the folder containing the to , e.g. ./results",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the folder where to save the results (ground and non ground), e.g. ./results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Patchwork++ verbose",
    )
    args = parser.parse_args()
    point_clouds_path = args.point_clouds
    output_path = args.output_path
    verbose = args.verbose

    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
    params.verbose = True if verbose else False

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    # Load point clouds
    pointclouds, pointcloud_filenames = read_pcds(
        point_clouds_path, add_intensity=True, return_filenames=True
    )

    path_ground = os.path.join(output_path, "ground")
    path_nonground = os.path.join(output_path, "nonground")

    if not os.path.exists(path_ground):
        os.makedirs(os.path.join(output_path, "ground"))

    if not os.path.exists(path_nonground):
        os.makedirs(os.path.join(output_path, "nonground"))

    for filename, pointcloud in tqdm(zip(pointcloud_filenames, pointclouds)):

        # Estimate Ground
        PatchworkPLUSPLUS.estimateGround(pointcloud)

        # Get Ground and Nonground
        ground = PatchworkPLUSPLUS.getGround()
        nonground = PatchworkPLUSPLUS.getNonground()
        time_taken = PatchworkPLUSPLUS.getTimeTaken()

        # Get centers and normals for patches
        centers = PatchworkPLUSPLUS.getCenters()
        normals = PatchworkPLUSPLUS.getNormals()

        print("Original Points  #: ", pointcloud.shape)
        print("Ground Points    #: ", ground.shape)
        print("Nonground Points #: ", nonground.shape)
        print("Time Taken : ", time_taken / 1000000, "(sec)")

        # Save results
        np.savetxt(os.path.join(path_ground, filename), ground)
        np.savetxt(os.path.join(path_nonground, filename), nonground)

    # print("Press ... \n")
    # print("\t H  : help")
    # print("\t N  : visualize the surface normals")
    # print("\tESC : close the Open3D window")

    # # Visualize
    # vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window(width = 600, height = 400)

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # ground_o3d = o3d.geometry.PointCloud()
    # ground_o3d.points = o3d.utility.Vector3dVector(ground)
    # ground_o3d.colors = o3d.utility.Vector3dVector(
    #     np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float) # RGB
    # )

    # nonground_o3d = o3d.geometry.PointCloud()
    # nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
    # nonground_o3d.colors = o3d.utility.Vector3dVector(
    #     np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float) #RGB
    # )

    # centers_o3d = o3d.geometry.PointCloud()
    # centers_o3d.points = o3d.utility.Vector3dVector(centers)
    # centers_o3d.normals = o3d.utility.Vector3dVector(normals)
    # centers_o3d.colors = o3d.utility.Vector3dVector(
    #     np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float) #RGB
    # )

    # vis.add_geometry(mesh)
    # vis.add_geometry(ground_o3d)
    # vis.add_geometry(nonground_o3d)
    # vis.add_geometry(centers_o3d)

    # vis.run()
    # vis.destroy_window()
