#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang (modified by かわいい妹キャラ)
# E-mail     : liang@informatik.uni-hamburg.de
# Description: pcl部分をopen3dに置き換えました！🌟
# Date       : 09/06/2018 7:47 PM (modified 2025/03/10)

import os
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.grasping import GraspableObject3D
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import time
from mayavi import mlab
import open3d as o3d  # open3dをインポートするよ〜！


def show_obj(surface_points_, color="b"):
    if color == "b":
        color_f = (0, 0, 1)
    elif color == "r":
        color_f = (1, 0, 0)
    elif color == "g":
        color_f = (0, 1, 0)
    else:
        color_f = (1, 1, 1)
    points = surface_points_
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=color_f, scale_factor=0.0007)


def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # 現在のパス
        if root.count("/") == file_dir_.count("/") + 1:
            file_list.append(root)
        # print(dirs)  # 現在のディレクトリ内のサブディレクトリ
        # print(files)  # 現在のディレクトリ内のファイル
    file_list.sort()
    return file_list


def show_grasp_norm(grasp_bottom_center, grasp_axis):
    un1 = grasp_bottom_center - 0.25 * grasp_axis * 0.25
    un2 = grasp_bottom_center  # - 0.25 * grasp_axis * 0.05
    mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=(0, 1, 0), tube_radius=0.0005)


def show_pcl_norm(grasp_bottom_center, normal_, color="r", clear=False):
    if clear:
        plt.figure()
        plt.clf()
        plt.gcf()
        plt.ion()

    ax = plt.gca(projection="3d")
    un1 = grasp_bottom_center + 0.5 * normal_ * 0.05
    ax.scatter(un1[0], un1[1], un1[2], marker="x", c=color)
    un2 = grasp_bottom_center
    ax.scatter(un2[0], un2[1], un2[2], marker="^", c="g")
    ax.plot([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], "b-", linewidth=1)  # bi normal


def do_job(job_i):
    ii = np.random.choice(all_p.shape[0])
    show_grasp_norm(all_p[ii], surface_normal[ii])
    print("done job", job_i, ii)


if __name__ == "__main__":
    file_dir = os.environ["PointNetGPD_FOLDER"] + "/PointNetGPD/data/ycb-tools/models/ycb"
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7))
    file_list_all = get_file_name(file_dir)
    object_numbers = len(file_list_all)
    i = 10  # 表示するオブジェクトのインデックスです♪
    obj_path = str(file_list_all[i]) + "/google_512k/nontextured.obj"
    sdf_path = str(file_list_all[i]) + "/google_512k/nontextured.sdf"
    if os.path.exists(obj_path):
        of = ObjFile(obj_path)
        sf = SdfFile(sdf_path)
    else:
        print("objやsdfファイルが見つかりませんでした…😢")
        raise NameError("objやsdfファイルが見つかりません！")
    mesh = of.read()
    sdf = sf.read()
    graspable = GraspableObject3D(sdf, mesh)
    print("Log: オブジェクトを開きました！")
    begin_time = time.time()
    surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
    all_p = surface_points  # 元の点群
    method = "voxel"
    if method == "random":
        surface_points = surface_points[np.random.choice(surface_points.shape[0], 1000, replace=False)]
        surface_normal = []
    elif method == "voxel":
        surface_points = surface_points.astype(np.float32)
        # ★ open3dを使ってvoxelダウンサンプリングと法線推定を実施 ★
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points)
        voxel_size = graspable.sdf.resolution * 5
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        # downsample後の点群を更新
        all_p = np.asarray(pcd_down.points)
        # 法線推定（K近傍：knn=10）を行います
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
        pcd_down.orient_normals_consistent_tangent_plane(k=10)
        surface_normal = np.asarray(pcd_down.normals)
        pcd_down.normals = o3d.utility.Vector3dVector(-surface_normal)
        # ★ ここまでopen3dの処理でした ★

        use_open3d = True
        if use_open3d:
            show_grasp_norm(all_p[0], surface_normal[0])
            # マルチプロセスはmayaviとの相性が悪いのでループで回すよ♪
            sample_points = 500
            for _ in range(sample_points):
                do_job(_)
            mlab.pipeline.surface(mlab.pipeline.open(str(file_list_all[i]) + "/google_512k/nontextured.ply"), opacity=1)
            mlab.show()
            print("処理時間:", time.time() - begin_time, "秒")
    else:
        raise ValueError("No such method", method)

    use_meshpy = False
    if use_meshpy:
        normal = []
        # meshpyを使った法線計算の例です
        surface_points = surface_points[:100]
        for ind in range(len(surface_points)):
            p_grid = graspable.sdf.transform_pt_obj_to_grid(surface_points[ind])
            normal_tmp = graspable.sdf.surface_normal(p_grid)
            if normal_tmp is not None:
                normal.append(normal_tmp)
                show_grasp_norm(surface_points[ind], normal_tmp)
            else:
                print(len(normal))
        mlab.pipeline.surface(mlab.pipeline.open(str(file_list_all[i]) + "/google_512k/nontextured.ply"))
        mlab.show()
