#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang
# E-mail     : liang@informatik.uni-hamburg.de
# Description:
# Date       : 22/05/2018 12:37 AM
# File Name  : read_file_sdf.py
import os
import multiprocessing
import subprocess

"""
This file convert obj file to sdf file automatically and multiprocessingly.
All the cores of a computer can do the job parallel.

pre request:
sudo apt install pcl-tools
"""
file_resolution = "google_512k"
# file_resolution = "google_16k"


def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count("/") == file_dir_.count("/") + 1:
            file_list.append(root)
        # print(dirs)  # all the directories in current path
        # print(files)  # all the files in current path
    file_list.sort()
    return file_list


def generate_sdf(path_to_sdfgen, obj_filename, dim, padding):
    """Converts mesh to an sdf object"""

    # create the SDF using binary tools
    sdfgen_cmd = '%s "%s" %d %d' % (path_to_sdfgen, obj_filename, dim, padding)
    os.system(sdfgen_cmd)
    print("SDF Command: %s" % sdfgen_cmd)
    return


def do_job_convert_obj_to_sdf(x):
    input_file = str(file_list_all[x]) + f"/{file_resolution}/nontextured.obj"
    if os.path.isfile(input_file):
        generate_sdf(path_sdfgen, input_file, 100, 5)  # for google scanner
    print("Done job number", x)


def generate_obj_from_ply(file_name_):
    base = file_name_.split(".")[0]
    if os.path.isfile(file_name_):
        p = subprocess.Popen(["pcl_ply2obj", base + ".ply", base + ".obj"])
        p.wait()


if __name__ == "__main__":
    home_dir = os.environ["PointNetGPD_FOLDER"]
    file_dir = home_dir + "/PointNetGPD/data/ycb-tools/models/ycb"  # for google ycb
    path_sdfgen = home_dir + "/SDFGen/bin/SDFGen"
    file_list_all = get_file_name(file_dir)
    object_numbers = file_list_all.__len__()
    if file_list_all == []:
        print(f"[ERROR] There is no file in {file_dir}, Please check your data folder")
    # generate obj from ply file
    for i in file_list_all:
        generate_obj_from_ply(i + f"/{file_resolution}/nontextured.ply")
        print("finish", i)
    # The operation for the multi core
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    pool.map(do_job_convert_obj_to_sdf, range(object_numbers))
