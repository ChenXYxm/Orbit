import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d as o3d
from source.standalone.thesis.place_new_obj_Feb16 import place_new_obj_fun
import pickle as pkl
file_list = os.listdir("images/")
file_path = "images/"
filename = file_path + 'output_pointcloud2.pkl'
fileObject = open(filename, 'rb')
modelInput = pkl.load(fileObject)
fileObject.close()
#print(modelInput.shape)
#points = np.array(modelInput)
p_o3d = o3d.geometry.PointCloud()
points = o3d.utility.Vector3dVector(modelInput)
p_o3d.points = points
o3d.visualization.draw_geometries([p_o3d])
    