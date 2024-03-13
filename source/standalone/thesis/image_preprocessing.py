import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d as o3d
from source.standalone.thesis.place_new_obj_Feb16 import place_new_obj_fun
import pickle as pkl
file_list = os.listdir("images/image2/")
def get_tsdf(occu):
    shape_occu = occu.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    points_np = np.zeros((Nx*Ny,3)).astype(np.float32)
    for i in range(Nx):
        for j in range(Ny):
            num_id = Nx*i + j
            points_np[num_id][1] = 0.5+0.01*i
            points_np[num_id][0] = 0.5+0.01*j
            if occu[i,j] == 1:
                points_np[num_id][2] = 0.015
    points_obj_id = np.where(points_np[:,2]>=0.01)
    # print('points_obj_id')
    # print(points_obj_id)
    points_obj = points_np[points_obj_id[0]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_obj)
    x = np.linspace(0.5, 1.0, Nx)
    y = np.linspace(0.5, 1.0, Ny)
    xv, yv = np.meshgrid(x, y)

    grid = np.zeros((Nx*Ny,3))
    grid[:,0] = xv.flatten()
    grid[:,1] = yv.flatten()
    pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
    distance = pts_grid.compute_point_cloud_distance(pcd)
    dist = np.array(distance)
    # norm = mplc.Normalize(vmin=min(distance), vmax=max(distance), clip=True)
    Tsdf = dist.reshape(Ny,Nx)
    # Tsdf = np.fliplr(Tsdf)
    # plt.imshow(Tsdf)
    # plt.show()
    # print('pointcloud 3d')
    # print(pointcloud_w)
    # o3d.visualization.draw_geometries([pcd])
    return Tsdf
for _ in file_list:
    image = cv2.imread("images/image2/"+_, cv2.IMREAD_UNCHANGED)
    print(_)
    # image = cv2.imread("images/"+_)
    print(image.shape)
    print(np.max(image),np.min(image),np.mean(image))
    # print(np.max(image[:,:,1]),np.min(image[:,:,1]))
    # print(np.max(image[:,:,2]),np.min(image[:,:,2]))
    # print(image[65:226,100:300,2])
    # # Display the image
    # image[115:450,150:760] = 0
    # print(image[320,320])
    # image_tmp = image[115:450,150:760].copy
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    image_tmp = np.array(image)
    
    print(image_tmp.shape)
    print(np.max(image_tmp),np.min(image_tmp),np.mean(image_tmp))
    image_tmp=np.where(image_tmp < 1000.0, image_tmp, 1000.0)
    image_tmp=np.where(image_tmp > 710, image_tmp, 0.0)
    image_tmp=np.where(image_tmp < 710, image_tmp, 255)
    # image_tmp[150:180] = 5
    kernel1 = np.ones((7, 7), np.float32)/49
    print(np.max(image_tmp),np.min(image_tmp),np.mean(image_tmp))
    print(_)
    # cv2.imshow("Image", image_tmp)
    # cv2.waitKey(0)
    plt.imshow(image_tmp)
    plt.show()
    image_tmp = np.array(image_tmp[50:370,290:610])
    plt.imshow(image_tmp)
    plt.show()
    img_blur = cv2.filter2D(src=image_tmp,ddepth=-1,kernel = kernel1)
    # cv2.imshow("Image", img_blur)
    # cv2.waitKey(0)
    # print(a)
    img_blur=np.where(img_blur <100, 1, img_blur)
    img_blur=np.where(img_blur >=100, 0, img_blur)
    # new_size = (50,50)
    # img_blur = cv2.resize(img_blur,new_size)
    plt.imshow(img_blur)
    plt.show()
    # Tsdf = get_tsdf(img_blur)
    # plt.imshow(Tsdf)
    # plt.show()
    # print(Tsdf)
    # data = np.zeros((50,50,2))
    # data[:,:,0] = img_blur
    # data[:,:,1] = Tsdf
    # file_name = 'data.pkl'
    # file_path = "images/"+file_name
    # f_save = open(file_path,'wb')
    # pkl.dump(data,f_save)
    # f_save.close()
    # img_blur=np.where(img_blur <100, 0, img_blur)
    # img_blur=np.where(img_blur >=100, 1000, img_blur)
    # cv2.imshow("Image", img_blur)
    # cv2.waitKey(0)
    # img_blur = img_blur/255.0
    # cv2.imshow("Image", img_blur)
    # cv2.waitKey(0)
    w_obj = np.random.randint(low=5,high=15, size=(1))
    l_obj = np.random.randint(low=5,high=15, size=(1))
    print(w_obj,l_obj)
    obj_vertices = np.zeros((4,2))
    obj_vertices[0,0] = -int(w_obj)
    obj_vertices[0,1] = -int(l_obj)
    obj_vertices[1,0] = int(w_obj)
    obj_vertices[1,1] = -int(l_obj)
    obj_vertices[2,0] = int(w_obj)
    obj_vertices[2,1] = int(l_obj)
    obj_vertices[3,0] = -int(w_obj)
    obj_vertices[3,1] = int(l_obj)
    print(obj_vertices)

    # flag_found, new_poly_vetices,occu_tmp,new_obj_pos = place_new_obj_fun(img_blur,obj_vertices)
    # cv2.imshow("Image", img_blur)
    # cv2.waitKey(0)
    