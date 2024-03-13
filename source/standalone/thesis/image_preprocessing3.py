import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d as o3d
from source.standalone.thesis.place_new_obj_Feb16 import place_new_obj_fun
import pickle as pkl
file_list = os.listdir("images/image2/")
camera_K = np.array([[602.1849879340944, 0.0, 320.5],
                      [0.0, 602.1849879340944, 240.5],
                       [0.0, 0.0, 1.0]])


# camera_K = np.array([[428.9827880859375, 0.0, 423.6623229980469],
#                       [0.0, 428.9827880859375, 238.1645050048828],
#                        [0.0, 0.0, 1.0]])

print(camera_K)
print(camera_K.flatten())
camera_cx = camera_K[0,2]
camera_cy = camera_K[1,2]
camera_fx = camera_K[0,0]
camera_fy = camera_K[1,1]
print(camera_fx,camera_fy,camera_cx,camera_cy)
def quaternion_to_matrix(q):
    x, y, z,w  = q
    R = np.array([
    [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
    [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
    return R
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
# quat = np.array([-0.615, 0.003, 0.789,0.])
quat = np.array([0.708, 0.700, -0.053, -0.075])
quat = quat/np.linalg.norm(quat)
rot_m = quaternion_to_matrix(quat)
# rot_m = np.linalg.inv(rot_m)
transform_arr = np.array([0.876, -0.179, 1.284])
# print(rot_m)
# quat_optical_frame = np.array([0.500, 0.500, -0.500, 0.500])
# rot_m2 = quaternion_to_matrix(quat_optical_frame)
# rot_m = np.matmul(rot_m2,rot_m)
i_num = 0
for _ in file_list:
    i_num += 1
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
    normalize_x = (np.arange(image.shape[1])-camera_cx)/camera_fx
    normalize_y = (np.arange(image.shape[0])-camera_cy)/camera_fy
    
    #print(normalize_x,normalize_y)
    depth_points = np.zeros((image_tmp.shape[0]*image_tmp.shape[1],3))
    for i in range(image_tmp.shape[1]):
        for j in range(image_tmp.shape[0]):
            depth_info = image_tmp[j,i]/1000.0
            # print(depth_info)
            depth_points[i*image_tmp.shape[0]+j] = np.array([depth_info*normalize_x[i],depth_info*normalize_y[j],depth_info])
    # print(depth_points) 
            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth_points)
    o3d.visualization.draw_geometries([pcd])
    [height, width] = image_tmp.shape
    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_cx)/camera_fx
    #print(x[:400])
    y = (v.flatten() - camera_cy)/camera_fy
    #print(y[:600])
    print(x.shape,y.shape)
    
    z = image_tmp.flatten() / 1000
    x = np.multiply(x,z)
    y = np.multiply(y,z)
    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]
    x = x[np.where(z<1)]
    y = y[np.where(z<1)]
    z = z[np.where(z<1)]
    z = z[np.where(z<1)]

    print(np.max(x),np.min(x),np.max(y),np.min(y))
    points=np.zeros((x.shape[0],4))
    points[:,3] = 1
    points[:,0] = x.flatten()
    points[:,1] = y.flatten()
    points[:,2] = z.flatten()
    points = points[np.where(points[:,0]<0.38)]
    points = points[np.where(points[:,0]>-0.23)]
    points = points[np.where(points[:,1]<0.22)]
    points = points[np.where(points[:,1]>-0.39)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    o3d.visualization.draw_geometries([pcd])

    world_points = points[:,:3].copy()
    for i in range(len(world_points)):
        world_points[i] = np.dot(rot_m,world_points[i])+transform_arr
    print(np.max(world_points[:,0]),np.min(world_points[:,0]),np.max(world_points[:,1]),np.min(world_points[:,1]))
    
    # world_points = world_points[np.where(world_points[:,0]<0.15)]
    # world_points = world_points[np.where(world_points[:,0])<0.15]
    print(world_points[:400])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points)
    obb = pcd.get_oriented_bounding_box()
    # mobb = pcd.get_minimal_oriented_bounding_box()
    aabb = pcd.get_axis_aligned_bounding_box()
    obb.color = [1,0,0]
    aabb.color = [0,0,0]
    # mobb = [0,0,1]
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    # axes.color = [0,0,1]
    print("Oriented Bounding Box:")
    print("Center:", obb.get_center())
    print("Extent (length x width x height):", obb.extent)
    print("Rotation matrix:")
    print(obb.R)
    print(obb.get_max_bound())
    print(obb.get_min_bound())
    print(obb.volume())
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    print("Axis-Aligned Bounding Box:")
    print("Min Bound:", min_bound)
    print("Max Bound:", max_bound)
    obb.scale(scale=1.0,center= obb.get_center())
    o3d.visualization.draw_geometries([pcd,obb,aabb,axes])
    # height_map = world_points[:, 2].reshape((480, 848))
    # plt.imshow(height_map)
    # plt.show()
    height_map = np.zeros((200, 200))
    proj_matrix = obb.R
    proj_points = np.dot(world_points - obb.get_center(), proj_matrix)
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(proj_points)
    projected_obb = projected_pcd.get_oriented_bounding_box()
    projected_obb.color = [0,1,0]
    o3d.visualization.draw_geometries([pcd,obb,aabb,axes,projected_pcd,projected_obb])
    proj_points += np.abs(proj_points.min(axis=0))
    proj_points *= 200 / proj_points.max(axis=0)
    proj_points = proj_points.astype(int)
    proj_points[proj_points < 0] = 0
    proj_points[proj_points >= 200] = 199

    # Update height map
    height_map[proj_points[:, 0], proj_points[:, 1]] = points[:, 2]
    plt.imshow(height_map, cmap='viridis')
    plt.title('Height Map')
    plt.colorbar()
    plt.show()
    height_map_tmp = height_map.copy()
    height_map_tmp[np.where(height_map_tmp>0)] = 1
    plt.imshow(height_map_tmp, cmap='viridis')
    plt.title('Height Map')
    plt.colorbar()
    plt.show()

    # rotation_matrix = obb.R
    # translation_vector = obb.center
    # transformation_matrix = np.eye(4)
    # transformation_matrix[:3, :3] = rotation_matrix
    # transformation_matrix[:3, 3] = translation_vector

    # # Transform the point cloud to the OBB frame
    # # transformation_matrix = np.eye(4)
    # # transformation_matrix[:3, :3] = rotation_matrix
    # # transformation_matrix[:3, 3] = translation_vector

    # # Transform the point cloud to the OBB frame
    # point_cloud_transformed = pcd.transform(transformation_matrix)

    # # point_cloud_transformed = pcd.transform(np.linalg.inv(np.hstack([rotation_matrix, translation_vector.reshape(-1, 1)])))

    # # Extract the height map from the transformed point cloud
    # proj_matrix = rotation_matrix.T[:2, :2]
    # proj_points = np.dot(np.asarray(point_cloud_transformed.points)[:, :2], proj_matrix)
    # proj_points += np.abs(proj_points.min(axis=0))
    # proj_points *= 100 / proj_points.max(axis=0)

    # # Round to integers to use as indices in the height map
    # proj_points = proj_points.astype(int)

    # # Ensure indices are within reasonable bounds
    # proj_points[proj_points < 0] = 0
    # proj_points[proj_points >= 100] = 99

    # # Update height map
    # height_map = np.zeros((100, 100))
    # height_map[proj_points[:, 0], proj_points[:, 1]] = np.asarray(point_cloud_transformed.points)[:, 2]

    # # Display the height map
    
    # plt.imshow(height_map, cmap='viridis')
    # plt.title('Height Map')
    # plt.colorbar()
    # plt.show()
    if i_num >=1:
        break
    
    # depth_points2 = np.zeros(((image_tmp.shape[0])*(image_tmp.shape[1]),3))
    # for i in range(image_tmp.shape[0]):
    #     for j in range(image_tmp.shape[1]):
    #         depth_info = image_tmp[i,j]/1000.0
    #         # print(depth_info)
    #         ind = int((i)*image_tmp.shape[1]+j)
    #         depth_points[ind] = np.array([depth_info*normalize_x[i],depth_info*normalize_y[j],depth_info])
    # # print(depth_points) 
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(depth_points)
    # o3d.visualization.draw_geometries([pcd])
    # print(np.mean(depth_points),np.max(depth_points),np.min(depth_points))       
    # image_tmp=np.where(image_tmp < 1000.0, image_tmp, 1000.0)
    # image_tmp=np.where(image_tmp > 710, image_tmp, 0.0)
    # image_tmp=np.where(image_tmp < 710, image_tmp, 255)
    # # image_tmp[150:180] = 5
    # kernel1 = np.ones((7, 7), np.float32)/49
    # print(np.max(image_tmp),np.min(image_tmp),np.mean(image_tmp))
    # print(_)
    # # cv2.imshow("Image", image_tmp)
    # # cv2.waitKey(0)
    # plt.imshow(image_tmp)
    # plt.show()
    # image_tmp = np.array(image_tmp[50:370,290:610])
    # plt.imshow(image_tmp)
    # plt.show()
    # img_blur = cv2.filter2D(src=image_tmp,ddepth=-1,kernel = kernel1)
    # # cv2.imshow("Image", img_blur)
    # # cv2.waitKey(0)
    # # print(a)
    # img_blur=np.where(img_blur <100, 1, img_blur)
    # img_blur=np.where(img_blur >=100, 0, img_blur)
    # # new_size = (50,50)
    # # img_blur = cv2.resize(img_blur,new_size)
    # plt.imshow(img_blur)
    # plt.show()
    # # Tsdf = get_tsdf(img_blur)
    # # plt.imshow(Tsdf)
    # # plt.show()
    # # print(Tsdf)
    # # data = np.zeros((50,50,2))
    # # data[:,:,0] = img_blur
    # # data[:,:,1] = Tsdf
    # # file_name = 'data.pkl'
    # # file_path = "images/"+file_name
    # # f_save = open(file_path,'wb')
    # # pkl.dump(data,f_save)
    # # f_save.close()
    # # img_blur=np.where(img_blur <100, 0, img_blur)
    # # img_blur=np.where(img_blur >=100, 1000, img_blur)
    # # cv2.imshow("Image", img_blur)
    # # cv2.waitKey(0)
    # # img_blur = img_blur/255.0
    # # cv2.imshow("Image", img_blur)
    # # cv2.waitKey(0)
    # w_obj = np.random.randint(low=5,high=15, size=(1))
    # l_obj = np.random.randint(low=5,high=15, size=(1))
    # print(w_obj,l_obj)
    # obj_vertices = np.zeros((4,2))
    # obj_vertices[0,0] = -int(w_obj)
    # obj_vertices[0,1] = -int(l_obj)
    # obj_vertices[1,0] = int(w_obj)
    # obj_vertices[1,1] = -int(l_obj)
    # obj_vertices[2,0] = int(w_obj)
    # obj_vertices[2,1] = int(l_obj)
    # obj_vertices[3,0] = -int(w_obj)
    # obj_vertices[3,1] = int(l_obj)
    # print(obj_vertices)

    # flag_found, new_poly_vetices,occu_tmp,new_obj_pos = place_new_obj_fun(img_blur,obj_vertices)
    # cv2.imshow("Image", img_blur)
    # cv2.waitKey(0)
    