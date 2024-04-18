#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf
import numpy as np
import open3d as o3d
import cv2
from stable_baselines3 import PPO
import torch
import pickle as pkl


def get_x_y(tsdf,occupancy):
    obs = np.zeros([1,50,50,2])
    obs[0,:,:,0] = occupancy.copy()
    obs[0,:,:,1] = tsdf.copy()
    checkpoint = './data/model.zip'
    agent = PPO.load(checkpoint)
    actions, _ = agent.predict(obs, deterministic=True)
    obs_tensor = torch.from_numpy(obs)
    # print(obs_tensor.size())
    obs_tensor = obs_tensor.permute(0,3,1,2)
    # print(obs_tensor.size())
    actions_tensor_tmp =  torch.from_numpy(actions)
    value,log_prob,entropy = agent.policy.evaluate_actions(obs_tensor,actions_tensor_tmp)
    # print('value log prob entropy')
    # print(value,log_prob,entropy)
    obs_tmp = obs.copy()
    obs_tensor_tmp = obs_tensor.detach().clone()
    act_app = np.zeros(len(obs))
    
    #print(image_name)
    occu = occupancy.copy()
    occu[actions.flatten()[0],actions.flatten()[1]] = 255
    image_name = "./data/point_cloud/tsdf_pushing_1.png"
    cv2.imwrite(image_name,occu)
    
    actions_new = np.c_[actions,act_app.T]   
    return actions_new 

class DepthImageTransformerTalker:
    def __init__(self):
        rospy.init_node('depth_image_transformer_talker', anonymous=True)
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.image_pub = rospy.Publisher('/transformed_depth_image', Image, queue_size=1)
        self.image_pub_occu = rospy.Publisher('/occu_image', Image, queue_size=1)
        self.occu_size = [50,50]
        self.save_path = './data/point_cloud/'
        #self.occu = np.zeros((50,50))
        # Subscribe to the depth image topic
        
        rospy.Subscriber('/wrist_camera/depth/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/wrist_camera/depth/image_rect_raw', Image, self.depth_image_callback)
        rospy.spin()
    def quaternion_to_matrix(self,q):
        x, y, z,w  = q
        self.R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        # return self.R
    def depth_pixel_to_camera(self,cv_image):
        [height, width] = cv_image.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)        
        x = (u.flatten() - self.camera_cx)/self.camera_fx
        #print(x[:400])
        y = (v.flatten() - self.camera_cy)/self.camera_fy
        
        ##########################################
        ## TODO: switch between real and simulation
        z = cv_image.flatten()/1000 ### real robot
        #z = cv_image.flatten() ### simulation
        ##########################################
        x = np.multiply(x,z)
        y = np.multiply(y,z)
        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]
        #print(np.max(x),np.min(x),np.max(y),np.min(y),np.max(z),np.min(z))
        points=np.zeros((x.shape[0],4))
        points[:,3] = 1
        points[:,0] = x.flatten()
        points[:,1] = y.flatten()
        points[:,2] = z.flatten()
        ########################################
        ## TODO: switch between real and simulation
        ########## real robot ##################
        #points = points[np.where(points[:,2]<1)]
        #print(np.max(points[:,0]),np.min(points[:,0]),np.max(points[:,1]),np.min(points[:,1]),np.max(points[:,2]),np.min(points[:,2]))
        #points = points[np.where(points[:,0]<0.395)]
        #points = points[np.where(points[:,0]>-0.22)]
        #points = points[np.where(points[:,1]<0.35)]
        #points = points[np.where(points[:,1]>-0.35)]
        ########################################
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        
        points_new_obj = points.copy()
        '''
        ##########################################
        ## TODO: switch between real and simulation
        ############# for real robot and fixed camera #############
        points = points[np.where(points[:,0]<0.288)]
        points = points[np.where(points[:,0]>-0.222)]
        points = points[np.where(points[:,1]<0.134)]
        points = points[np.where(points[:,1]>-0.36)]
        ##########################################
        '''
        
        ##########################################
        ## TODO: switch between real and simulation
        ############# for real robot and wrist camera #############
        points = points[np.where(points[:,0]<0.28)]  ### -y
        points = points[np.where(points[:,0]>-0.5)] ### +y
        points = points[np.where(points[:,1]<0.26)] ###-x
        points = points[np.where(points[:,1]>-0.38)] ### +x
        ##########################################
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud("/data/videos/experiment_1/pcd_camera_frame.ply", pcd)
        ##########################################
        
        '''
        ##########################################
        ###### get the shape info of new obj fixed camera#####
        points_new_obj = points_new_obj[np.where(points_new_obj[:,0]<-0.15)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,0]>-0.38)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,1]<0.15)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,1]>-0.290)]
        ##########################################
        '''
        ##########################################
        ###### get the shape info of new obj fixed camera#####
        points_new_obj = points_new_obj[np.where(points_new_obj[:,0]<0.25)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,0]>0.02)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,1]<0.18)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,1]>-0.27)]
        ##########################################
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(points_new_obj[:,:3])
        o3d.visualization.draw_geometries([pcd_obj])
        ##########################################
        
        
        return pcd, points,pcd_obj
    def camera_info_callback(self,msg):
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.camera_cx = self.camera_matrix[0,2]
        self.camera_cy = self.camera_matrix[1,2]
        self.camera_fx = self.camera_matrix[0,0]
        self.camera_fy = self.camera_matrix[1,1]
        #print('self.camera_cx')
        #print(self.camera_cx)
    def camera_to_world(self,world_points):
        
        for i in range(len(world_points)):
            world_points[i] = np.dot(self.R,world_points[i])+self.trans
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points)
        o3d.visualization.draw_geometries([pcd])
        print('save pcd')
        o3d.io.write_point_cloud("./data/videos/placing/experiment16/experiment_16_proposed.ply", pcd)
        # o3d.io.write_point_cloud("./data/videos/pre_point_cloud/pre_3_items_2.ply", pcd)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        plane_model_ori = plane_model
        plane_model = np.array([plane_model[0],plane_model[1],plane_model[2]]).reshape((-1,1))
        world_points=world_points[np.where(world_points[:,0]>=0.275)]
        world_points=world_points[np.where(world_points[:,0]<=0.755)]
        world_points=world_points[np.where(world_points[:,1]<=0.24)]
        world_points=world_points[np.where(world_points[:,1]>=-0.24)]
        
        print('world: ',np.max(world_points[:,0]),np.min(world_points[:,0]),np.max(world_points[:,1]),np.min(world_points[:,1]))
        self.max_x = np.max(world_points[:,0])
        self.min_x = np.min(world_points[:,0])
        self.max_y = np.max(world_points[:,1])
        self.min_y = np.min(world_points[:,1])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points)
        #o3d.io.write_point_cloud(self.save_path+"pcd_world_frame.ply", pcd)

        ####################### visualization
        obb = pcd.get_oriented_bounding_box()
        aabb = pcd.get_axis_aligned_bounding_box()
        obb.color = [1,0,0]
        aabb.color = [0,0,0]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd,aabb,axes])
        #o3d.io.write_point_cloud(self.save_path+"pcd_world2.ply", pcd)
        ########################################

        dist_point = (np.dot(world_points,plane_model) + plane_model_ori[3]).reshape(-1)
        inliers = np.array(np.where(dist_point<=0.015)).reshape(-1)
        # plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        self.pcd = pcd
        self.inlier_cloud = pcd.select_by_index(inliers)
        # Extract outliers (non-plane points)
        self.outlier_cloud = pcd.select_by_index(inliers, invert=True)
        # Visualize the results
        # print('table pcd')
        # o3d.visualization.draw_geometries([self.inlier_cloud])
        # o3d.visualization.draw_geometries([self.outlier_cloud])
        outlier_cloud_np = np.array(self.outlier_cloud.points)
        outlier_cloud_list = []
        ############################################
        A,B,C,D = plane_model_ori
        for i in range(len(outlier_cloud_np)):
            x,y,z = outlier_cloud_np[i]
            numerator = A * x + B * y + C * z + D
            distance = np.abs(numerator)/np.sqrt(A**2 + B**2 + C**2)
            if numerator > 0 and distance<0.3:
                outlier_cloud_list.append(outlier_cloud_np[i])
        outlier_cloud = o3d.geometry.PointCloud()    
        outlier_cloud_np = np.array(outlier_cloud_list).reshape(-1,3) 
        outlier_cloud_np[:,2] = 0 
        outlier_cloud.points = o3d.utility.Vector3dVector(outlier_cloud_np)
        # o3d.visualization.draw_geometries([outlier_cloud])
        self.outlier_cloud = outlier_cloud
        # Visualize the results
        
        return pcd
    def get_center_new_obj(self,pcd_obj):
        points_np = np.asarray(pcd_obj.points)
        for i in range(len(points_np)):
            points_np[i] = np.dot(self.R,points_np[i])+self.trans
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(points_np)
        plane_model, inliers = pcd_obj.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        obj_outlier_cloud = pcd_obj.select_by_index(inliers)
  
        obj_inlier_cloud = pcd_obj.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([obj_inlier_cloud])
        o3d.visualization.draw_geometries([obj_outlier_cloud])
        obj_inlier_cloud_np = np.array(obj_inlier_cloud.points)
        obj_inlier_cloud_list = []
        ############################################
        A,B,C,D = plane_model
        for i in range(len(obj_inlier_cloud_np)):
            x,y,z = obj_inlier_cloud_np[i]
            numerator = A * x + B * y + C * z + D
            distance = np.abs(numerator)/np.sqrt(A**2 + B**2 + C**2)
            if numerator > 0 and distance<0.3:
                obj_inlier_cloud_list.append(obj_inlier_cloud_np[i])
        obj_inlier_cloud = o3d.geometry.PointCloud()    
        obj_inlier_cloud_np = np.array(obj_inlier_cloud_list).reshape(-1,3) 
        obj_inlier_cloud_np[:,2] = 0 
        obj_inlier_cloud.points = o3d.utility.Vector3dVector(obj_inlier_cloud_np)
        print('visualize obj mask')
        o3d.visualization.draw_geometries([obj_inlier_cloud])
        print('point cloud center')
        aabb = obj_inlier_cloud.get_axis_aligned_bounding_box()
        obj_center = obj_inlier_cloud.get_center()
        aabb_center = aabb.get_center()
        print(obj_center,aabb_center)
        
        print('length')
        length = aabb.get_extent()
        print(length)
    def pcd_to_tsdf(self):
        Nx, Ny = self.occu_size
        x = np.linspace(self.min_x, self.max_x, Nx)
        y = np.linspace(self.min_y, self.max_y, Ny)
        xv, yv = np.meshgrid(x, y)
        grid = np.zeros((Nx*Ny,3))
        grid[:,0] = xv.flatten()
        grid[:,1] = yv.flatten()
        pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
        distance = pts_grid.compute_point_cloud_distance(self.outlier_cloud)
        dist = np.array(distance)
        Tsdf = dist.reshape(Ny,Nx)
        Tsdf = Tsdf*255
        Tsdf = np.fliplr(Tsdf)
        Tsdf = np.rot90(Tsdf)
        #print(np.max(Tsdf),np.min(Tsdf))
        return Tsdf
    def pcd_to_occu(self):
        Nx, Ny = self.occu_size
        pts = np.asarray(self.outlier_cloud.points)
        u = (pts[:,0] - self.min_x)/ ( self.max_x-self.min_x )
        v = (pts[:,1] - self.min_y)/ ( self.max_y-self.min_y )
        u = (Nx-1)*u
        v = (Ny-1)*v
        occupancy = np.zeros( (Ny,Nx) )
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        u_ind = np.where(u<Nx)
        u = u[u_ind]
        v = v[u_ind]
        v_ind = np.where(v<Ny)
        u = u[v_ind]
        v = v[v_ind]
        u_ind = np.where(u>=0)
        u = u[u_ind]
        v = v[u_ind]
        v_ind = np.where(v>=0)
        u = u[v_ind]
        v = v[v_ind]
        occupancy[v,u] = 1
        occupancy = occupancy*255
        occupancy = np.fliplr(occupancy)
        occupancy = np.rot90(occupancy)
        return occupancy
    def postprocessed_occu_to_tsdf(self,occu):
        Nx,Ny = self.occu_size
        
        occu_index = np.where(occu>=0.5)
        print('occu index',occu_index)
        occu_index_x = occu_index[0].flatten().astype(int)
        occu_index_y = occu_index[1].flatten().astype(int)
        point_x = (occu_index_x-0.5)*(self.max_x-self.min_x)/float(Nx)+self.min_x
        point_y = (occu_index_y-0.5)*(self.max_y-self.min_y)/float(Ny)+self.min_y
        xyz = np.zeros((len(point_x),3))
        xyz[:,0] = point_x
        xyz[:,1] = point_y
        obj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        x = np.linspace(self.min_x, self.max_x, Nx)
        y = np.linspace(self.min_y, self.max_y, Ny)
        xv, yv = np.meshgrid(x, y)
        grid = np.zeros((Nx*Ny,3))
        grid[:,0] = xv.flatten()
        grid[:,1] = yv.flatten()
        pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
        distance = pts_grid.compute_point_cloud_distance(obj_pcd)
        dist = np.array(distance)
        Tsdf = dist.reshape(Ny,Nx)
        Tsdf = (Tsdf-np.min(Tsdf))/np.max(Tsdf)*255
        Tsdf = np.fliplr(Tsdf)
        Tsdf = np.rot90(Tsdf)
        cv2.imwrite(self.save_path+'post_Tsdf.png',Tsdf)
        print('save image to ', self.save_path)
    def decode_image_pos(self):
        pass
    def add_salt_and_pepper_noise(self,image,salt_prob=0.01, pepper_prob=0.01):
        noisy_image = np.copy(image)
        
    
        num_salt = np.ceil(salt_prob * image.size)
        salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 255

    
        num_pepper = np.ceil(pepper_prob * image.size)
        pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0
        # cv2.imwrite(self.save_path+'occu_noise.png',noisy_image)
        # data_file_name = self.save_path+'occu_data_noisy.pkl'
        # fileObject = open(data_file_name,'wb')
        # pkl.dump(noisy_image,fileObject)
        # fileObject.close()
        
       
        data_file_name = self.save_path+'occu_data.pkl'
        fileObject = open(data_file_name,'wb')
        pkl.dump(image,fileObject)
        fileObject.close()
        
        Nx, Ny = self.occu_size
        x = np.linspace(self.min_x, self.max_x, Nx)
        y = np.linspace(self.min_y, self.max_y, Ny)
        xv, yv = np.meshgrid(x, y)
        grid = np.zeros((Nx*Ny,3))
        grid[:,0] = xv.flatten()
        grid[:,1] = yv.flatten()
        pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
        
        occupied_ind = np.where(noisy_image>100)
        #print(occupied_ind)
        occupied_ind = np.array(occupied_ind).reshape(2,-1).T
        #print(occupied_ind)
        np_points = np.zeros((len(occupied_ind),3))
        
        return noisy_image
    def image_processing(self,image):
        kernel = np.ones((3,3),np.float32)/9
        img_blur = cv2.filter2D(image,-1,kernel)
        img_blur=np.where(img_blur <125, 0, img_blur)
        img_blur=np.where(img_blur >=125, 255, img_blur)
        cv2.imwrite(self.save_path+'occu_filtered.png',img_blur)
        return img_blur
    def depth_image_callback(self, msg):
        try:
            # Transform pixel coordinates to camera coordinates
            # (Add your pixel to camera coordinate transformation code here)
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            print('receive new depth image: ', np.mean(cv_image))
            pcd, points,pcd_obj = self.depth_pixel_to_camera(cv_image)
            # Transform camera coordinates to world coordinates using tf
            camera_frame = "/wrist_camera_depth_optical_frame"  # Update with your camera frame
            world_frame = "/world"    # Update with your world frame

            (self.trans, self.quat) = self.listener.lookupTransform(world_frame, camera_frame, rospy.Time(0))
            #print(self.quat,self.trans)
            # get the rotation matrix
            self.quaternion_to_matrix(self.quat)
            #print(self.R)
            self.get_center_new_obj(pcd_obj)
            world_points = points[:,:3].copy()
            pcd_w = self.camera_to_world(world_points)
            tsdf = self.pcd_to_tsdf()
            tsdf_tmp = tsdf.copy()
            tsdf = tsdf.astype(np.uint8)
            tsdf = cv2.cvtColor(tsdf, cv2.COLOR_GRAY2BGR)
            tsdf_ros = self.bridge.cv2_to_imgmsg(tsdf, encoding="bgr8")
            # (Add your camera to world coordinate transformation code here)
            occu = self.pcd_to_occu()
            occu_tmp = occu.copy()
            occu = occu.astype(np.uint8)
            occu_ros = self.bridge.cv2_to_imgmsg(occu)

            # Now you can use the transformed depth image data as needed
            self.occu = occu.copy()
            occu[24:26,24:26] = 100
            #print('max occu real')
            #print(np.max(occu))
            # Print the transformed depth image data (replace this with your logic)
            #rospy.loginfo("Transformed Depth Image Data: %s", transformed_depth_image_data)

            # Publish the transformed depth image
            #self.image_pub.publish(tsdf_ros)
            #self.image_pub_occu.publish(occu_ros)
            cv2.imwrite(self.save_path+'occu.png',occu)
            cv2.imwrite(self.save_path+'tsdf.png',tsdf)
            # self.add_salt_and_pepper_noise(occu_tmp)
            self.image_processing(occu_tmp)
            action=get_x_y(tsdf_tmp,occu_tmp)
            self.postprocessed_occu_to_tsdf(occu_tmp)
            print('action')
            print(action)
            rospy.sleep(3)
            '''
            occu = self.get_occu()
            print('max occu')
            print(np.max(occu))
            '''
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
    def get_occu(self):
        return self.occu
if __name__ == '__main__':
    try:
        depth_image_transformer_talker = DepthImageTransformerTalker()
        rospy.spin()
        occu = depth_image_transformer_talker.get_occu()
        print('max occu')
        print(np.max(occu))
        
    except rospy.ROSInterruptException:
        pass

