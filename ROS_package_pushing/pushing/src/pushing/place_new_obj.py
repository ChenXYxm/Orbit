import numpy as np
# import matplotlib.pyplot as plt
import cv2
from shapely import Polygon, STRtree, area, contains, buffer, intersection,get_coordinates, concave_hull
# from shapely.geometry import 
import rospy
from scipy.ndimage import rotate
################## get the centers of the bboxes for objects on table 
def get_center_range(occu:np.array):
    mask = occu.copy()
    shape_occu = occu.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    ############## transform mask to opencv image form
    mask = np.array((mask-np.min(mask))*255/(np.max(mask)-np.min(mask)),dtype=np.uint8)
    ret,mask = cv2.threshold(mask,50,255,0)
    ############## get contour
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ############## get bboxes for the contours
    shape_dict = dict()
    i = 0
    polygons = []
    occu_tmp = occu.copy()
    for cnt in contours:
        i += 1
        approx = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(approx)
        approx = np.int0(box)
        if len(approx) >=2:
            # img = cv2.drawContours(img,[approx],-1,(0,255,255),3)

            # print(approx)
            approx = approx.reshape((-1,2))
            # print(approx)
            shape_dict[i] = approx
            polygons.append(Polygon(approx))
    ############## remove holes
    if len(polygons)>=1:
        tree_ori = STRtree(polygons)
        del_ind = []
        for i,poly in enumerate(polygons):
            poly_tmp = buffer(poly,distance=1)
            # print(poly_tmp,poly)
            if i == 0:
                polygons_tmp = polygons[1:]
            elif i == len(polygons)-1:
                polygons_tmp = polygons[:-1]
            else:
                polygons_tmp = polygons[:i] +polygons[i+1:]
            tree = STRtree(polygons_tmp)
            indice = tree.query(poly_tmp, predicate="contains").tolist()
            if len(indice) >0:
                for j in indice:
                    if contains(poly_tmp,tree.geometries.take(j)) and area(poly_tmp)>area(tree.geometries.take(j)):
                        if j >= i:
                            j_tmp = j+1
                        else:
                            j_tmp = j
                        if j_tmp+1 in shape_dict:
                            # print("show remove")
                            # print(j_tmp)
                            del(shape_dict[j_tmp+1])
                            del_ind.append(int(j_tmp))
        #################### get centers of the masks
        polygons = []
        occu_tmp = occu.copy()
        centers = []
        for i in shape_dict:
            poly_tmp = Polygon(shape_dict[i])
            if poly_tmp.is_valid:
                polygons.append(poly_tmp)
                centers.append(np.asarray(poly_tmp.centroid.coords))

################## get the bbox info of the object mask
def get_new_obj_contour_bbox(occu:np.array):
    mask = occu.copy()
    shape_occu = occu.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    mask = np.array((mask-np.min(mask))*255/(np.max(mask)-np.min(mask)),dtype=np.uint8)
    ret,mask = cv2.threshold(mask,50,255,0)
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    cnt = []
    for i in contours:
        area_tmp = cv2.contourArea(i)
        if area_tmp>max_area:
            max_area = area_tmp
            cnt = i
    # approx = cv2.minAreaRect(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    # print(approx)
    # box = cv2.boxPoints(approx)
    # approx = np.int0(box)
    if x+w >=occu.shape[1]:
        w = occu.shape[1]-x-1
    if y+h >=occu.shape[0]:
        h = occu.shape[0]-1-y
    approx = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    vertices_new_obj = []
    mask_tmp = mask.copy()
    if len(approx) >=2:
        approx = approx.reshape((-1,2))
        for i in range(len(approx)):
            mask_tmp[approx[i][1],approx[i][0]] = 130
        vertices_new_obj = approx 
        # print(vertices_new_obj)
        vertices_new_obj = vertices_new_obj - np.array([Nx/2,Ny/2])
        # print(vertices_new_obj)
        # plt.imshow(mask_tmp)
        # plt.show()
        
        l = []
        for i in range(2):
            l.append(np.linalg.norm(vertices_new_obj[i]-vertices_new_obj[i+1]))
        # print(l)
        return vertices_new_obj
    else:
        return None
def draw_bbox(occu_ori):
    shape_occu = occu_ori.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    # print(shape_occu)
    num_check_edge = 0
    occu = occu_ori.copy()
    occu = np.array(occu*255,dtype=np.uint8)
    ret,thresh = cv2.threshold(occu,50,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of contours detected:",len(contours))
    shape_dict = dict()
    i = 0
    polygons = []
    occu_tmp = occu_ori.copy()
    for cnt in contours:
        i += 1
        approx = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(approx)
        approx = np.int0(box)
        if len(approx) >=2:
            # img = cv2.drawContours(img,[approx],-1,(0,255,255),3)

            # print(approx)
            approx = approx.reshape((-1,2))
            # print(approx)
            shape_dict[i] = approx
            polygons.append(Polygon(approx))
    if len(polygons)>=1:
        tree_ori = STRtree(polygons)
        del_ind = []
        for i,poly in enumerate(polygons):
            poly_tmp = buffer(poly,distance=1)
            # print(poly_tmp,poly)
            if i == 0:
                polygons_tmp = polygons[1:]
            elif i == len(polygons)-1:
                polygons_tmp = polygons[:-1]
            else:
                polygons_tmp = polygons[:i] +polygons[i+1:]
            tree = STRtree(polygons_tmp)
            indice = tree.query(poly_tmp, predicate="contains").tolist()
            if len(indice) >0:
                for j in indice:
                    if contains(poly_tmp,tree.geometries.take(j)) and area(poly_tmp)>area(tree.geometries.take(j)):
                        if j >= i:
                            j_tmp = j+1
                        else:
                            j_tmp = j
                        if j_tmp+1 in shape_dict:
                            # print("show remove")
                            # print(j_tmp)
                            del(shape_dict[j_tmp+1])
                            del_ind.append(int(j_tmp))
        polygons = []
        occu_tmp = occu_ori.copy()
        for i in shape_dict:
            polygons.append(Polygon(shape_dict[i]))
            for j in range(len(shape_dict[i])):
                if shape_dict[i][j][1] >= occu_tmp.shape[0]:
                    shape_dict[i][j][1] = occu_tmp.shape[0]-1
                if shape_dict[i][j][0] >= occu_tmp.shape[1]:
                    shape_dict[i][j][0] = occu_tmp.shape[1]-1
                if shape_dict[i][j][1] < 0:
                    shape_dict[i][j][1] = 0
                if shape_dict[i][j][0] < 0:
                    shape_dict[i][j][0] = 0    
                occu_tmp[shape_dict[i][j][1],shape_dict[i][j][0]] = 3
    for i in shape_dict:
        points_tmp = shape_dict[i].copy()
        points_tmp[:,0] = points_tmp[:,1]
        points_tmp[:,1] = shape_dict[i][:,0].copy()
        for j in range(len(points_tmp)):
            p_s = points_tmp[j]
            if j < len(points_tmp)-1:
                p_e = points_tmp[j+1]
            else:
                p_e = points_tmp[0]
            line = p_e - p_s
            length = np.linalg.norm(line)
            # print(p_s,p_e,length)
            for k in range(int(np.ceil(length))):
                tmp_delta = [k*line[0]/length,k*line[1]/length]
                for _,l in enumerate(tmp_delta):
                    if l >=0:
                        tmp_delta[_] = np.ceil(l)
                    else:
                        tmp_delta[_] = np.floor(l)
                occu_tmp[int(np.round(p_s[0]+tmp_delta[0])),int(np.round(p_s[1]+tmp_delta[1]))] = 3
                    
    return occu_tmp
def rotate_45(matrix,degree=45):
    # Get dimensions of the matrix
    rows, cols = matrix.shape

    # Create a rotated matrix with zeros
    rotated_matrix = np.zeros((rows, cols), dtype=matrix.dtype)

    # Rotate by -45 degrees (to simulate a 45-degree clockwise rotation)
    rotated = rotate(matrix, -degree, reshape=False, order=1, mode='constant', cval=0, prefilter=False)

    # Extract the center part of the rotated matrix
    center = (rotated.shape[0] - rows) // 2
    rotated_matrix = rotated[center:center + rows, center:center + cols]
    rotated_matrix[np.where(rotated_matrix<0.5)] = 0
    rotated_matrix[np.where(rotated_matrix>=0.5)] = 1
    return rotated_matrix
def placing_compare_fun(occu_ori,mask):
    # mask = np.zeros((40,40))
    # w_m = np.random.randint(2,10)
    # l_m = np.random.randint(2,10)
    # mask[20-w_m:20+w_m,20-l_m:20+l_m] = 1
    # plt.imshow(mask)
    # plt.show()
    occu = np.zeros((54,54))
    occu[2:52,2:52] = occu_ori.copy()
    occu_w,occu_l = occu.shape
    # print("occu shape: ", occu_l,occu_w)
    target_pos = [0,0,0]
    flag_found = False
    # plt.imshow(mask)
    # plt.show()
    mask_tmp = mask.copy()
    mask_diff_x = np.diff(mask_tmp)
    ind_xy = np.where(mask_diff_x==1)
    ind_x = ind_xy[0]
    ind_y = ind_xy[1]
    ind_x = np.array(ind_x).reshape(-1)
    ind_y = np.array(ind_y).reshape(-1)
    for i in range(len(ind_x)):
        mask[ind_x[i],ind_y[i]-2:ind_y[i]+1] = 1
    # plt.imshow(mask)
    # plt.show()
    ind_xy = np.where(mask_diff_x==-1)
    ind_x = ind_xy[0]
    ind_y = ind_xy[1]
    ind_x = np.array(ind_x).reshape(-1)
    ind_y = np.array(ind_y).reshape(-1)
    for i in range(len(ind_x)):
        mask[ind_x[i],ind_y[i]+1:ind_y[i]+4] = 1
    # plt.imshow(mask)
    # plt.show()
    mask_tmp = mask.copy()
    mask_diff_x = np.diff(mask_tmp, axis=0)
    ind_xy = np.where(mask_diff_x==1)
    ind_x = ind_xy[0]
    ind_y = ind_xy[1]
    ind_x = np.array(ind_x).reshape(-1)
    ind_y = np.array(ind_y).reshape(-1)
    for i in range(len(ind_x)):
        mask[ind_x[i]-2:ind_x[i]+1,ind_y[i]] = 1
    # plt.imshow(mask)
    # plt.show()
    ind_xy = np.where(mask_diff_x==-1)
    ind_x = ind_xy[0]
    ind_y = ind_xy[1]
    ind_x = np.array(ind_x).reshape(-1)
    ind_y = np.array(ind_y).reshape(-1)
    for i in range(len(ind_x)):
       mask[ind_x[i]+1:ind_x[i]+4,ind_y[i]] = 1
    # plt.imshow(mask)
    # plt.show()
    # ind_xy = np.where(mask>0)
    # s_x = int(np.min(np.array(ind_xy[0]).astype(int)))-3
    # e_x = int(np.max(np.array(ind_xy[0]).astype(int))+4)
    # s_y = int(np.min(np.array(ind_xy[1]).astype(int)))-3
    # e_y = int(np.max(np.array(ind_xy[1]).astype(int))+4)
    # mask[s_x:e_x,s_y:e_y] = 1
    # plt.imshow(mask)
    # plt.show()
    # print('original mask shape',mask.shape)
    for i in range(360):
        occu_tmp = occu.copy()
        mask_tmp = mask.copy()
        theta = 1*i
        mask_tmp = rotate_45(mask,degree=theta)
        w_mask_tmp,l_mask_tmp = mask_tmp.shape
        # plt.imshow(mask_tmp)
        # plt.show()
        ind_xy = np.where(mask_tmp>0)
        # print('length ind_xy',len(ind_xy[0]))
        for j in range(len(ind_xy[0])):
            if ind_xy[0][j] <0:
                ind_xy[0][j] = w_mask_tmp + ind_xy[0][j]
                # print('ind_xy',ind_xy[0][j])
            if ind_xy[1][j] <0:
                ind_xy[1][j] = l_mask_tmp + ind_xy[1][j]
        s_x = int(np.min(np.array(ind_xy[0])))
        if s_x <0:
            s_x = 0
        e_x = int(np.max(np.array(ind_xy[0]))+1)
        if e_x >w_mask_tmp:
            e_x = w_mask_tmp
        
        s_y = int(np.min(np.array(ind_xy[1])))
        if s_y <0:
            s_y = 0
        e_y = int(np.max(np.array(ind_xy[1]))+1)
        if e_y >l_mask_tmp:
            e_y = l_mask_tmp
        mask_tmp = mask_tmp[s_x:e_x,s_y:e_y]
        l_m_tmp = -s_y+e_y
        w_m_tmp = -s_x+e_x
        # image_name = "./data/point_cloud/rotatemask_"+str(i)+".png"
        # cv2.imwrite(image_name,mask_tmp*255)
        # print('mask',w_m_tmp,l_m_tmp)
        # print('mask shape',mask_tmp.shape,s_x,e_x,s_y,e_y)
        # plt.imshow(mask_tmp)
        # plt.show()
        for j in range(occu_w-w_m_tmp):
            
            for k in range(occu_l-l_m_tmp):
                # print('start:',j,k,'target area',j+w_m_tmp,k+l_m_tmp)
                result = np.max(mask_tmp+occu_tmp[j:j+w_m_tmp,k:k+l_m_tmp])
                if result ==1:
                    occu_tmp[j:j+w_m_tmp,k:k+l_m_tmp] = mask_tmp*2
                    flag_found = True
                    # print(occu_tmp)
                    # plt.imshow(occu_tmp)
                    # plt.show()
                    # cv2.imwrite(image_name,mask_tmp*255/2)
                    target_pos = [int(j+w_m_tmp/2),int(k+l_m_tmp/2),np.deg2rad(theta)]
                    return(flag_found,target_pos) 
                    break
            if flag_found:
                break
        if flag_found:
            break
    return flag_found,target_pos
def place_new_obj_fun(occu_ori,new_obj):
    ######################## input format
    # occu_ori: numpy 2d array: binary 0,1
    # new_obj: numpy 2d array: vertices of bbox in relative to the center point
    ########################
    shape_occu = occu_ori.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    # print(shape_occu)
    # print(Nx,Ny)
    num_check_edge = 0
    occu = occu_ori.copy()
    bbox = []
    for i in range(2):
        bbox.append(np.linalg.norm(new_obj[i]-new_obj[i+1]))
    num_grid_l = int(np.ceil(np.max(np.array(bbox))))
    num_grid_s = int(np.ceil(np.min(np.array(bbox))))
    occu = np.array(occu*255,dtype=np.uint8)
    ret,thresh = cv2.threshold(occu,50,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of contours detected:",len(contours))
    shape_dict = dict()
    i = 0
    ori_polygons = []
    polygons = []
    table_polygon = Polygon(np.array([[0,0],[Nx-1,0],[Nx-1,Ny-1],[0,Ny-1]]))
    new_lines = []
    
    for cnt in contours:
        i += 1
        defect_points = []
        defect_vertices_points = []
        # approx = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(approx)
        # approx = np.int0(box)
        occu_tmp = occu_ori.copy()
        # occu_tmp2 = occu_ori.copy()
        # occu_tmp1 = occu_ori.copy()
        cnt_tmp = cnt.copy()
        cnt_tmp = cnt_tmp.reshape(-1,2)
        if len(cnt_tmp)>=3:
            polygon_tmp = Polygon(cnt_tmp)
            if polygon_tmp.area>3:
                ori_polygons.append(polygon_tmp)
        ############################# get extract lines ################
        hull = cv2.convexHull(cnt,returnPoints = True)
        # print(hull)
        approx = np.int0(hull)
        # print('approx',approx)
        hull_2 = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull_2)
        # print('defects: ',defects,i)
        if len(approx) >=3:
            # img = cv2.drawContours(img,[approx],-1,(0,255,255),3)

            # print(approx)
            approx = approx.reshape((-1,2))
            # for i in range(len(approx)):
            #     if approx[i,0] == 649:
            #         continue
            
            # print('approx',approx)
            #shape_dict[i] = approx
            polygon_tmp2 = Polygon(approx)
            if polygon_tmp2.area>3:
                polygons.append(polygon_tmp2)
        #concave_hull_tmp = concave_hull(polygon_tmp)
        if defects is not None: 
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = cnt[s][0]
                end = cnt[e][0]
                far = cnt[f][0]
                #print('distance: ',d)
                defect_points.append(far)
                defect_vertices_points.append(start)
                defect_vertices_points.append(end)
                # print('start: ',start,'end: ',end,'far: ',far)
                p1 = np.array(start).flatten()
                p2 = np.array(end).flatten()
                p3 = np.array(far).flatten()
                distance = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
                # print('distance: ',distance)
                # print(type(cnt))
                # print(s,e,f)
                cnt_tmp = cnt.copy()
                cnt_tmp = cnt_tmp.reshape(-1,2)
                if np.abs(distance) > 2:
                    if s<e:
                        new_lines.append(cnt_tmp[s:e+1])
                        # print(cnt_tmp[s:e+1])
                    else:
                        points = cnt_tmp[s:-1]
                        points = np.concatenate((points,cnt_tmp[e].reshape(-1,2)),axis=0)
                        new_lines.append(points)
                        # print(points)
                # print(cnt_tmp[s:e+1])
        # defect_vertices_points = [np.array([point]).flatten() for point in defect_vertices_points]  
        # print('defect_vertices_points',defect_vertices_points)  
        # defect_contours = [np.array([point]).flatten() for point in defect_points]
        # poly_points_tmp = np.array(np.round(np.array(get_coordinates(polygon_tmp2))),dtype=np.int16)

        # print('contour: ',cnt)
        # print('contour2: ',cnt_tmp)
        # for j in range(len(cnt_tmp)):
        #     occu_tmp[cnt_tmp[j,1],cnt_tmp[j,0]] = 3
        # for j in range(len(poly_points_tmp)):
        #     occu_tmp2[poly_points_tmp[j,1],poly_points_tmp[j,0]] = 3
        # for j in range(len(defect_points)):
        #     occu_tmp1[defect_contours[j][1],defect_contours[j][0]] = 3
        # for j in range(len(defect_vertices_points)):
        #     occu_tmp1[defect_vertices_points[j][1],defect_vertices_points[j][0]] = 4
        # fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(7, 4))
        # ax1.imshow(occu_tmp)
        # ax2.imshow(occu_tmp2)
        # ax3.imshow(occu_tmp1)
        # plt.show()

        # hull = cv2.convexHull(cnt,returnPoints = True)
        # # print(hull)
        # hull_2 = cv2.convexHull(cnt,returnPoints = False)
        # defects = cv2.convexityDefects(cnt, hull_2)
        
        
    # for cnt in contours:
    #     i += 1
    #     # approx = cv2.minAreaRect(cnt)
    #     # box = cv2.boxPoints(approx)
    #     # approx = np.int0(box)
    #     #######################
    #     # approx = cv2.minAreaRect(cnt)
    #     # box = cv2.boxPoints(approx)
    #     # approx = np.int0(box)
    #     # if len(approx) >=2:
    #     #     # img = cv2.drawContours(img,[approx],-1,(0,255,255),3)

    #     #     # print(approx)
    #     #     approx = approx.reshape((-1,2))
    #     #     # print(approx)
    #     #     shape_dict[i] = approx
    #     #     polygons.append(Polygon(approx))
    #     ############################
        
    #     hull = cv2.convexHull(cnt,returnPoints = True)

    #     # print(hull)
    #     approx = np.int0(hull)
    #     if len(approx) >=3:
    #         # img = cv2.drawContours(img,[approx],-1,(0,255,255),3)

    #         # print(approx)
    #         approx = approx.reshape((-1,2))
    #         # for i in range(len(approx)):
    #         #     if approx[i,0] == 649:
    #         #         continue

    #         print('approx',approx)
    #         shape_dict[i] = approx
    #         polygon_tmp = Polygon(approx)
    #         polygon_tmp_2 = table_polygon.intersection(polygon_tmp)
    #         if polygon_tmp_2.area>0:
    #             polygons.append(polygon_tmp_2)
    # print('number of polygons',len(polygons))
    if len(polygons)>=1:
        
        # del_ind = []
        # for i,poly in enumerate(polygons):
        #     poly_tmp = buffer(poly,distance=1)
        #     # print(poly_tmp,poly)
        #     if i == 0:
        #         polygons_tmp = polygons[1:]
        #     elif i == len(polygons)-1:
        #         polygons_tmp = polygons[:-1]
        #     else:
        #         polygons_tmp = polygons[:i] +polygons[i+1:]
        #     tree = STRtree(polygons_tmp)
        #     indice = tree.query(poly_tmp, predicate="contains").tolist()
        #     if len(indice) >0:
        #         for j in indice:
        #             if contains(poly_tmp,tree.geometries.take(j)) and area(poly_tmp)>area(tree.geometries.take(j)):
        #                 if j >= i:
        #                     j_tmp = j+1
        #                 else:
        #                     j_tmp = j
        #                 if j_tmp+1 in shape_dict:
        #                     # print("show remove")
        #                     # print(j_tmp)
        #                     del(shape_dict[j_tmp+1])
        #                     del_ind.append(int(j_tmp))
        shape_dict = dict()
        # polygons_tmp = polygons.copy()
        # # print(del_ind)
        # # print(polygons_tmp)
        # polygons = []
        # for i in range(len(polygons_tmp)):
        #     if i not in del_ind:
        #         polygons.append(polygons_tmp[i])
        # print('number of polygon after filter: ',len(polygons))
        # occu_tmp = occu_ori.copy()
        for i in range(len(polygons)):
            # print(polygons[i])
            
            # np.asarray(polygons[i])
            # print(polygons[i].coords)
            # poly_points = np.array(polygons[i]).reshape(-1)
            # print(poly_points)
            # print(get_coordinates(polygons[i]))
            # print(polygons[i])
            poly_points_tmp = np.array(np.round(np.array(get_coordinates(polygons[i]))),dtype=np.int16)

            shape_dict[i] = poly_points_tmp.copy()
            # print(shape_dict)
            for j in range(len(shape_dict[i])):
                ## TODO
                if shape_dict[i][j][1] >= Nx:
                    shape_dict[i][j][1] = Nx-1
                    # print(shape_dict[i][j][1])
                if shape_dict[i][j][0] >= Ny:
                    shape_dict[i][j][0] = Ny-1
                    # print(shape_dict[i][j][0])
                if shape_dict[i][j][1] < 0:
                    shape_dict[i][j][1] = 0
                    # print(shape_dict[i][j][1])
                if shape_dict[i][j][0] < 0:
                    shape_dict[i][j][0] = 0 
                    # print(shape_dict[i][j][0])   
                # occu_tmp[shape_dict[i][j][1],shape_dict[i][j][0]] = 3
        # print('shape mask')
        # occu_tmp1 = occu_tmp.copy()
        # plt.imshow(occu_tmp)
        # plt.show()
        # print(shape_dict)
        length_dict = dict()
        length_list = []
        # occu_tmp = np.array(occu).copy()
        for i in shape_dict:
            points_tmp = shape_dict[i].copy()
            points_tmp[:,0] = points_tmp[:,1]
            points_tmp[:,1] = shape_dict[i][:,0].copy()
            for j in range(len(points_tmp)-1):
                p_s = points_tmp[j]
                # if j < len(points_tmp)-1:
                #     p_e = points_tmp[j+1]
                # else:
                #     p_e = points_tmp[0]
                p_e = points_tmp[j+1]
                line = p_e - p_s
                length = np.linalg.norm(line)
                if length>1:
                    if length not in length_dict:
                        length_dict[length] = [p_s,p_e]
                        length_list.append(length)
                    else:
                        length_dict[length].append(p_s)
                        length_dict[length].append(p_e)
                # print(p_s,p_e,length)
                # for k in range(int(np.ceil(length))):
                #     tmp_delta = [k*line[0]/length,k*line[1]/length]
                #     for _,l in enumerate(tmp_delta):
                #         if l >=0:
                #             tmp_delta[_] = np.ceil(l)
                #         else:
                #             tmp_delta[_] = np.floor(l)
                #     occu_tmp[int(np.round(p_s[0]+tmp_delta[0])),int(np.round(p_s[1]+tmp_delta[1]))] = 3
                #     # occu_tmp[int(np.round(p_s[0]+k*line[0]/length)),int(np.round(p_s[1]+k*line[1]/length))] = 2
        for _ in new_lines:
            points_tmp = _.copy()
            points_tmp[:,0] = points_tmp[:,1]
            points_tmp[:,1] = _[:,0].copy()
            for j in range(len(points_tmp)-1):
                p_s = points_tmp[j]
                p_e = points_tmp[j+1]
                line = p_e - p_s
                length = np.linalg.norm(line)
                if length>1:
                    if length not in length_dict:
                        length_dict[length] = [p_s,p_e]
                        length_list.append(length)
                    else:
                        length_dict[length].append(p_s)
                        length_dict[length].append(p_e)
                # print(p_s,p_e,length)
                # for k in range(int(np.ceil(length))):
                #     tmp_delta = [k*line[0]/length,k*line[1]/length]
                #     for _,l in enumerate(tmp_delta):
                #         if l >=0:
                #             tmp_delta[_] = np.ceil(l)
                #         else:
                #             tmp_delta[_] = np.floor(l)
                #     occu_tmp[int(np.round(p_s[0]+tmp_delta[0])),int(np.round(p_s[1]+tmp_delta[1]))] = 3
               
        # print('new_lines: ',new_lines)
        length_list.append(Ny)
        length_dict[Ny] = [np.array([0,0]),np.array([Ny-1,0])]
        length_dict[Ny].append(np.array([0,Nx-1]))
        length_dict[Ny].append(np.array([Ny-1,Nx-1]))
        length_list.append(Nx)
        length_dict[Nx] = [np.array([0,0]),np.array([0,Nx-1])]
        length_dict[Nx].append(np.array([Ny-1,0]))
        length_dict[Nx].append(np.array([Ny-1,Nx-1]))
        # occu_tmp[Ny-1,Nx-1] = 3
        # occu_tmp2 = occu_tmp.copy()
        # print('convexity defect')
        # print(defect_contours)
        # print('edges')
        # plt.imshow(occu_tmp)
        # plt.show()

        ######################################## visualize convixity defect
        # occu_tmp3 = occu_tmp.copy()
        # for i in defect_contours:
        #     print(i)
        #     i = np.array(i).flatten()
            
        #     occu_tmp3[int(i[1]),int(i[0])] = 4
        # fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(7, 4))
        # ax1.imshow(occu_tmp1)
        # ax2.imshow(occu_tmp2)
        # ax3.imshow(occu_tmp3)
        # plt.show()


        ########################################
        flag_found = False
        dila_polygons = []
        for i in ori_polygons:
            dila_polygons.append(i.buffer(2.5))
        tree = STRtree(dila_polygons)
        # print("new obj shape")
        # print([num_grid_l,num_grid_s])
        for length_ori in [num_grid_l,num_grid_s]:
            if length_ori == num_grid_l:
                length_other = num_grid_s
            else:
                length_other = num_grid_l

            length_arr = abs(np.array(length_list)-length_ori)
            for i in range(len(length_list)):
                # print(i)
                # if flag_found:
                #     break
                ind_tmp = np.argmin(length_arr)
                for m in range(int(len(length_dict[length_list[ind_tmp]])/2)):
                    num_check_edge +=1
                    # print("num check edge")
                    # print(num_check_edge,len(length_dict[length_list[ind_tmp]]))
                    p_s_first = length_dict[length_list[ind_tmp]][2*m]
                    p_e_first = length_dict[length_list[ind_tmp]][2*m+1]
                    p_s_first = [p_s_first[1],p_s_first[0]]
                    p_e_first = [p_e_first[1],p_e_first[0]]
                    # print("points")
                    # print(p_s,p_e)
                    line = np.array(p_e_first) - np.array(p_s_first)
                    line = line.astype(np.float32)
                    length = np.linalg.norm(line)
                    offset_l = [0,1,-1,2,-2,3.,-3.,4.,-4.,5.,-5.,6.,-6.,7.,-7.,8.,-8.,9.,-9.,10.,-10.]
                    # offset_l = [5.,-5.,6.,-6.,7.,-7.,8.,-8.,9.,-9.,10.,-10.,11.,-11.,12.,-12.,13.,-13.,14.,-14.,15.,-15.]
                    if length < 1:
                        length_arr[ind_tmp] = 1000
                        break
                    for p in offset_l:
                        # p_s = length_dict[length_list[ind_tmp]][2*m]
                        # p_e = length_dict[length_list[ind_tmp]][2*m+1]
                        # p_s = [p_s[1],p_s[0]]
                        # p_e = [p_e[1],p_e[0]]
                        # print("points")
                        # print(p_s,p_e)
                        # line = np.array(p_e) - np.array(p_s)
                        # length = np.linalg.norm(line)
                        delta_l = length_ori-length
                        # print(length,line,p_s_first,p_e_first)
                        p_s_ori = np.array(p_s_first).copy() + p*line.copy()/length
                        p_e_ori = (np.array(p_e_first).copy() + delta_l*line.copy()/length).copy() + p*line.copy()/length
                        
                        # print(p_s_ori,p_e_ori,line,length_ori,length,delta_l,p_s_first,p_e_first)
                        # occu_tmp[int(round(p_s_ori[1])),int(round(p_s_ori[0]))] = 2
                        # occu_tmp[int(round(p_e_ori[1])),int(round(p_e_ori[0]))] = 2
                        # plt.imshow(occu_tmp)
                        # plt.show()
                        if p == 0:
                            # print("ordinary")
                            range_o = int(np.ceil(abs(delta_l)))
                        else:
                            range_o = 1
                        for o in range(range_o):
                            p_s = p_s_ori - np.sign(delta_l)*o*line.copy()/length
                            p_e = p_e_ori - np.sign(delta_l)*o*line.copy()/length
                            # print("check original points")
                            # print(p_s,p_e,p,o)
                            for gap in range(3,10):
                                for n in range(2):
                                    sign = (-1)**n
                                    tmp_delta = gap*line.copy()/length
                                    # print("s points")
                                    # print(tmp_delta,gap,line,length)
                                    # print(length_other,length_ori)
                                    tmp_delta = np.array([(tmp_delta[1]*sign),(tmp_delta[0]*sign*(-1))])
                                    p_s_new = p_s + tmp_delta
                                    p_e_new = p_e + tmp_delta
                                    if np.max(p_s_new)>Nx-1 or np.max(p_e_new)>Nx-1 or np.min(p_s_new)<1 or np.min(p_e_new)<1:
                                        break
                                    if (occu_ori[int(round(p_s_new[1])),int(round(p_s_new[0]))] >0 or 
                                        occu_ori[int(round(p_e_new[1])),int(round(p_e_new[0]))]>0):
                                        break
                                    p_s_next = tmp_delta*length_other/gap + p_s_new
                                    
                                    p_e_next = tmp_delta*length_other/gap + p_e_new
                                    
                                    if np.max(p_s_next)>Nx-1 or np.max(p_e_next)>Nx-1 or np.min(p_s_next)<1 or np.min(p_e_next)<1:
                                        break
                                    if (occu_ori[int(round(p_s_next[1])),int(round(p_s_next[0]))] >0 or 
                                        occu_ori[int(round(p_e_next[1])),int(round(p_e_next[0]))]>0):
                                        break
                                    new_poly_vetices = [p_s_new,p_e_new,p_e_next,p_s_next]
                                    # print(new_poly_vetices)
                                    new_poly_vetices = np.array(new_poly_vetices,dtype=np.int16).reshape((-1,2))
                                    # print(new_poly_vetices)
                                    # print(max)
                                    if (np.max(new_poly_vetices[:,0])< Nx-1 and np.max(new_poly_vetices[:,1])< Ny-1
                                        and np.min(new_poly_vetices[:,0])>=1 and np.min(new_poly_vetices[:,1])>=1):
                                        new_poly_vetices = [p_s_new,p_e_new,p_e_next,p_s_next]
                                        new_poly_vetices = np.array(new_poly_vetices).reshape((-1,2))
                                        points_tmp = new_poly_vetices.copy()
                                        points_tmp[:,1] = points_tmp[:,0].copy()
                                        points_tmp[:,0] = new_poly_vetices[:,1].copy()
                                        poly = Polygon([p_s_new,p_e_new,p_e_next,p_s_next])
                                        if not poly.is_valid:
                                            continue
                                        indices = tree.nearest(poly)
                                        nearest_poly = tree.geometries.take(indices)
                                        # indices_2 = tree.query(poly)
                                        # print(poly,nearest_poly)
                                        if nearest_poly.is_valid and poly.disjoint(nearest_poly) and area(intersection(poly,nearest_poly))==0:
                                            # print("find the position")
                                            # print(poly)
                                            # for j in range(len(new_poly_vetices)):
                                            #     occu_tmp[int(new_poly_vetices[j][1]),int(new_poly_vetices[j][0])] = 3
                                            # for j in range(len(points_tmp)):
                                            #     p_s_1 = points_tmp[j]
                                            #     if j < len(points_tmp)-1:
                                            #         p_e_1 = points_tmp[j+1]
                                            #     else:
                                            #         p_e_1 = points_tmp[0]
                                                # line_1 = p_e_1 - p_s_1
                                                # length_1 = np.linalg.norm(line_1)
                                                # for k in range(int(np.ceil(length_1))):
                                                #     tmp_delta_1 = [k*line_1[0]/length_1,k*line_1[1]/length_1]
                                                #     for _,l in enumerate(tmp_delta_1):
                                                #         if l >=0:
                                                #             tmp_delta_1[_] = np.ceil(l)
                                                #         else:
                                                #             tmp_delta_1[_] = np.floor(l)
                                                #     if np.round(p_s_1[0]+tmp_delta_1[0])>=Ny:
                                                #         tmp_delta_1[0]=Ny-1-p_s_1[0]
                                                #     if np.round(p_s_1[1]+tmp_delta_1[1])>=Nx:
                                                #         tmp_delta_1[1] = Nx-1 - p_s_1[1]
                                                #     occu_tmp[int(np.round(p_s_1[0]+tmp_delta_1[0])),int(np.round(p_s_1[1]+tmp_delta_1[1]))] = 3
                                            flag_found = True
                                            # print(new_obj,new_poly_vetices)
                                            new_obj_pos = get_pos(new_obj,new_poly_vetices)
                                            # print(points_tmp)
                                            # print(new_obj_pos)
                                            # plt.imshow(occu_tmp)
                                            # plt.show()
                                            # fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(7, 4))
                                            # ax1.imshow(occu_ori)
                                            # ax2.imshow(occu_tmp2)
                                            # ax3.imshow(occu_tmp)
                                            # plt.show()
                                            
                                            return flag_found,new_poly_vetices,occu_tmp,new_obj_pos
                                            break
                                    if flag_found:
                                        break
                                if flag_found:
                                    break
                            if flag_found:
                                break
                        if flag_found:
                            break
                    if flag_found:
                        break
                length_arr[ind_tmp] = 1000
        return False,None, None,None
    else:
        occu_tmp = np.array(occu)
        l = int(np.ceil(bbox[0]))
        w = int(np.ceil(bbox[1]))
        new_poly_vetices = [[0,Ny-w-1],[l,Ny-1-w],[l,Ny-1],[0,Ny-1]]
        new_poly_vetices = np.array(new_poly_vetices).reshape((-1,2))
        for j in range(len(new_poly_vetices)):
            occu_tmp[new_poly_vetices[j][1],new_poly_vetices[j][0]] = 3
        new_obj_pos = get_pos(new_obj,new_poly_vetices)
        return True,new_poly_vetices,occu_tmp,new_obj_pos
def get_pos(new_obj,new_poly_vetices):
    l1 = []
    l2 = []
    pos = [0,0,0]
    obj_l1 = np.array(new_obj[1]) - np.array(new_obj[0])
    obj_l2 = np.array(new_obj[3]) - np.array(new_obj[0])
    sign_obj_p0 = np.sign(np.cross(obj_l1,obj_l2))
    for i in range(2):
        l1.append(np.linalg.norm(new_obj[i]-new_obj[i+1]))
        l2.append(np.linalg.norm(new_poly_vetices[i]-new_poly_vetices[i+1]))
    if l2[0] <1:
        l2[0] = 1
    if l2[1] < 1:
        l2[1] = 1
    if l1[1] < 1:
        l1[1] = 1
    if l1[0] < 1:
        l1[0] = 1
    # print(l1,l2)
    if l1[0] >=l1[1]:
        if l2[0]>=l2[1]:
            
            l_tmp = new_poly_vetices[1]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[3]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[0][0])*l_tmp/l2[0] + abs(new_obj[0][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[2][0])*l_tmp/l2[0] + abs(new_obj[2][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
        else:
            l_tmp = new_poly_vetices[3]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[1]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[1][0])*l_tmp/l2[1] + abs(new_obj[1][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[3][0])*l_tmp/l2[1] + abs(new_obj[3][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
    else:
        if l2[0]<l2[1]:
            l_tmp = new_poly_vetices[1]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[3]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[0][0])*l_tmp/l2[0] + abs(new_obj[0][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[2][0])*l_tmp/l2[0] + abs(new_obj[2][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
        else:
            l_tmp = new_poly_vetices[3]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[1]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[1][0])*l_tmp/l2[1] + abs(new_obj[1][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[3][0])*l_tmp/l2[1] + abs(new_obj[3][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
        #     l_tmp = new_poly_vetices[1]-new_poly_vetices[0]
        #     l_tmp_2 = new_poly_vetices[2]-new_poly_vetices[1]
        #     sign_pos_p0 = np.sign(np.cross(l_tmp_2,l_tmp))
        #     if sign_obj_p0 == sign_pos_p0:
        #         angle = np.arctan2(l_tmp[1],l_tmp[0])
        #         pos_tmp = new_poly_vetices[0] + abs(new_obj[0][0])*l_tmp/l2[0] + abs(new_obj[0][1])*l_tmp_2/l2[1]
        #         pos[2] = angle
        #         pos[0] = pos_tmp[1]
        #         pos[1] = pos_tmp[0]
        #     else:
        #         angle = np.arctan2(-l_tmp[1],-l_tmp[0])
        #         pos_tmp = new_poly_vetices[0] + abs(new_obj[2][0])*l_tmp/l2[0] + abs(new_obj[2][1])*l_tmp_2/l2[1]
        #         pos[2] = angle
        #         pos[0] = pos_tmp[1]
        #         pos[1] = pos_tmp[0]
        # else:
        #     l_tmp = new_poly_vetices[3]-new_poly_vetices[0]
        #     l_tmp_2 = new_poly_vetices[1]-new_poly_vetices[0]
        #     sign_pos_p0 = np.sign(np.cross(l_tmp_2,l_tmp))
            
        #     if sign_obj_p0 == sign_pos_p0:
        #         angle = np.arctan2(-l_tmp[1],-l_tmp[0])
        #         pos_tmp = new_poly_vetices[0] + abs(new_obj[1][0])*l_tmp/l2[1] + abs(new_obj[1][1])*l_tmp_2/l2[0]
        #         pos[2] = angle
        #         pos[0] = pos_tmp[1]
        #         pos[1] = pos_tmp[0]
        #     else:
        #         angle = np.arctan2(l_tmp_2[1],l_tmp_2[0])
        #         pos_tmp = new_poly_vetices[0] + abs(new_obj[3][0])*l_tmp/l2[1] + abs(new_obj[3][1])*l_tmp_2/l2[0]
        #         pos[2] = angle
        #         pos[0] = pos_tmp[1]
        #         pos[1] = pos_tmp[0]

            # l_tmp = new_poly_vetices[2]-new_poly_vetices[1]
            # l_tmp_2 = new_poly_vetices[0]-new_poly_vetices[1]
            # angle = np.arctan2(l_tmp[1],l_tmp[0])
            # pos_tmp = new_poly_vetices[1] + abs(new_obj[0][0])*l_tmp/l2[1] + abs(new_obj[0][1])*l_tmp_2/l2[0]
            # pos[2] = angle
            # pos[0] = pos_tmp[1]
            # pos[1] = pos_tmp[0]
    return pos


# def get_new_obj_info(obj_type):

#     occupancy = np.zeros((40,40))
#     file_list = os.listdir("obj_mask/")
#     for i in range(len(file_list)):
#         if obj_type in file_list[i]:
#             fileObject2 = open('obj_mask/'+file_list[i], 'rb')
#             occupancy=  pickle.load(fileObject2)

#             fileObject2.close()
#     # plt.imshow(occupancy)
#     # plt.show()
#     vertices_new_obj = get_new_obj_contour_bbox(occupancy)
#     return vertices_new_obj