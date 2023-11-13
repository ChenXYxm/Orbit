import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely import Polygon, STRtree, area, contains, buffer, intersection
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
def place_new_obj_fun(occu_ori,new_obj):
    ######################## input format
    # occu_ori: numpy 2d array: binary 0,1
    # new_obj: numpy 2d array: vertices of bbox in relative to the center point
    ########################
    shape_occu = occu_ori.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    # print(shape_occu)
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
    polygons = []
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
        # plt.imshow(occu_tmp)
        # plt.show()
        length_dict = dict()
        length_list = []
        # occu_tmp = np.array(occu).copy()
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
                if length not in length_dict:
                    length_dict[length] = [p_s,p_e]
                    length_list.append(length)
                else:
                    length_dict[length].append(p_s)
                    length_dict[length].append(p_e)
                # print(p_s,p_e,length)
                for k in range(int(np.ceil(length))):
                    tmp_delta = [k*line[0]/length,k*line[1]/length]
                    for _,l in enumerate(tmp_delta):
                        if l >=0:
                            tmp_delta[_] = np.ceil(l)
                        else:
                            tmp_delta[_] = np.floor(l)
                    occu_tmp[int(np.round(p_s[0]+tmp_delta[0])),int(np.round(p_s[1]+tmp_delta[1]))] = 3
                    # occu_tmp[int(np.round(p_s[0]+k*line[0]/length)),int(np.round(p_s[1]+k*line[1]/length))] = 2
        length_list.append(Ny)
        length_dict[Ny] = [np.array([0,0]),np.array([Ny-1,0])]
        length_dict[Ny].append(np.array([0,Nx-1]))
        length_dict[Ny].append(np.array([Ny-1,Nx-1]))
        length_list.append(Nx)
        length_dict[Nx] = [np.array([0,0]),np.array([0,Nx-1])]
        length_dict[Nx].append(np.array([Ny-1,0]))
        length_dict[Nx].append(np.array([Ny-1,Nx-1]))
        occu_tmp[Ny-1,Nx-1] = 3
        # plt.imshow(occu_tmp)
        # plt.show()
        flag_found = False
        dila_polygons = []
        for i in polygons:
            dila_polygons.append(i.buffer(1.5))
        tree = STRtree(dila_polygons)
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
                    p_s = length_dict[length_list[ind_tmp]][2*m]
                    p_e = length_dict[length_list[ind_tmp]][2*m+1]
                    p_s = [p_s[1],p_s[0]]
                    p_e = [p_e[1],p_e[0]]
                    # print("points")
                    # print(p_s,p_e)
                    line = np.array(p_e) - np.array(p_s)
                    length = np.linalg.norm(line)
                    offset_l = [0,1,2,3,4,-1,-2,-3,-4]
                    for p in offset_l:
                        p_s = length_dict[length_list[ind_tmp]][2*m]
                        p_e = length_dict[length_list[ind_tmp]][2*m+1]
                        p_s = [p_s[1],p_s[0]]
                        p_e = [p_e[1],p_e[0]]
                        # print("points")
                        # print(p_s,p_e)
                        line = np.array(p_e) - np.array(p_s)
                        length = np.linalg.norm(line)
                        if length < 1:
                            if m == int(len(length_dict[length_list[ind_tmp]])/2):
                                length_arr[ind_tmp] = 1000
                            continue
                        
                        delta_l = length_ori-length
                        p_s_ori = np.array(p_s).copy() + p*line/length
                        p_e_ori = (np.array(p_e).copy() + delta_l*line/length).copy() + p*line/length
                        if p == 0:
                            # print("ordinary")
                            range_o = int(np.ceil(abs(delta_l)))
                        else:
                            range_o = 1
                        for o in range(range_o):
                            p_s = p_s_ori - np.sign(delta_l)*o*line/length
                            p_e = p_e_ori - np.sign(delta_l)*o*line/length
                            # print("check original points")
                            # print(p_s,p_e)
                            for gap in range(2,6):
                                for n in range(2):
                                    sign = (-1)**n
                                    tmp_delta = np.round(gap*line/length)
                                    tmp_delta = np.array([(tmp_delta[1]*sign),(tmp_delta[0]*sign*(-1))])
                                    p_s_new = p_s + tmp_delta
                                    p_e_new = p_e + tmp_delta
                                    p_s_next = tmp_delta*length_other/gap + p_s_new
                                    # p_s_next = np.array(p_s_next,dtype=int)
                                    p_e_next = tmp_delta*length_other/gap + p_e_new
                                    # p_e_next = np.array(p_e_next,dtype=int)
                                    # bound_box = Polygon([[0,0],[0,60],[100,0],[100,60]])
                                    # print(bound_box)
                                    new_poly_vetices = [p_s_new,p_e_new,p_e_next,p_s_next]
                                    new_poly_vetices = np.array(new_poly_vetices,dtype=np.uint8).reshape((-1,2))
                                    # print(max)
                                    if (np.max(new_poly_vetices[:,0])< Nx-1 and np.max(new_poly_vetices[:,1])< Ny-1
                                        and np.min(new_poly_vetices[:,0])>=1 and np.min(new_poly_vetices[:,1])>=1):
                                        new_poly_vetices = [p_s_new,p_e_new,p_e_next,p_s_next]
                                        new_poly_vetices = np.array(new_poly_vetices).reshape((-1,2))
                                        points_tmp = new_poly_vetices.copy()
                                        points_tmp[:,1] = points_tmp[:,0].copy()
                                        points_tmp[:,0] = new_poly_vetices[:,1].copy()
                                        poly = Polygon([p_s_new,p_e_new,p_e_next,p_s_next])
                                        indices = tree.nearest(poly)
                                        nearest_poly = tree.geometries.take(indices)
                                        indices_2 = tree.query(poly)
                                        # print(poly,nearest_poly)
                                        if poly.disjoint(nearest_poly) and area(intersection(poly,nearest_poly))==0 and len(indices_2)==0:
                                            # print("find the position")
                                            # print(poly)
                                            # for j in range(len(new_poly_vetices)):
                                            #     occu_tmp[int(new_poly_vetices[j][1]),int(new_poly_vetices[j][0])] = 3
                                            for j in range(len(points_tmp)):
                                                p_s_1 = points_tmp[j]
                                                if j < len(points_tmp)-1:
                                                    p_e_1 = points_tmp[j+1]
                                                else:
                                                    p_e_1 = points_tmp[0]
                                                line_1 = p_e_1 - p_s_1
                                                length_1 = np.linalg.norm(line_1)
                                                for k in range(int(np.ceil(length_1))):
                                                    tmp_delta_1 = [k*line_1[0]/length_1,k*line_1[1]/length_1]
                                                    for _,l in enumerate(tmp_delta_1):
                                                        if l >=0:
                                                            tmp_delta_1[_] = np.ceil(l)
                                                        else:
                                                            tmp_delta_1[_] = np.floor(l)
                                                    if np.round(p_s_1[0]+tmp_delta_1[0])>=Ny:
                                                        tmp_delta_1[0]=Ny-1-p_s_1[0]
                                                    if np.round(p_s_1[1]+tmp_delta_1[1])>=Nx:
                                                        tmp_delta_1[1] = Nx-1 - p_s_1[1]
                                                    occu_tmp[int(np.round(p_s_1[0]+tmp_delta_1[0])),int(np.round(p_s_1[1]+tmp_delta_1[1]))] = 3
                                            flag_found = True
                                            new_obj_pos = get_pos(new_obj,new_poly_vetices)
                                            # print(points_tmp)
                                            # print(new_obj_pos)
                                            # plt.imshow(occu_tmp)
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