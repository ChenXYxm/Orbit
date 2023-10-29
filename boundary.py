import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im 
import cv2
from scipy import ndimage
from shapely import Polygon, STRtree, area, contains, buffer, Point
##############################################
##### Attention: the index return by cv2 must be switched
##############################################
# def slice_occu(occu=np.array,tol=1):
    
#     occu_tmp = occu.copy()
#     occu_ori = occu.copy()
#     (lx,ly) = occu.shape
#     index_tmp_e = dict()
#     index_tmp_s = dict()
#     shape_dict = dict()
#     shape_num = 0
#     for i in range(lx):
#         while len(np.where(occu_tmp[i]==1)[0]) >0:
#             flag = True
#             temp_s = np.where(occu_tmp[i]==1)[0][0]
#             temp_e = np.where(occu_tmp[i]==-1)[0][0]
#             shape_num +=1
#             shape_dict[shape_num] = [[i,temp_s]]
#             shape_dict[shape_num].append([i,temp_e])
#             occu_tmp[i,temp_s] = 0
#             occu_tmp[i,temp_e] = 0
#             i_d = i
#             while flag:
#                 i_d +=11
#                 if len(np.where(occu_tmp[i_d]==1)[0]) >0:
#                     # find closest continuous point
#                     diff_tmp = np.abs(np.where(occu_tmp[i_d]==1)[0] - temp_s)
#                     if len(np.where(diff_tmp<=tol)[0]) == 0:
#                         flag = False
#                     else:
#                         temp_s = np.where(occu_tmp[i_d]==1)[0][np.where(diff_tmp<=tol)[0][0]]
#                         temp_e = np.where(occu_tmp[i_d]==-1)[0][np.where(diff_tmp<=tol)[0][0]]
#                         shape_dict[shape_num].append([i_d,temp_s])
#                         shape_dict[shape_num].append([i_d,temp_e])
#                         occu_tmp[i_d,temp_s] = 0
#                         occu_tmp[i_d,temp_e] = 0
#                 else:
#                     flag = False
#     print(shape_dict)
#     for i in shape_dict:
#         points_tmp = np.array(shape_dict[i],dtype=int)
#         bbox_min = np.min(points_tmp,axis=0)
#         bbox_max = np.max(points_tmp,axis=0)
#         occu_tmp[bbox_min[0]:bbox_max[0]+1,bbox_min[1]:bbox_max[1]+1] = -1
#     plt.imshow(occu_tmp)
#     plt.show()
    
        # print(points_tmp.shape)
        # print(points_tmp)
            # index_tmp_s[i] = np.where(occu_tmp[i]==1)[0]
        # if len(np.where(occu_tmp[i]==-1)[0]) >0:
        #     index_tmp_e[i] = np.where(occu_tmp[i]==-1)[0]
    # i_s_y = -1
    # i_s_x = -1
    # i_e_y = -1
    # i_e_x = -1
    # num_cu = 0
    # for i in range(lx):
    #     if i_s_y == -1:
    #         i_s_y = i
    #         if len(index_tmp_s[i].reshape(-1))>1:
    #             i_s_x = index_tmp_s[i][0]


    


# occu = np.load('occu.npy')
# # print(occu)
# occu_p = occu.copy()/np.max(occu)
# occu_p = np.array(occu*255,dtype=np.uint8)
# data = im.fromarray(occu_p) 

# data.save('occu_1.png') 
# (lx,ly) = occu.shape
# occu_ori = occu.copy()
# occu_t = np.zeros((lx+2,ly+2))
# occu_t[1:-1,1:-1] = occu
# occu = occu_t*-1
# plt.imshow(occu)
# plt.show()
# # plt.savefig('occu.png')
# occu_dx = np.zeros(occu.shape)
# occu_dy = np.zeros(occu.shape)
# occu_dx[1:] = np.diff(occu,axis=0)
# occu_dy[:,1:] = np.diff(occu,axis=1)
# slice_occu(occu_dy)
# plt.figure()
# plt.imshow(occu_dx)
# plt.figure()
# plt.imshow(occu_dy)
# plt.figure()
# plt.imshow(occu_dy+occu_dx)
# plt.show()
bbox = [0.1,0.3]
length = np.max(np.array(bbox))
width = np.min(np.array(bbox))
num_grid_l = int(np.ceil(np.max(np.array(bbox))/0.01))
num_grid_s = int(np.ceil(np.min(np.array(bbox))/0.01))
mask = np.zeros((int(2*num_grid_l),int(2*num_grid_l)))
mask[int(num_grid_l-np.ceil(num_grid_s/2)):int(num_grid_l+np.ceil(num_grid_s/2)),int(num_grid_l-np.ceil(num_grid_l/2)):int(num_grid_l+np.ceil(num_grid_l/2))] = 1
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
approx = cv2.minAreaRect(cnt)
(x,y) = cnt[0,0]
# print(approx)
box = cv2.boxPoints(approx)
approx = np.int0(box)
vertices_new_obj = []
mask_tmp = mask.copy()
if len(approx) >=2:
    approx = approx.reshape((-1,2))
    for i in range(len(approx)):
        mask_tmp[approx[i][1],approx[i][0]] = 130
    vertices_new_obj = approx
plt.imshow(mask_tmp)
plt.show()
# vertices = [[int(num_grid_l-np.ceil(num_grid_s/2)),int(num_grid_l-np.ceil(num_grid_l/2))],
#             [int(num_grid_l-np.ceil(num_grid_s/2)),int(num_grid_l+np.ceil(num_grid_l/2))-1],
#             [int(num_grid_l+np.ceil(num_grid_s/2))-1,int(num_grid_l+np.ceil(num_grid_l/2))-1],
#             [int(num_grid_l+np.ceil(num_grid_s/2))-1,int(num_grid_l-np.ceil(num_grid_l/2))]]
# for i in vertices:
#     mask[i[0],i[1]] = 2
# plt.imshow(mask)
# plt.show()
# mask_45 = np.array(ndimage.rotate(mask,45,reshape=False))
# mask_45[np.where(mask_45>=0.4)] = 1
# mask_45[np.where(mask_45<1)] = 0
# print(np.max(mask_45),np.min(mask_45),mask_45.shape)
# mask_45 = np.array((mask_45-np.min(mask_45))*255/(np.max(mask_45)-np.min(mask_45)),dtype=np.uint8)
# print(np.max(mask_45),np.min(mask_45),mask_45.shape)
# # mask_45 = im.fromarray(mask_45)
# # gray = cv2.cvtColor(mask_45,cv2.COLOR_BGR2GRAY)
# plt.imshow(mask_45)
# plt.show()
# ret,mask_45 = cv2.threshold(mask_45,50,255,0)
# contours,hierarchy = cv2.findContours(mask_45,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print("Number of contours detected:",len(contours))
# max_area = 0
# cnt = []
# for i in contours:
#     area_tmp = cv2.contourArea(i)
#     if area_tmp>max_area:
#         max_area = area_tmp
#         cnt = i
# # cnt = contours[0]
# # for cnt in contours:
# approx = cv2.minAreaRect(cnt)
# (x,y) = cnt[0,0]
# # print(approx)
# box = cv2.boxPoints(approx)
# approx = np.int0(box)
# vertices_new_obj = []
# if len(approx) >=2:
#     approx = approx.reshape((-1,2))
#     for i in range(len(approx)):
#         mask_45[approx[i][1],approx[i][0]] = 50
#     vertices_new_obj = approx
# plt.imshow(mask_45)
# plt.show()
# l1 = [vertices_new_obj[0],vertices_new_obj[1]]
# length_l1 = np.linalg.norm(l1[1]-l1[0])
# if length_l1 == 7:
#     length_l2 = 20+1
# else:
#     length_l2 = 7+1
# l2 = [vertices_new_obj[1],0]
# edge_1 = (l1[1]-l1[0])/length_l1
# edge_2_tmp = np.array([-edge_1[1],edge_1[0]])
# l2[1] = l2[0] + length_l2*edge_2_tmp
# l2[1] = np.round(l2[1])
# l3 = [l2[1],np.round(l2[1]- edge_1*length_l1)] 
# mask_45[int(l2[1][0]),int(l2[1][1])] = 50
# mask_45[int(l3[1][0]),int(l3[1][1])] = 50
# # print(np.max(mask_45))
# # print(mask_45.shape)
# # print(np.where(mask_45>1))

# # print(len(np.where(mask>=1)[0]))
# # print(len(np.where(mask_45>=1)[0]))


# plt.imshow(mask_45)
# plt.show()

img = cv2.imread('occu.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,50,255,0)

occu = np.array(thresh)/np.max(np.array(thresh))
plt.imshow(occu)
plt.show()
print(thresh.shape)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours detected:",len(contours))
shape_dict = dict()
i = 0
polygons = []
for cnt in contours:
    i += 1
    # approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
    # approx = cv2.convexHull(cnt)
    approx = cv2.minAreaRect(cnt)
    (x,y) = cnt[0,0]
    # print(approx)
    box = cv2.boxPoints(approx)
    approx = np.int0(box)
    if len(approx) >=2:
        # img = cv2.drawContours(img,[approx],-1,(0,255,255),3)
        print(approx)
        approx = approx.reshape((-1,2))
        print(approx)
        shape_dict[i] = approx
        polygons.append(Polygon(approx))
        
print(polygons)
tree_ori = STRtree(polygons)
print("remove containing boxes")
del_ind = []
# for i,poly in enumerate(polygons):
#     indice = tree.query(poly, predicate="contains").tolist()
#     print(i)
#     print(tree.geometries.take(indice))
#     print(tree.geometries.take(i))
#     if len(indice) >0:
#         for j in indice:
#             if contains(poly,tree.geometries.take(j)) and area(poly)>area(tree.geometries.take(j)):
#                 if shape_dict[j+1] is not None:
#                     print("show remove")
#                     print(j)
#                     del(shape_dict[j+1])
#                     del_ind.append(int(j))
print(shape_dict)
# print(shape_dict)
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
#     print(i,indice)
#     print(tree.geometries.take(indice))
#     print(tree_ori.geometries.take(i))
#     # if len(indice) >0:
#     #     for j in indice:
#     #         if contains(poly,tree.geometries.take(j)) and area(poly)>area(tree.geometries.take(j)):
#     #             if j >= i:
#     #                 if shape_dict[j+1] is not None:
#     #                     print("show remove")
#     #                     print(j+1)
#     #                     del(shape_dict[j+1])
#     #                     del_ind.append(int(j+1))
# print(polygons)
print(del_ind)
print(shape_dict)
polygons = []
for i in shape_dict:
    polygons.append(Polygon(shape_dict[i]))
tree_ori = STRtree(polygons)
# shape_dict = dict()
# for i,poly in enumerate(polygons):
#     shape_dict[i] = poly
print(polygons)
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
    print(i,indice)
    print(tree.geometries.take(indice))
    print(tree_ori.geometries.take(i))
    if len(indice) >0:
        for j in indice:
            if contains(poly_tmp,tree.geometries.take(j)) and area(poly_tmp)>area(tree.geometries.take(j)):
                if j >= i:
                    j_tmp = j+1
                else:
                    j_tmp = j
                if shape_dict[j_tmp+1] is not None:
                    print("show remove")
                    print(j_tmp)
                    del(shape_dict[j_tmp+1])
                    del_ind.append(int(j_tmp))

print(polygons)
polygons = []
occu_tmp = np.array(occu)
for i in shape_dict:
    polygons.append(Polygon(shape_dict[i]))
    for j in range(len(shape_dict[i])):
        occu_tmp[shape_dict[i][j][1],shape_dict[i][j][0]] = 2
plt.imshow(occu_tmp)
plt.show()
length_dict = dict()
length_list = []
occu_tmp = np.array(occu)
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
        print(p_s,p_e,length)
        for k in range(int(np.ceil(length))):
            tmp_delta = [k*line[0]/length,k*line[1]/length]
            for _,l in enumerate(tmp_delta):
                if l >=0:
                    tmp_delta[_] = np.ceil(l)
                else:
                    tmp_delta[_] = np.floor(l)
            occu_tmp[int(np.round(p_s[0]+tmp_delta[0])),int(np.round(p_s[1]+tmp_delta[1]))] = 2
            # occu_tmp[int(np.round(p_s[0]+k*line[0]/length)),int(np.round(p_s[1]+k*line[1]/length))] = 2
plt.imshow(occu_tmp)
plt.show()
flag_found = False
length_arr = abs(np.array(length_list)-num_grid_l)
flag = False
print(length_dict)
tree = STRtree(polygons)
for i in range(len(length_list)):
    print(i)
    if flag_found:
        break
    ind_tmp = np.argmin(length_arr)
    p_s = length_dict[length_list[ind_tmp]][0]
    p_e = length_dict[length_list[ind_tmp]][1]
    p_s = [p_s[1],p_s[0]]
    p_e = [p_e[1],p_e[0]]
    # print("points")
    # print(p_s,p_e)
    line = np.array(p_e) - np.array(p_s)
    length = np.linalg.norm(line)

    # print(line,length)
    # angle = (np.arccos(line[1]/length))/np.pi*180
    for n in range(2):
        sign = (-1)**n
        tmp_delta = np.round(2*line/length)
        tmp_delta = np.array([int(tmp_delta[1]*sign),int(tmp_delta[0]*sign*(-1))])
        p_s_new = np.array(p_s,dtype = int) + tmp_delta
        p_e_new = np.array(p_e,dtype = int) + tmp_delta
        p_s_next = tmp_delta*num_grid_s/2 + p_s_new
        p_s_next = np.array(p_s_next,dtype=int)
        p_e_next = tmp_delta*num_grid_s/2 + p_e_new
        p_e_next = np.array(p_e_next,dtype=int)
        bound_box = Polygon([[0,0],[0,61],[101,0],[101,61]])
        print(bound_box)
        new_poly_vetices = [p_s_new,p_e_new,p_e_next,p_s_next]
        new_poly_vetices = np.array(new_poly_vetices).reshape((-1,2))
        print(max)
        if (np.max(new_poly_vetices[:,0])< 101 and np.max(new_poly_vetices[:,1])< 61 
            and np.min(new_poly_vetices[:,0])>=0 and np.min(new_poly_vetices[:,1])>=0):
            new_poly_vetices = [p_s_new,p_e_new,p_e_next,p_s_next]
            new_poly_vetices = np.array(new_poly_vetices).reshape((-1,2))
            points_tmp = new_poly_vetices.copy()
            points_tmp[:,1] = points_tmp[:,0].copy()
            points_tmp[:,0] = new_poly_vetices[:,1].copy()
            poly = Polygon([p_s_new,p_e_new,p_e_next,p_s_next])
            indices = tree.nearest(poly)
            nearest_poly = tree.geometries.take(indices)
            print(poly,nearest_poly)
            if poly.disjoint(nearest_poly):
                print("find the position")
                print(poly)
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
                        occu_tmp[int(np.round(p_s_1[0]+tmp_delta_1[0])),int(np.round(p_s_1[1]+tmp_delta_1[1]))] = 3
                flag_found = True
                break
        print(p_s_new,p_e_new,p_s_next,p_e_next)
        # occu_tmp[int(np.round(p_s_new[1])),int(np.round(p_s_new[0]))] = 3
        # occu_tmp[int(np.round(p_e_new[1])),int(np.round(p_e_new[0]))] = 3
        # occu_tmp[int(np.round(p_s_next[1])),int(np.round(p_s_next[0]))] = 3
        # occu_tmp[int(np.round(p_e_next[1])),int(np.round(p_e_next[0]))] = 3
plt.imshow(occu_tmp)
plt.show()        

    # p_e
    # print(angle)
    
    # angle = 90
    # mask_45 = np.array(ndimage.rotate(mask,angle,reshape=False))
    
    # mask_45[np.where(mask_45>=0.4)] = 1
    # mask_45[np.where(mask_45<1)] = 0
    # index = np.where(mask_45>=1)
    # index_arr = np.zeros((len(index[0]),2))
    # index_arr[:,0] = index[0]
    # index_arr[:,1] = index[1]
    # index_arr = np.array(index_arr,dtype=int).reshape((-1,2))

    # vertices = [index_arr[np.where(index_arr[:,1]==np.min(index_arr[:,1]))[0]].reshape(-1),
    #             index_arr[np.where(index_arr[:,0]==np.max(index_arr[:,0]))[0]].reshape(-1),
    #             index_arr[np.where(index_arr[:,1]==np.max(index_arr[:,1]))[-1]].reshape(-1),
    #             index_arr[np.where(index_arr[:,0]==np.min(index_arr[:,0]))[0]].reshape(-1)]
    
    # print(vertices)
    # # for i in vertices:
    # #     print(i)
    # #     mask_45[i[0],i[1]] = 2
    # plt.imshow(mask_45)
    # plt.show()
    # occu_ori = occu.copy()
    # print(p_e,p_s)
    # print(mask_45.shape)
    # print(2*num_grid_l)
    # occu_ori[29:60,14:54] = occu_ori[29:60,14:54] + mask_45[:31,]
    # plt.imshow(occu_ori)
    # plt.show()
    # flag = True
    # occu_ori2 = np.zeros(occu_ori.shape)
    # if flag:
    #     break
#     for k in range(int(np.ceil(length))):
#         occu_ori2[int(np.ceil(p_s[0]+k*line[0]/length)),int(np.ceil(p_s[1]+k*line[1]/length))] =1
# plt.imshow(occu_ori2)
# plt.show()



        # cv2.putText(img,'Polygon',(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
# cv2.imshow("Polygon",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

