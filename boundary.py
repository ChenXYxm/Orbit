import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im 
import cv2
from scipy import ndimage
from shapely import Polygon
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
#                 i_d +=1
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
bbox = [0.07,0.2]
num_grid_l = int(np.ceil(np.max(np.array(bbox))/0.01))
num_grid_s = int(np.ceil(np.min(np.array(bbox))/0.01))
mask = np.zeros((int(2*num_grid_l),int(2*num_grid_l)))
mask[int(num_grid_l-np.ceil(num_grid_s/2)):int(num_grid_l+np.ceil(num_grid_s/2)),int(num_grid_l-np.ceil(num_grid_l/2)):int(num_grid_l+np.ceil(num_grid_l/2))] = 1
vertices = [[int(num_grid_l-np.ceil(num_grid_s/2)),int(num_grid_l-np.ceil(num_grid_l/2))],
            [int(num_grid_l-np.ceil(num_grid_s/2)),int(num_grid_l+np.ceil(num_grid_l/2))-1],
            [int(num_grid_l+np.ceil(num_grid_s/2))-1,int(num_grid_l+np.ceil(num_grid_l/2))-1],
            [int(num_grid_l+np.ceil(num_grid_s/2))-1,int(num_grid_l-np.ceil(num_grid_l/2))]]
for i in vertices:
    mask[i[0],i[1]] = 2
plt.imshow(mask)
plt.show()
mask_45 = np.array(ndimage.rotate(mask,45,reshape=False))
mask_45[np.where(mask_45>=0.4)] = 1
mask_45[np.where(mask_45<1)] = 0
print(np.max(mask_45),np.min(mask_45),mask_45.shape)
mask_45 = np.array((mask_45-np.min(mask_45))*255/(np.max(mask_45)-np.min(mask_45)),dtype=np.uint8)
print(np.max(mask_45),np.min(mask_45),mask_45.shape)
# mask_45 = im.fromarray(mask_45)
# gray = cv2.cvtColor(mask_45,cv2.COLOR_BGR2GRAY)
ret,mask_45 = cv2.threshold(mask_45,50,255,0)
contours,hierarchy = cv2.findContours(mask_45,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours detected:",len(contours))
max_area = 0
cnt = []
for i in contours:
    area_tmp = cv2.contourArea(i)
    if area_tmp>max_area:
        max_area = area_tmp
        cnt = i
cnt = contours[0]
# for cnt in contours:
approx = cv2.minAreaRect(cnt)
(x,y) = cnt[0,0]
# print(approx)
box = cv2.boxPoints(approx)
approx = np.int0(box)
vertices_new_obj = []
if len(approx) >=2:
    approx = approx.reshape((-1,2))
    for i in range(len(approx)):
        mask_45[approx[i][0],approx[i][1]] = 50
    vertices_new_obj = approx
plt.imshow(mask_45)
plt.show()
l1 = [vertices_new_obj[0],vertices_new_obj[1]]
length_l1 = np.linalg.norm(l1[1]-l1[0])
if length_l1 == 7:
    length_l2 = 20+1
else:
    length_l2 = 7+1
l2 = [vertices_new_obj[1],0]
edge_1 = (l1[1]-l1[0])/length_l1
edge_2_tmp = np.array([-edge_1[1],edge_1[0]])
l2[1] = l2[0] + length_l2*edge_2_tmp
l2[1] = np.round(l2[1])
l3 = [l2[1],np.round(l2[1]- edge_1*length_l1)] 
mask_45[int(l2[1][0]),int(l2[1][1])] = 50
mask_45[int(l3[1][0]),int(l3[1][1])] = 50
# print(np.max(mask_45))
# print(mask_45.shape)
# print(np.where(mask_45>1))

# print(len(np.where(mask>=1)[0]))
# print(len(np.where(mask_45>=1)[0]))


plt.imshow(mask_45)
plt.show()

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
        length_dict[length] = [p_s,p_e]
        length_list.append(length)
        print(p_s,p_e,length)
        for k in range(int(np.ceil(length))):
            occu_tmp[int(np.ceil(p_s[0]+k*line[0]/length)),int(np.ceil(p_s[1]+k*line[1]/length))] = 2
plt.imshow(occu_tmp)
plt.show()
length_arr = abs(np.array(length_list)-num_grid_l)
flag = False
print(length_dict)
for i in range(len(length_list)):
    ind_tmp = np.argmin(length_arr)
    p_s = length_dict[length_list[ind_tmp]][0]
    p_e = length_dict[length_list[ind_tmp]][1]
    print("points")
    print(p_s,p_e)
    line = np.array(p_e) - np.array(p_s)
    length = np.linalg.norm(line)
    print(line,length)
    angle = (np.arccos(line[1]/length))/np.pi*180
    print(angle)
    # angle = 90
    mask_45 = np.array(ndimage.rotate(mask,angle,reshape=False))
    
    mask_45[np.where(mask_45>=0.4)] = 1
    mask_45[np.where(mask_45<1)] = 0
    index = np.where(mask_45>=1)
    index_arr = np.zeros((len(index[0]),2))
    index_arr[:,0] = index[0]
    index_arr[:,1] = index[1]
    index_arr = np.array(index_arr,dtype=int).reshape((-1,2))

    vertices = [index_arr[np.where(index_arr[:,1]==np.min(index_arr[:,1]))[0]].reshape(-1),
                index_arr[np.where(index_arr[:,0]==np.max(index_arr[:,0]))[0]].reshape(-1),
                index_arr[np.where(index_arr[:,1]==np.max(index_arr[:,1]))[-1]].reshape(-1),
                index_arr[np.where(index_arr[:,0]==np.min(index_arr[:,0]))[0]].reshape(-1)]
    
    print(vertices)
    # for i in vertices:
    #     print(i)
    #     mask_45[i[0],i[1]] = 2
    plt.imshow(mask_45)
    plt.show()
    occu_ori = occu.copy()
    print(p_e,p_s)
    print(mask_45.shape)
    print(2*num_grid_l)
    occu_ori[29:60,14:54] = occu_ori[29:60,14:54] + mask_45[:31,]
    plt.imshow(occu_ori)
    plt.show()
    flag = True
    occu_ori2 = np.zeros(occu_ori.shape)
    if flag:
        break
#     for k in range(int(np.ceil(length))):
#         occu_ori2[int(np.ceil(p_s[0]+k*line[0]/length)),int(np.ceil(p_s[1]+k*line[1]/length))] =1
# plt.imshow(occu_ori2)
# plt.show()



        # cv2.putText(img,'Polygon',(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
# cv2.imshow("Polygon",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

