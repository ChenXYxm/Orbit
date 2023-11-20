import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import torch
from shapely import centroid, Point, Polygon
a1 = np.array([17,9])
a2 = np.array([17,0])
a3 = np.array([0,0])
a4 = np.array([0,9])
l1 = a2 - a1
l2 = a4 - a1
print(np.cross(l2,l1),np.sign(np.cross(l1,l2)))
print(np.arctan2(-.5,.5),np.arctan2(.5,-.5),np.arctan2(-.5,-.5),np.arctan2(.5,.5))
mask = dict()
file_list = os.listdir("obj_mask/")
file_list = os.listdir("generated_table/")
# obj_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle','mug','largeMarker','tunaFishCan',
#         'banana','bowl','largeClamp','scissors']
# for i in range(len(file_list)):
    # for j in range(len(obj_name)):
#         if obj_name[j] in file_list[i]:
#             print(obj_name[j],file_list[i])
#             fileObject2 = open('obj_mask/'+file_list[i], 'rb')
#             mask[obj_name[j]]=  pkl.load(fileObject2)

#             fileObject2.close()
#             plt.imshow(mask[obj_name[j]])
#             plt.show()
max_num = 0
max_num_name = 0
for i in range(len(file_list)):
    fileObject2 = open('generated_table/'+file_list[i], 'rb')
    list_obj =  pkl.load(fileObject2)
    for j in list_obj[0]:
        num_ob = len(list_obj[0][j])
        if num_ob > max_num:
            max_num = num_ob
            max_num_name = j
print(max_num,max_num_name)
arr = np.eye(3)
print(torch.from_numpy(arr))

polygon_p = np.array([[0,0],[1,0],[1,1],[0,1]])
poly = Polygon(polygon_p)
print(centroid(poly))
print(np.asarray(poly.centroid.coords))
a = np.random.randint(0,9,(2,2,3))
print(a)
print('max')
# print(np.argmax(a,axis=0))
x = np.arange(6).reshape((2, 3))
x[0,0] = 5
res = np.max(x)
print(x)
print(res)
print(np.where(x==res))
x = np.arange(6).reshape((2, 3))
x[0,0] = 5
x = torch.from_numpy(x).to('cuda:0')
res = torch.max(x)
print(x)
print(res)
print(torch.where(x==res))
# print(np.array(np.where(x==res)).reshape(-1,2))
yy = np.eye(3)
print(np.where(yy==1)[1])

yy= np.zeros((4,3))
yy[:3,:] = np.eye(3)
yy = torch.from_numpy(yy)
xx = yy.clone()
xx[1,0] = 1
print(torch.sum(torch.abs(yy-xx)>=0.1,dim=1))
print(torch.where(torch.sum(torch.abs(yy-xx),dim=1)>=0.1,1,0))

xx = {}
xx[1] = 12
yy = xx.copy()
yy[1] = 10
print(xx)
print(yy)
