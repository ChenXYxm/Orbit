import numpy as np
from scipy.ndimage import zoom
import os
import pickle as pkl
import matplotlib.pyplot as plt
# Original array
# mask = dict()
# file_list = os.listdir("obj_mask_2_80_80/")
# obj_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle','mug','largeMarker','tunaFishCan',
#         'banana','bowl','largeClamp','scissors']
# for i in range(len(file_list)):
#     for j in range(len(obj_name)):
#         if obj_name[j] in file_list[i]:
#             # print(obj_name[j],file_list[i])
#             fileObject2 = open('obj_mask_2_80_80/'+file_list[i], 'rb')
#             mask[obj_name[j]]=  pkl.load(fileObject2)
#             fileObject2.close()

# # Upsample the array by a factor of 2 using bilinear interpolation
# for mask_i in mask:
#     original_array = mask[mask_i].copy()
#     upsampled_array = zoom(original_array, 2, order=1)
#     ind = np.where(upsampled_array>=0.5)
#     upsampled_array[ind[0],ind[1]] = 1
#     ind = np.where(upsampled_array<0.5)
#     upsampled_array[ind[0],ind[1]] = 0
#     fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 10))
#     ax1.imshow(original_array)
#     ax2.imshow(upsampled_array)
#     plt.show()
#     file_list = os.listdir("obj_mask_2_160_160/")
#     file_name = mask_i +"_mask.pkl"
#     if file_name not in file_list:
#         file_path = "obj_mask_2_160_160/"+file_name
#         f_save = open(file_path,'wb')
#         pkl.dump(upsampled_array,f_save)
#         f_save.close()
mask = dict()
file_list = os.listdir("obj_mask_2_160_160/")
obj_name = ['crackerBox','sugarBox','tomatoSoupCan','mustardBottle','mug','largeMarker','tunaFishCan',
        'banana','bowl','largeClamp','scissors']
for i in range(len(file_list)):
    for j in range(len(obj_name)):
        if obj_name[j] in file_list[i]:
            # print(obj_name[j],file_list[i])
            fileObject2 = open('obj_mask_2_160_160/'+file_list[i], 'rb')
            mask[obj_name[j]]=  pkl.load(fileObject2)
            fileObject2.close()
            plt.imshow(mask[obj_name[j]])
            plt.show()