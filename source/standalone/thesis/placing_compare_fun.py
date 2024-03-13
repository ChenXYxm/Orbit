import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely import Polygon, STRtree, area, contains, buffer, intersection,get_coordinates, concave_hull
from scipy.ndimage import rotate
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
mask = np.zeros((40,40))
w_m = np.random.randint(2,10)
l_m = np.random.randint(2,10)
mask[20-w_m:20+w_m,20-l_m:20+l_m] = 1
plt.imshow(mask)
plt.show()
occu = np.zeros((50,50))
occu_w,occu_l = occu.shape
print("shape: ", occu_l,occu_w)
target_pos = [0,0,0]
flag_found = False
for i in range(8):
    occu_tmp = occu.copy()
    mask_tmp = mask.copy()
    theta = 22.5*i
    mask_tmp = rotate_45(mask,degree=theta)
    plt.imshow(mask_tmp)
    plt.show()
    ind_xy = np.where(mask_tmp>0)
    s_x = int(np.min(np.array(ind_xy[0]).astype(int)))-2
    e_x = int(np.max(np.array(ind_xy[0]).astype(int))+3)
    s_y = int(np.min(np.array(ind_xy[1]).astype(int)))-2
    e_y = int(np.max(np.array(ind_xy[1]).astype(int))+3)
    mask_tmp = mask_tmp[s_x:e_x,s_y:e_y]
    l_m_tmp = -s_y+e_y
    w_m_tmp = -s_x+e_x
    # print(l_m_tmp,w_m_tmp)
    plt.imshow(mask_tmp)
    plt.show()
    for j in range(occu_w-w_m_tmp):
        for k in range(occu_l-l_m_tmp):
            result = np.max(mask_tmp+occu_tmp[j:j+w_m_tmp,k:k+l_m_tmp])
            if result ==1:
                occu_tmp[j:j+w_m_tmp,k:k+l_m_tmp] = mask_tmp*2
                flag_found = True
                # print(occu_tmp)
                plt.imshow(occu_tmp)
                plt.show()
                
                target_pos = [int(j+w_m_tmp/2),int(k+l_m_tmp/2),np.deg2rad(theta)]
                # print(flag_found,target_pos) 
                break
        if flag_found:
            break
    if flag_found:
        break
    # print(flag_found,target_pos)
