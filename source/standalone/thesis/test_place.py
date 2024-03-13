import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
# fileObject2 = open('ex_occu.pkl', 'rb')
# file_info=  pickle.load(fileObject2)
# fileObject2.close()
# file_info[np.where(file_info>0)]=2
# file_info_tmp = file_info.copy()
# file_info_tmp[6:56,8:58]=1
# file_info_tmp[np.where(file_info>0)]=2
# plt.imshow(file_info_tmp)
# plt.show()
place_p_time = 0
final_num_items_p = 0
occu_ratio_p = 0
for i in range(10):
        file_list = os.listdir("placing_test/")
        file_name = "dict_"+str(i+1)+".pkl"
        if file_name in file_list:
            fileObject2 = open('placing_test/'+file_name, 'rb')
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            place_p_time+=file_info[2]
            final_num_items_p+=file_info[3]
            occu_ratio_p+=file_info[4]
print(place_p_time,final_num_items_p,occu_ratio_p)
place_p_time = place_p_time/final_num_items_p
final_num_items_p = final_num_items_p/10
print(place_p_time,final_num_items_p,occu_ratio_p/10)
place_p_time = 0
final_num_items_p = 0
occu_ratio_p = 0
for i in range(10):
        file_list = os.listdir("placing_test/")
        file_name = "dict_com_"+str(i+1)+".pkl"
        if file_name in file_list:
            fileObject2 = open('placing_test/'+file_name, 'rb')
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            place_p_time+=file_info[2]
            final_num_items_p+=file_info[3]
            occu_ratio_p+=file_info[4]
print(place_p_time,final_num_items_p,occu_ratio_p)
place_p_time = place_p_time/final_num_items_p
final_num_items_p = final_num_items_p/10
print(place_p_time,final_num_items_p,occu_ratio_p/10)