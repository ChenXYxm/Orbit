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
def range_std(l):
     l_np = np.array(l)
     range_min = np.mean(l_np)-0.18*(np.max(np.array(l))-np.min(np.array(l)))-np.min(np.array(l))
     range_max = 0.84*(np.max(np.array(l))-np.min(np.array(l)))+np.min(np.array(l))-np.mean(l_np)
     return (range_min,range_max)
place_p_time = 0
final_num_items_p = 0
occu_ratio_p = 0
place_p_time_l = []
final_num_items_p_l = []
for i in range(21):
        file_list = os.listdir("placing_test2/")
        file_name = "dict_"+str(i+1)+".pkl"
        if file_name in file_list:
            fileObject2 = open('placing_test2/'+file_name, 'rb')
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            place_p_time+=file_info[2]
            place_p_time_l.append(file_info[2]/float(file_info[3]-2))
            final_num_items_p+=file_info[3]
            final_num_items_p_l.append(file_info[3])
            occu_ratio_p+=file_info[4]
# print(place_p_time,final_num_items_p,occu_ratio_p)

# place_p_time = place_p_time/final_num_items_p
# final_num_items_p = final_num_items_p/10
# print(place_p_time,final_num_items_p,occu_ratio_p/10)
print('final num item',final_num_items_p_l)
print('per item time', place_p_time_l)
print('############################### '+str(i)+' policy')            
print('avg final num item',np.mean(np.array(final_num_items_p_l)))
print('avg per item time', np.mean(np.array(place_p_time_l)))
print('############################## range #########')
print('range final num', range_std(final_num_items_p_l))
print('range per item time',range_std(place_p_time_l))
print('############################## std #########')
print('std final num', np.std(np.array(final_num_items_p_l)))
print('std per item time',np.std(np.array(place_p_time_l)))

place_p_time = 0
final_num_items_p = 0
occu_ratio_p = 0
place_p_time_l = []
final_num_items_p_l = []

for i in range(21):
        file_list = os.listdir("placing_test2/")
        file_name = "dict_com_"+str(i+1)+".pkl"
        if file_name in file_list:
            fileObject2 = open('placing_test2/'+file_name, 'rb')
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            place_p_time+=file_info[2]
            place_p_time_l.append(file_info[2]/float(file_info[3]-2))
            final_num_items_p+=file_info[3]
            final_num_items_p_l.append(file_info[3])
            occu_ratio_p+=file_info[4]
print('############################### '+str(i)+' com')
print('final num item',final_num_items_p_l)
print('per item time', place_p_time_l)
print('############################### avg')
print('avg final num item',np.mean(np.array(final_num_items_p_l)))
print('avg per item time', np.mean(np.array(place_p_time_l)))
print('############################## range #########')
print('range final num', range_std(final_num_items_p_l))
print('range per item time',range_std(place_p_time_l))
print('############################## std #########')
print('std final num', np.std(np.array(final_num_items_p_l)))
print('std per item time',np.std(np.array(place_p_time_l)))
# print(place_p_time,final_num_items_p,occu_ratio_p)
# place_p_time = place_p_time/final_num_items_p
# final_num_items_p = final_num_items_p/10
# print(place_p_time,final_num_items_p,occu_ratio_p/10)