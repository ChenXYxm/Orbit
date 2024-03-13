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
final_item_num = 0
pushing_steps_num = 0
added_item = 0
occu_ratio_p = 0
fallen_item_num = 0
per_item_per_pushing = 0
for i in range(10):
        file_list = os.listdir("placing_test/")
        file_name = "pushing_policy_new_"+str(i+1)+".pkl"
        if file_name in file_list:
            fileObject2 = open('placing_test/'+file_name, 'rb')
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            final_item_num+=file_info[0]
            fallen_item_num+=file_info[1]
            pushing_steps_num+=file_info[2]
            added_item += file_info[0] - file_info[4]
            after_oc = file_info[3].copy()
            pre_oc = file_info[5].copy()
            after_oc[np.where(after_oc==1)] = 0
            pre_oc[np.where(pre_oc==1)] = 0
            occu_ratio_p += (np.sum(after_oc[6:56,6:56]) - np.sum(np.sum(pre_oc[6:56,6:56])))/2.0
            print('final_item_num',file_info[0])
            print('added num',file_info[0] - file_info[4])
            print('fallen num',file_info[1])
            print('steps',file_info[2])
print('final_number_of_items',final_item_num/10.0)
print('fallen_number_of_items',fallen_item_num/10.0)
print('added item num',added_item/10.0)
print('pushing steps vs placed item num',pushing_steps_num/float(added_item))
print('oc ratio increase',occu_ratio_p)
print('oc ratio increase',occu_ratio_p/25000.0)
# print(place_p_time,final_num_items_p,occu_ratio_p)
# place_p_time = place_p_time/final_num_items_p
# final_num_items_p = final_num_items_p/10
# print(place_p_time,final_num_items_p,occu_ratio_p/10)
# place_p_time = 0
# final_num_items_p = 0
# occu_ratio_p = 0
final_item_num = 0
pushing_steps_num = 0
added_item = 0
occu_ratio_p = 0
fallen_item_num = 0
per_item_per_pushing = 0
for i in range(10):
        file_list = os.listdir("placing_test/")
        file_name = "pushing_policy_com_"+str(i+1)+".pkl"
        if file_name in file_list:
            fileObject2 = open('placing_test/'+file_name, 'rb')
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            final_item_num+=file_info[0]
            fallen_item_num+=file_info[1]
            pushing_steps_num+=file_info[2]
            added_item += file_info[0] - file_info[4]
            after_oc = file_info[3].copy()
            pre_oc = file_info[5].copy()
            after_oc[np.where(after_oc==1)] = 0
            pre_oc[np.where(pre_oc==1)] = 0
            occu_ratio_p += (np.sum(after_oc[6:56,6:56]) - np.sum(np.sum(pre_oc[6:56,6:56])))/2.0
            print('final_item_num',file_info[0])
            print('added num',file_info[0] - file_info[4])
            print('fallen num',file_info[1])
            print('steps',file_info[2])
print('final_number_of_items',final_item_num/10.0)
print('fallen_number_of_items',fallen_item_num/10.0)
print('added item num',added_item/10.0)
print('pushing steps vs placed item num',pushing_steps_num/float(added_item))
print('oc ratio increase',occu_ratio_p/25000.0)