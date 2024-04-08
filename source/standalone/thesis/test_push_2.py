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
final_item_num = 0
pushing_steps_num = 0
added_item = 0
occu_ratio_p = 0
fallen_item_num = 0
per_item_per_pushing = 0
final_num_item_l = []
per_pushing_steps_num_l = []
added_item_l = []
fallen_item_num_l = []
total_item_l = []
raw_added_item_l = []
raw_per_push_l = []
for i in range(20):
        file_list = os.listdir("placing_test/")
        file_name = "pushing_policy_"+str(i+1)+".pkl"
        if file_name in file_list:
            fileObject2 = open('placing_test/'+file_name, 'rb')
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            final_item_num+=file_info[0]
            final_num_item_l.append(file_info[0])
            fallen_item_num+=file_info[1]
            fallen_item_num_l.append(file_info[1])
            pushing_steps_num+=file_info[2]
            added_item += file_info[0] - file_info[4]
            added_item_l.append(file_info[0] - file_info[4])
            total_item_l.append(file_info[0])
            raw_added_item_l.append(file_info[0] - file_info[4]+file_info[1])
            if (file_info[0] - file_info[4])==0:
                per_pushing_steps_num_l.append(file_info[2])
            else:
                per_pushing_steps_num_l.append(abs(float(file_info[2])/(file_info[0] - file_info[4])))
            if (file_info[0] - file_info[4]+file_info[1])==0:
                raw_per_push_l.append(file_info[2])
            else:

                raw_per_push_l.append(abs(float(file_info[2])/(file_info[0] - file_info[4]+file_info[1])))
            after_oc = file_info[3].copy()
            pre_oc = file_info[5].copy()
            after_oc[np.where(after_oc==1)] = 0
            pre_oc[np.where(pre_oc==1)] = 0
            occu_ratio_p += (np.sum(after_oc[6:56,6:56]) - np.sum(np.sum(pre_oc[6:56,6:56])))/2.0
            print('final_item_num',file_info[0])
            print('added num',file_info[0] - file_info[4])
            print('fallen num',file_info[1])
            print('steps',file_info[2])
print('final num items',final_num_item_l)
print('fallen num items',fallen_item_num_l)
print('added num items', added_item_l)
print('per pushing',per_pushing_steps_num_l)
print('############################### '+str(i)+' policy')
print('avg final num items',np.mean(np.array(final_num_item_l)))
print('avg fallen num items',np.mean(np.array(fallen_item_num_l)))
print('avg added num items', np.mean(np.array(added_item_l)))
print('avg raw added num items', np.mean(np.array(raw_added_item_l)))
print('avg final num items', np.mean(np.array(total_item_l)))
print('avg per pushing',np.mean(np.array(per_pushing_steps_num_l)))
print('avg  raw per pushing',np.mean(np.array(raw_per_push_l)))
print('########################################### range ######################')
print('range final num items',range_std(final_num_item_l))
print('range fallen num items',range_std(fallen_item_num_l))
print('range added num items', range_std(added_item_l))
print('range raw added num items', range_std(raw_added_item_l))
print('range final num items', range_std(total_item_l))
print('range per pushing',range_std(per_pushing_steps_num_l))
print('range raw per pushing',range_std(raw_per_push_l))
print('########################################### std ######################')
print('std final num items',np.std(np.array(final_num_item_l)))
print('std fallen num items',np.std(np.array(fallen_item_num_l)))
print('std added num items', np.std(np.array(added_item_l)))
print('std raw added num items', np.std(np.array(raw_added_item_l)))
print('std final num items', np.std(np.array(total_item_l)))
print('std per pushing',np.std(np.array(per_pushing_steps_num_l)))
print('std  raw per pushing',np.std(np.array(raw_per_push_l)))
print('########################################### compare ######################')
# print('final_number_of_items',final_item_num/10.0)
# print('fallen_number_of_items',fallen_item_num/10.0)
# print('added item num',added_item/10.0)
# print('pushing steps vs placed item num',pushing_steps_num/float(added_item))
# print('oc ratio increase',occu_ratio_p)
# print('oc ratio increase',occu_ratio_p/25000.0)
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
final_num_item_l = []
per_pushing_steps_num_l = []
added_item_l = []
fallen_item_num_l = []
total_item_l = []
raw_added_item_l = []
raw_per_push_l = []
for i in range(20):
        file_list = os.listdir("placing_test/")
        file_name = "pushing_policy_com_"+str(i+1)+".pkl"
        if file_name in file_list:
            fileObject2 = open('placing_test/'+file_name, 'rb')
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            final_item_num+=file_info[0]
            final_num_item_l.append(file_info[0])
            fallen_item_num+=file_info[1]
            fallen_item_num_l.append(file_info[1])
            pushing_steps_num+=file_info[2]
            added_item += file_info[0] - file_info[4]
            added_item_l.append(file_info[0] - file_info[4])
            total_item_l.append(file_info[0])
            raw_added_item_l.append(file_info[0] - file_info[4]+file_info[1])
            if (file_info[0] - file_info[4])==0:
                per_pushing_steps_num_l.append(file_info[2])
            else:
                per_pushing_steps_num_l.append(abs(float(file_info[2])/(file_info[0] - file_info[4])))
            if (file_info[0] - file_info[4]+file_info[1])==0:
                raw_per_push_l.append(file_info[2])
            else:

                raw_per_push_l.append(abs(float(file_info[2])/(file_info[0] - file_info[4]+file_info[1])))
            after_oc = file_info[3].copy()
            pre_oc = file_info[5].copy()
            after_oc[np.where(after_oc==1)] = 0
            pre_oc[np.where(pre_oc==1)] = 0
            occu_ratio_p += (np.sum(after_oc[6:56,6:56]) - np.sum(np.sum(pre_oc[6:56,6:56])))/2.0
            print('final_item_num',file_info[0])
            print('added num',file_info[0] - file_info[4])
            print('fallen num',file_info[1])
            print('steps',file_info[2])
print('final num items',final_num_item_l)
print('fallen num items',fallen_item_num_l)
print('added num items', added_item_l)
print('per pushing',per_item_per_pushing)
print('############################### '+str(i)+' com')
print('avg final num items',np.mean(np.array(final_num_item_l)))
print('avg fallen num items',np.mean(np.array(fallen_item_num_l)))
print('avg added num items', np.mean(np.array(added_item_l)))
print('avg raw added num items', np.mean(np.array(raw_added_item_l)))
print('avg final num items', np.mean(np.array(total_item_l)))
print('avg per pushing',np.mean(np.array(per_pushing_steps_num_l)))
print('avg  raw per pushing',np.mean(np.array(raw_per_push_l)))
print('########################################### range ######################')
print('range final num items',range_std(final_num_item_l))
print('range fallen num items',range_std(fallen_item_num_l))
print('range added num items', range_std(added_item_l))
print('range raw added num items', range_std(raw_added_item_l))
print('range final num items', range_std(total_item_l))
print('range per pushing',range_std(per_pushing_steps_num_l))
print('range raw per pushing',range_std(raw_per_push_l))
print('########################################### std ######################')
print('std final num items',np.std(np.array(final_num_item_l)))
print('std fallen num items',np.std(np.array(fallen_item_num_l)))
print('std added num items', np.std(np.array(added_item_l)))
print('std raw added num items', np.std(np.array(raw_added_item_l)))
print('std final num items', np.std(np.array(total_item_l)))
print('std per pushing',np.std(np.array(per_pushing_steps_num_l)))
print('std  raw per pushing',np.std(np.array(raw_per_push_l)))

final_item_num = 0
pushing_steps_num = 0
added_item = 0
occu_ratio_p = 0
fallen_item_num = 0
per_item_per_pushing = 0
final_num_item_l = []
per_pushing_steps_num_l = []
added_item_l = []
fallen_item_num_l = []
total_item_l = []
raw_added_item_l = []
raw_per_push_l = []
for i in range(20):
        file_list = os.listdir("placing_test/")
        file_name = "pushing_FCN_mask_"+str(i+1)+".pkl"
        print(file_name)
        if file_name in file_list:
            fileObject2 = open('placing_test/'+file_name, 'rb')
            
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            final_item_num+=file_info[0]
            final_num_item_l.append(file_info[0])
            fallen_item_num+=file_info[1]
            fallen_item_num_l.append(file_info[1])
            pushing_steps_num+=file_info[2]
            added_item += file_info[0] - file_info[4]
            added_item_l.append(file_info[0] - file_info[4])
            total_item_l.append(file_info[0])
            raw_added_item_l.append(file_info[0] - file_info[4]+file_info[1])
            if (file_info[0] - file_info[4])==0:
                per_pushing_steps_num_l.append(file_info[2])
            else:
                per_pushing_steps_num_l.append(abs(float(file_info[2])/(file_info[0] - file_info[4])))
            if (file_info[0] - file_info[4]+file_info[1])==0:
                raw_per_push_l.append(file_info[2])
            else:

                raw_per_push_l.append(abs(float(file_info[2])/(file_info[0] - file_info[4]+file_info[1])))
            after_oc = file_info[3].copy()
            pre_oc = file_info[5].copy()
            after_oc[np.where(after_oc==1)] = 0
            pre_oc[np.where(pre_oc==1)] = 0
            occu_ratio_p += (np.sum(after_oc[6:56,6:56]) - np.sum(np.sum(pre_oc[6:56,6:56])))/2.0
            print('final_item_num',file_info[0])
            print('added num',file_info[0] - file_info[4])
            print('fallen num',file_info[1])
            print('steps',file_info[2])
print('final num items',final_num_item_l)
print('fallen num items',fallen_item_num_l)
print('added num items', added_item_l)
print('per pushing',per_item_per_pushing)
print('############################### '+str(i)+' FCN mask')
print('avg final num items',np.mean(np.array(final_num_item_l)))
print('avg fallen num items',np.mean(np.array(fallen_item_num_l)))
print('avg added num items', np.mean(np.array(added_item_l)))
print('avg raw added num items', np.mean(np.array(raw_added_item_l)))
print('avg final num items', np.mean(np.array(total_item_l)))
print('avg per pushing',np.mean(np.array(per_pushing_steps_num_l)))
print('avg  raw per pushing',np.mean(np.array(raw_per_push_l)))
print('########################################### range ######################')
print('range final num items',range_std(final_num_item_l))
print('range fallen num items',range_std(fallen_item_num_l))
print('range added num items', range_std(added_item_l))
print('range raw added num items', range_std(raw_added_item_l))
print('range final num items', range_std(total_item_l))
print('range per pushing',range_std(per_pushing_steps_num_l))
print('range raw per pushing',range_std(raw_per_push_l))
print('########################################### std ######################')
print('std final num items',np.std(np.array(final_num_item_l)))
print('std fallen num items',np.std(np.array(fallen_item_num_l)))
print('std added num items', np.std(np.array(added_item_l)))
print('std raw added num items', np.std(np.array(raw_added_item_l)))
print('std final num items', np.std(np.array(total_item_l)))
print('std per pushing',np.std(np.array(per_pushing_steps_num_l)))
print('std  raw per pushing',np.std(np.array(raw_per_push_l)))


final_item_num = 0
pushing_steps_num = 0
added_item = 0
occu_ratio_p = 0
fallen_item_num = 0
per_item_per_pushing = 0
final_num_item_l = []
per_pushing_steps_num_l = []
added_item_l = []
fallen_item_num_l = []
total_item_l = []
raw_added_item_l = []
raw_per_push_l = []
for i in range(40,60):
        file_list = os.listdir("placing_test/")
        file_name = "pushing_FCN_withoutmask_"+str(i+1)+".pkl"
        print(file_name)
        if file_name in file_list:
            fileObject2 = open('placing_test/'+file_name, 'rb')
            
            file_info=  pickle.load(fileObject2)
            fileObject2.close()
            final_item_num+=file_info[0]
            final_num_item_l.append(file_info[0])
            fallen_item_num+=file_info[1]
            fallen_item_num_l.append(file_info[1])
            pushing_steps_num+=file_info[2]
            added_item += file_info[0] - file_info[4]
            added_item_l.append(file_info[0] - file_info[4])
            total_item_l.append(file_info[0])
            raw_added_item_l.append(file_info[0] - file_info[4]+file_info[1])
            if (file_info[0] - file_info[4])==0:
                per_pushing_steps_num_l.append(file_info[2])
            else:
                per_pushing_steps_num_l.append(abs(float(file_info[2])/(file_info[0] - file_info[4])))
            if (file_info[0] - file_info[4]+file_info[1])==0:
                raw_per_push_l.append(file_info[2])
            else:

                raw_per_push_l.append(abs(float(file_info[2])/(file_info[0] - file_info[4]+file_info[1])))
            after_oc = file_info[3].copy()
            pre_oc = file_info[5].copy()
            after_oc[np.where(after_oc==1)] = 0
            pre_oc[np.where(pre_oc==1)] = 0
            occu_ratio_p += (np.sum(after_oc[6:56,6:56]) - np.sum(np.sum(pre_oc[6:56,6:56])))/2.0
            print('final_item_num',file_info[0])
            print('added num',file_info[0] - file_info[4])
            print('fallen num',file_info[1])
            print('steps',file_info[2])
print('final num items',final_num_item_l)
print('fallen num items',fallen_item_num_l)
print('added num items', added_item_l)
print('per pushing',per_item_per_pushing)
print('############################### '+str(i)+' FCN without mask')
print('avg final num items',np.mean(np.array(final_num_item_l)))
print('avg fallen num items',np.mean(np.array(fallen_item_num_l)))
print('avg added num items', np.mean(np.array(added_item_l)))
print('avg raw added num items', np.mean(np.array(raw_added_item_l)))
print('avg final num items', np.mean(np.array(total_item_l)))
print('avg per pushing',np.mean(np.array(per_pushing_steps_num_l)))
print('avg  raw per pushing',np.mean(np.array(raw_per_push_l)))
print('########################################### range ######################')
print('range final num items',range_std(final_num_item_l))
print('range fallen num items',range_std(fallen_item_num_l))
print('range added num items', range_std(added_item_l))
print('range raw added num items', range_std(raw_added_item_l))
print('range final num items', range_std(total_item_l))
print('range per pushing',range_std(per_pushing_steps_num_l))
print('range raw per pushing',range_std(raw_per_push_l))
print('########################################### std ######################')
print('std final num items',np.std(np.array(final_num_item_l)))
print('std fallen num items',np.std(np.array(fallen_item_num_l)))
print('std added num items', np.std(np.array(added_item_l)))
print('std raw added num items', np.std(np.array(raw_added_item_l)))
print('std final num items', np.std(np.array(total_item_l)))
print('std per pushing',np.std(np.array(per_pushing_steps_num_l)))
print('std  raw per pushing',np.std(np.array(raw_per_push_l)))