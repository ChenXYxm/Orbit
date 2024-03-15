import pickle as pkl
import os

file_path = "un_satisfied_scenes.pkl"
f_load = open(file_path,'rb')

# file_names=pkl.load(f_load)
file_names = ['dict_725.pkl', 'dict_305.pkl', 'old_dict_1168.pkl', 'old_dict_470.pkl', 'dict_371.pkl', 'old_dict_1439.pkl', 'dict_456.pkl', 'old_dict_1060.pkl', 'dict_780.pkl', 'old_dict_44.pkl', 'dict_939.pkl', 'dict_856.pkl', 'old_dict_1152.pkl', 'dict_58.pkl', 'old_dict_397.pkl', 'dict_113.pkl', 'dict_787.pkl', 'dict_101.pkl', 'old_dict_513.pkl', 'dict_377.pkl']

f_load.close()
print(file_names)
def remove_files(file_list):
    for file_name_i in file_list:
        try:
            file_name = "train_table4/"+file_name_i
            os.remove(file_name)
            print(f"File '{file_name}' removed successfully.")
        except FileNotFoundError:
            print(f"File '{file_name}' not found.")
        except Exception as e:
            print(f"Error removing file '{file_name}': {e}")

remove_files(file_names)