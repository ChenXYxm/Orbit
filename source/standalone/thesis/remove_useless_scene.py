import pickle as pkl
import os

file_path = "un_satisfied_scenes.pkl"
f_load = open(file_path,'rb')

file_names=pkl.load(f_load)
f_load.close()
print(file_names)
def remove_files(file_list):
    for file_name_i in file_list:
        try:
            file_name = "generated_table2/"+file_name_i
            os.remove(file_name)
            print(f"File '{file_name}' removed successfully.")
        except FileNotFoundError:
            print(f"File '{file_name}' not found.")
        except Exception as e:
            print(f"Error removing file '{file_name}': {e}")

remove_files(file_names)