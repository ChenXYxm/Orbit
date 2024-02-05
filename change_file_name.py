import os

def rename_files(directory_path, new_prefix,new_directory_path):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Iterate through the files and rename them
    for filename in files:
        # Create the new filename by adding the new_prefix
        new_filename = new_prefix+filename

        # Construct the full paths for the old and new filenames
        old_path = os.path.join(directory_path, filename)
        new_path = os.path.join(new_directory_path, new_filename)

        # Rename the file
        os.rename(old_path, new_path)

# Example usage:
directory_path = '/home/chenxiny/orbit/Orbit/generated_table2/'
new_directory_path ='/home/chenxiny/orbit/Orbit/train_table4/'
new_prefix = 'old_'
rename_files(directory_path, new_prefix,new_directory_path)
