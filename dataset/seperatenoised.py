import os
import shutil

# List of your main folders containing voices
main_folders = ['angry', 'bored', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Base directory where the main folders are located
base_dir = '/Users/ishananand/Desktop/ser/combined_dataset/'

# Loop through each folder in the list
for folder in main_folders:
    folder_path = os.path.join(base_dir, folder)
    
    # Create the corresponding _noised folder
    # noised_folder_path = os.path.join(base_dir, f'{folder}_noised')
    # if not os.path.exists(noised_folder_path):
    #     os.makedirs(noised_folder_path)
    
    # Now copy all the files from the original folder to the new _noised folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):  # Only process files (skip directories)
            # Create the full path for the new file in the _noised folder
            if("_noised" in file_path):
                print(filename)
                os.remove(file_path)

            # new_file_path = os.path.join(noised_folder_path, filename)
            
            # # Copy the file to the new _noised folder
            # shutil.copy(file_path, new_file_path)
            
            # # OPTIONAL: Add noise to the file (if you have a method for adding noise)
            # # For now, the script just copies the files.
            # # You can integrate a noise-adding function here.
            
            # print(f"Copied {filename} to {noised_folder_path}")
