import os
import shutil

# Define the path to the temp folder
TEMP_FOLDER = '/temp'

# Function to delete the contents of the temp folder
def clear_temp_folder_and_reload():
    if os.path.exists(TEMP_FOLDER):
        for filename in os.listdir(TEMP_FOLDER):
            file_path = os.path.join(TEMP_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

# Run the function to clear the temp folder on reload
# clear_temp_folder_and_reload()
