import cv2
import glob
import os 

# Set the folder path
# folder_path = '/path/to/folder'
# Define the input and output directories

output_directory = 'path/to/dataset/folder_output/presidentx'

# Create a list of image files

image_files = glob.glob(os.path.join(output_directory, '*.jpg')) + glob.glob(os.path.join(output_directory, '*.jpeg'))

# Initialize the index
index = 0

# Loop through the list of images and rename them
for file in image_files:
    # Read the image using OpenCV
    img = cv2.imread(file)

    # Increment the index
    index += 1

    # Create a new file name using the format "user+index"
    new_file_name = os.path.join(output_directory, f'presidentx_{index}.jpg') # make sure to change label

    # Rename the image file
    cv2.imwrite(new_file_name, img)

    # Delete the original image file
    os.remove(file)

    # Print the new file name
    print(new_file_name)