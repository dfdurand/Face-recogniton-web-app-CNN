import cv2
import os
# from glob
import glob
# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')

# Define the input and output directories
input_directory = 'path/to/dataset/folder_input'  # *** remember to change label ****
output_directory = 'path/to/dataset/folder_output'  # **** remember to change label ****

# # Create the output directory if it doesn't exist

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to detect and save faces in an image
def detect_and_save_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
   
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop the face region from the image
            face = image[y:y+h, x:x+w]
            # Save the face to the output directory
            output_path = os.path.join(output_directory, f'presidentx_{os.path.basename(image_path)}')  #**** remember to change label ****
            cv2.imwrite(output_path, face)
            # i += 1

# List all image files in the input directory with .jpg or .jpeg extensions
image_files = glob.glob(os.path.join(input_directory, '*.jpg')) + glob.glob(os.path.join(input_directory, '*.jpeg'))

# Process each image in the input directory
for image_path in image_files:
    detect_and_save_faces(image_path)
    

print("Face detection and saving completed.")