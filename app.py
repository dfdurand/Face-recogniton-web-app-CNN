import streamlit as st
import cv2
import numpy as np
import cvzone

from tensorflow.keras.models import load_model

# Load the pre-trained CNN model
model = load_model('models/presidents.h5')
#class names or label 
class_names = ['J. BIDEN', 'P. BIYA', 'P. KAGAME', 'E. MACRON']
#label index        0         1          2             3

# Load the OpenCV face detection cascade classifier
face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')  # Replace with the path to your cascade classifier file



def read_image(image_file):
    img = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    st.title("Face Detection and Prediction App")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is None:
        st.text("Please upload an image file")

    if uploaded_image is not None:
        
        image = read_image(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5) #, minSize=(30, 30))
        print('nombres de face: ', len(faces))
        for (x, y, w, h) in faces:
            # Crop the detected face region
            face = image[y:y + h, x:x + w]

            # Resize the cropped face to (64, 64)
            face = cv2.resize(face, (64, 64))

            # Preprocess the image for model input 
            face = face.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

            # Make predictions using the model
            predictions = model.predict(np.expand_dims(face, axis=0))

            # Assuming you have 4 classes and want to get the class with the highest probability
            predicted_class = np.argmax(predictions) #index of predictions

            print("predicted class: ", predicted_class)

            label = class_names[predicted_class] # bind index -> label

            confidence = round(np.max(predictions) * 100, 2) #get the precison or accuracy of detection 
            print("confidence: ", confidence)

            print("label: ", label)

            # text shown up to bounding box
            final_label = label +' '+ str(confidence) + '%' 
            
            # Display the result with bounding box on the face
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
             # cv2.putText(image, final_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display class label
            # label = f"Class: {predicted_class}"  # Modify this label based on your class labels
           
            image = cvzone.cornerRect(image, (x, y, w, h))
            image, bbox = cvzone.putTextRect(
            image, final_label, (x-10, y-20),  # Image and starting position of the rectangle
            scale=2, thickness=2,  # Font scale and thickness
            colorT=(255, 255, 255), colorR=(255, 0, 255),  # Text color and Rectangle color
            font=cv2.FONT_HERSHEY_PLAIN,  # Font type
            offset=10,  # Offset of text inside the rectangle
            border=5, colorB=(0, 255, 0)  # Border thickness and color
            )


        # Display the image with bounding boxes and predictions
        st.image(image, caption="Result Image", use_column_width=True)

if __name__ == "__main__":
    main()
