import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

# Load trained model
model = tf.keras.models.load_model("mask_detector.keras")  # Load the pre-trained mask detection model

# Check class indices for mask and no mask
class_indices = {"Mask": 0, "No Mask": 1}  # Class labels in the model: Mask = 0, No Mask = 1
print(f"Class Indices: {class_indices}")

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)  # Start video capture from webcam (0 is the default camera)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")  # If webcam is not opened successfully, print error
    exit()  # Exit the program if webcam fails to open


# Function to preprocess and predict mask in the given frame
def predict_mask(frame):
    # Convert the frame from BGR (OpenCV default) to RGB (as the model expects RGB input)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize image to the size the model expects (224x224)
    img = cv2.resize(img, (224, 224))

    # Convert image to array and normalize the pixel values to the range [0, 1]
    img_array = image.img_to_array(img) / 255.0

    # Add batch dimension as the model expects a batch of images
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    predictions = model.predict(img_array)

    # Get the probability of the mask class
    mask_prob = predictions[0][class_indices["Mask"]]

    # Apply confidence threshold to decide the label (Mask or No Mask)
    confidence_threshold = 0.50
    label = "Mask" if mask_prob > confidence_threshold else "No Mask"  # Label based on threshold
    return label, mask_prob  # Return the predicted label and probability


# Loop to read frames from the video stream
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")  # If the frame could not be captured, print error
        break  # Exit the loop if frame capture fails

    # Convert the frame to grayscale for face detection (gray scale image for efficiency)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load pre-trained face cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detect faces using the cascade classifier

    # Loop over each detected face and predict if it's wearing a mask
    for (x, y, w, h) in faces:
        # Crop the face from the frame for prediction
        face = frame[y:y + h, x:x + w]

        # Get the prediction (mask or no mask) for the detected face
        label, mask_prob = predict_mask(face)

        # Define the color for bounding box and label text
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # Green for Mask, Red for No Mask

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display the label (Mask or No Mask) and probability on the frame near the face
        label_text = f"{label}: {mask_prob * 100:.2f}%"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display the frame with bounding boxes and predictions
    cv2.imshow("Frame", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press
        break  # Exit the loop if 'q' is pressed

# Release the video capture and close OpenCV windows
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close any OpenCV windows
