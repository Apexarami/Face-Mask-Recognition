import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("mask_detector.keras")

# ✅ Check class order
class_indices = {"Mask": 0, "No Mask": 1}  # Update this if different
print(f"Class Indices: {class_indices}")

# ✅ Load image
img_path = 'C:/Users/91635/PycharmProjects/Face_MaskDetection/dataset/train/without_mask/2.png'

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
import os
if os.path.exists(img_path):
    print("File exists!")
else:
    print(f"File not found: {img_path}")

# ✅ Make predictions
predictions = model.predict(img_array)

# ✅ Use correct probability index
mask_prob = predictions[0][class_indices["Mask"]]

# ✅ Apply confidence threshold
confidence_threshold = 0.50
label = "Mask" if mask_prob > confidence_threshold else "No Mask"

print(f"Predicted: {label} (Confidence: {mask_prob:.2f})")
