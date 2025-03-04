# Face Mask Detection

## Overview
This project is a Deep Learning-based Face Mask Detection system that can detect whether a person is wearing a mask or not using a trained Convolutional Neural Network (CNN). The model is built using TensorFlow and Keras and utilizes OpenCV for real-time face detection.

## Features
- Detects whether a person is wearing a mask or not in real-time.
- Uses TensorFlow and Keras for model training.
- Utilizes OpenCV for face detection from a webcam.
- Trained on a dataset containing images of masked and unmasked faces.

## Dataset
The dataset consists of two categories:
- **With Mask**: Images of people wearing masks.
- **Without Mask**: Images of people without masks.

Dataset Structure:
```
 dataset/
   train/
     with_mask/
     without_mask/
   val/
     with_mask/
     without_mask/
```

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Face-Mask-Detection.git
   cd Face-Mask-Detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or prepare a dataset and ensure it follows the correct structure.

## Model Training
To train the model, run:
```bash
python train.py
```
This script will train the model and save it as `mask_detector.h5`.

## Running the Face Mask Detector
To run the real-time face mask detection, execute:
```bash
python detect_mask.py
```
This will open the webcam and start detecting faces with or without masks.

## Results
- The model achieves **~95% validation accuracy**.
- The system can detect masked and unmasked faces in real-time.

## Future Improvements
- Enhance accuracy by using a larger and more diverse dataset.
- Deploy the model as a web application using Flask or FastAPI.
- Optimize the model for mobile and embedded systems.

## Contributing
Feel free to fork this repository and make improvements. If you have any suggestions, submit a pull request!

## License
This project is licensed under the MIT License.

## Contact
For any queries, contact:
- **Name:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [Your GitHub Profile](https://github.com/yourusername/)
